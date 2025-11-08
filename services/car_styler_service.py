import torch
from PIL import Image
from typing import Optional, Dict
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import AutoencoderKL
import numpy as np
import logging

from services.image_processor import ImageProcessor
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarStylerService:
    """
    Main service for applying car styling products to car images.
    Uses ControlNet + Stable Diffusion XL for structure-preserving generation.
    """

    def __init__(self):
        """Initialize the car styler service with models."""
        self.device = settings.device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CarStylerService on device: {self.device}")

        self.image_processor = ImageProcessor()
        self.pipeline: Optional[StableDiffusionXLControlNetPipeline] = None

    def load_models(self):
        """Load all required models."""
        if self.pipeline is not None:
            logger.info("Models already loaded")
            return

        logger.info("Loading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            settings.controlnet_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=settings.model_cache_dir
        )

        logger.info("Loading Stable Diffusion XL pipeline...")
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            settings.sd_model,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=settings.model_cache_dir
        )

        self.pipeline.to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()

        logger.info("Models loaded successfully")

    @staticmethod
    def _align_dimensions_to_vae(height: int, width: int, vae_scale_factor: int = 8) -> tuple:
        """
        Align dimensions to VAE requirements (multiples of 8).

        Args:
            height: Original height
            width: Original width
            vae_scale_factor: VAE requires dimensions to be multiples of this (default 8)

        Returns:
            Tuple of (aligned_height, aligned_width)
        """
        aligned_height = (height // vae_scale_factor) * vae_scale_factor
        aligned_width = (width // vae_scale_factor) * vae_scale_factor
        return aligned_height, aligned_width

    def apply_styling(
        self,
        car_image: Image.Image,
        product_image: Image.Image,
        product_description: str,
        preserve_background: bool = True,
        match_lighting: bool = True
    ) -> Dict[str, Image.Image]:
        """
        Apply car styling product to the car image.

        Args:
            car_image: Original car image
            product_image: Styling product reference image
            product_description: Text description of the product and how to apply it
            preserve_background: Whether to preserve the background
            match_lighting: Whether to match lighting conditions

        Returns:
            Dictionary containing:
                - 'result': Final styled car image
                - 'edges': Edge map used for control
                - 'intermediate': Intermediate generation result
        """
        if self.pipeline is None:
            self.load_models()

        logger.info("Starting car styling process...")

        # Step 1: Preprocess images
        logger.info("Preprocessing images...")
        car_image_resized = self.image_processor.resize_image(
            car_image, settings.image_size
        )
        product_image_resized = self.image_processor.resize_image(
            product_image, settings.image_size
        )

        # Align dimensions to VAE requirements (multiples of 8) before SDXL processing
        aligned_height, aligned_width = self._align_dimensions_to_vae(
            car_image_resized.height, car_image_resized.width
        )
        if (aligned_height, aligned_width) != (car_image_resized.height, car_image_resized.width):
            logger.info(f"Aligning dimensions to VAE requirements: {car_image_resized.size} -> ({aligned_width}, {aligned_height})")
            car_image_resized = car_image_resized.resize((aligned_width, aligned_height), Image.Resampling.LANCZOS)

        # Step 2: Extract edges from car for structure preservation
        logger.info("Extracting edge map for structure preservation...")
        edge_map = self.image_processor.detect_edges(
            car_image_resized,
            settings.canny_low_threshold,
            settings.canny_high_threshold
        )

        # Step 3: Extract lighting features
        lighting_features = None
        if match_lighting:
            logger.info("Analyzing lighting conditions...")
            lighting_features = self.image_processor.extract_lighting_features(
                car_image_resized
            )

        # Step 4: Create prompt for generation
        prompt = self._create_prompt(product_description, lighting_features)
        negative_prompt = self._create_negative_prompt()

        logger.info(f"Generation prompt: {prompt}")

        # Step 5: Generate styled car with ControlNet
        logger.info("Generating styled car image...")
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=edge_map,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                controlnet_conditioning_scale=settings.controlnet_conditioning_scale,
                height=car_image_resized.height,
                width=car_image_resized.width
            ).images[0]

        logger.info(f"Generated image size: {result.size}, Target size: {car_image_resized.size}")

        # Step 6: Post-processing
        logger.info("Applying post-processing...")

        # Match lighting if requested
        if match_lighting:
            result = self.image_processor.match_lighting(
                result, car_image_resized, alpha=0.6
            )

        # Enhance contrast
        result = self.image_processor.enhance_contrast(result, clip_limit=2.0)

        # Optionally blend with original to preserve background
        if preserve_background:
            result = self.image_processor.blend_images(
                car_image_resized, result, alpha=0.85
            )

        logger.info("Car styling complete!")

        return {
            'result': result,
            'edges': edge_map,
            'intermediate': result,
            'original_resized': car_image_resized
        }

    def _create_prompt(
        self,
        product_description: str,
        lighting_features: Optional[Dict] = None
    ) -> str:
        """
        Create a detailed prompt for image generation.

        Args:
            product_description: Description of the styling product
            lighting_features: Optional lighting characteristics

        Returns:
            Generated prompt string
        """
        base_prompt = f"A high-quality professional photograph of a car with {product_description}, "
        base_prompt += "photorealistic, detailed, sharp focus, 8k uhd, dslr quality, "

        if lighting_features:
            brightness = lighting_features['mean_brightness']
            if brightness > 150:
                base_prompt += "bright daylight, "
            elif brightness > 100:
                base_prompt += "natural lighting, "
            else:
                base_prompt += "studio lighting, "

        base_prompt += "automotive photography, professional car photo, realistic materials and reflections"

        return base_prompt

    def _create_negative_prompt(self) -> str:
        """Create negative prompt to avoid unwanted artifacts."""
        return (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "unrealistic, cartoon, anime, painting, drawing, sketch, "
            "low resolution, pixelated, watermark, text, jpeg artifacts, "
            "oversaturated, duplicate, malformed"
        )

    def batch_apply_styling(
        self,
        car_images: list[Image.Image],
        product_image: Image.Image,
        product_description: str,
        **kwargs
    ) -> list[Dict[str, Image.Image]]:
        """
        Apply styling to multiple car images in batch.

        Args:
            car_images: List of car images
            product_image: Styling product reference image
            product_description: Text description of the product
            **kwargs: Additional arguments passed to apply_styling

        Returns:
            List of result dictionaries for each car image
        """
        results = []
        for idx, car_image in enumerate(car_images):
            logger.info(f"Processing image {idx + 1}/{len(car_images)}")
            result = self.apply_styling(
                car_image, product_image, product_description, **kwargs
            )
            results.append(result)

        return results

    def unload_models(self):
        """Unload models from memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Models unloaded from memory")
