import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch
from skimage import exposure


class ImageProcessor:
    """Handles image preprocessing and post-processing operations."""

    @staticmethod
    def resize_image(image: Image.Image, target_size: int = 1024) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: Input PIL Image
            target_size: Target size for the longest edge

        Returns:
            Resized PIL Image
        """
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def detect_edges(image: Image.Image, low_threshold: int = 100,
                     high_threshold: int = 200) -> Image.Image:
        """
        Detect edges using Canny edge detection.

        Args:
            image: Input PIL Image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny

        Returns:
            Edge map as PIL Image
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert back to PIL Image
        return Image.fromarray(edges)

    @staticmethod
    def extract_lighting_features(image: Image.Image) -> dict:
        """
        Extract lighting characteristics from an image.

        Args:
            image: Input PIL Image

        Returns:
            Dictionary with lighting features
        """
        img_array = np.array(image)

        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        return {
            'mean_brightness': np.mean(l_channel),
            'std_brightness': np.std(l_channel),
            'mean_a': np.mean(a_channel),
            'mean_b': np.mean(b_channel)
        }

    @staticmethod
    def match_lighting(source: Image.Image, target: Image.Image,
                       alpha: float = 0.7) -> Image.Image:
        """
        Transfer lighting characteristics from target to source.

        Args:
            source: Source image to adjust
            target: Target image to match
            alpha: Blending factor (0-1)

        Returns:
            Color-matched PIL Image
        """
        source_array = np.array(source)
        target_array = np.array(target)

        # Convert to LAB
        source_lab = cv2.cvtColor(source_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_array, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Match histogram on L channel
        source_lab[:, :, 0] = ImageProcessor._match_histogram_channel(
            source_lab[:, :, 0],
            target_lab[:, :, 0]
        )

        # Blend with original
        source_lab = alpha * source_lab + (1 - alpha) * cv2.cvtColor(
            source_array, cv2.COLOR_RGB2LAB
        ).astype(np.float32)

        # Convert back to RGB
        result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        return Image.fromarray(result)

    @staticmethod
    def _match_histogram_channel(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of source channel to target channel."""
        # Compute CDFs
        source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        target_hist, _ = np.histogram(target.flatten(), 256, [0, 256])

        source_cdf = source_hist.cumsum()
        target_cdf = target_hist.cumsum()

        # Normalize
        source_cdf = source_cdf / source_cdf[-1]
        target_cdf = target_cdf / target_cdf[-1]

        # Create mapping
        mapping = np.zeros(256)
        for i in range(256):
            diff = np.abs(target_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)

        # Apply mapping
        matched = mapping[source.astype(int)]

        return matched

    @staticmethod
    def create_depth_map(image: Image.Image) -> Image.Image:
        """
        Create a simple depth map estimation.
        For production, consider using MiDaS or other depth estimation models.

        Args:
            image: Input PIL Image

        Returns:
            Depth map as PIL Image
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Simple depth estimation using blur gradient
        # This is a placeholder - for production use MiDaS or similar
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        depth = cv2.Laplacian(blurred, cv2.CV_64F)
        depth = np.uint8(np.absolute(depth))

        # Normalize
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        return Image.fromarray(depth)

    @staticmethod
    def enhance_contrast(image: Image.Image, clip_limit: float = 2.0) -> Image.Image:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Input PIL Image
            clip_limit: Clipping limit for CLAHE

        Returns:
            Enhanced PIL Image
        """
        img_array = np.array(image)

        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(result)

    @staticmethod
    def blend_images(base: Image.Image, overlay: Image.Image,
                     mask: Optional[Image.Image] = None,
                     alpha: float = 0.5) -> Image.Image:
        """
        Blend two images together with optional mask.

        Args:
            base: Base image
            overlay: Overlay image
            mask: Optional mask (white = overlay, black = base)
            alpha: Blending factor if no mask provided

        Returns:
            Blended PIL Image
        """
        base_array = np.array(base).astype(np.float32)
        overlay_array = np.array(overlay).astype(np.float32)

        if mask is not None:
            mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0
            if len(base_array.shape) == 3:
                mask_array = np.expand_dims(mask_array, axis=2)
            result = overlay_array * mask_array + base_array * (1 - mask_array)
        else:
            result = overlay_array * alpha + base_array * (1 - alpha)

        return Image.fromarray(result.astype(np.uint8))
