from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings and configuration."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model Configuration
    model_cache_dir: str = "./models"
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Stable Diffusion Models
    sd_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"

    # Generation Parameters
    image_size: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.8

    # Edge Detection Parameters
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200

    # SAM Model
    sam_checkpoint: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
