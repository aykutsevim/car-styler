"""
Unit tests for the Car Styler Service.
Run with: pytest tests/
"""

import pytest
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.image_processor import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(img_array)

    def test_resize_image(self, sample_image):
        """Test image resizing."""
        processor = ImageProcessor()
        resized = processor.resize_image(sample_image, target_size=1024)

        assert resized.width == 1024 or resized.height == 1024
        assert isinstance(resized, Image.Image)

    def test_detect_edges(self, sample_image):
        """Test edge detection."""
        processor = ImageProcessor()
        edges = processor.detect_edges(sample_image)

        assert isinstance(edges, Image.Image)
        assert edges.mode == 'L'  # Grayscale

    def test_extract_lighting_features(self, sample_image):
        """Test lighting feature extraction."""
        processor = ImageProcessor()
        features = processor.extract_lighting_features(sample_image)

        assert 'mean_brightness' in features
        assert 'std_brightness' in features
        assert 'mean_a' in features
        assert 'mean_b' in features
        assert isinstance(features['mean_brightness'], float)

    def test_enhance_contrast(self, sample_image):
        """Test contrast enhancement."""
        processor = ImageProcessor()
        enhanced = processor.enhance_contrast(sample_image)

        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == sample_image.size

    def test_blend_images(self, sample_image):
        """Test image blending."""
        processor = ImageProcessor()

        # Create a second image
        img_array2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image2 = Image.fromarray(img_array2)

        blended = processor.blend_images(sample_image, image2, alpha=0.5)

        assert isinstance(blended, Image.Image)
        assert blended.size == sample_image.size

    def test_create_depth_map(self, sample_image):
        """Test depth map creation."""
        processor = ImageProcessor()
        depth = processor.create_depth_map(sample_image)

        assert isinstance(depth, Image.Image)
        assert depth.mode == 'L'


def test_image_processor_initialization():
    """Test ImageProcessor can be initialized."""
    processor = ImageProcessor()
    assert processor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
