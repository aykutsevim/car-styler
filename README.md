# Car Styler API

An AI-powered service that applies car styling products to car images using advanced image-to-image generation models.

## Features

- **AI-Powered Styling**: Uses Stable Diffusion XL with ControlNet for photorealistic results
- **Edge Preservation**: Maintains car structure using Canny edge detection
- **Lighting Matching**: Automatically matches lighting conditions from the original image
- **Background Preservation**: Option to keep the original background intact
- **Batch Processing**: Process multiple car images in one request
- **RESTful API**: Easy-to-use HTTP endpoints with FastAPI

## Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **AI Models**:
  - Stable Diffusion XL (base model for generation)
  - ControlNet (structure preservation)
  - Canny Edge Detection (edge extraction)
- **Image Processing**: OpenCV, Pillow, NumPy, scikit-image
- **ML Framework**: PyTorch, Diffusers (Hugging Face)

## Architecture

```
┌─────────────────┐
│   Car Image     │
└────────┬────────┘
         │
         ├─────────────────────────────┐
         │                             │
         v                             v
┌────────────────┐          ┌─────────────────┐
│ Edge Detection │          │    Lighting     │
│  (Canny)       │          │    Analysis     │
└────────┬───────┘          └────────┬────────┘
         │                           │
         v                           v
┌────────────────────────────────────────────┐
│   ControlNet + Stable Diffusion XL        │
│   (Product Description as Prompt)          │
└────────────────┬───────────────────────────┘
                 │
                 v
┌────────────────────────────────────────────┐
│   Post-Processing                          │
│   - Lighting Matching                      │
│   - Contrast Enhancement                   │
│   - Background Blending                    │
└────────────────┬───────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────┐
│   Styled Car Image                       │
└──────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with at least 8GB VRAM (recommended) or CPU
- CUDA 11.8+ (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd carstyler
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
# Copy the example env file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your preferences
```

5. First run will download models (this can take 10-20 minutes):
   - Stable Diffusion XL (~7GB)
   - ControlNet (~3GB)

## Usage

### Starting the Server

```bash
python run.py
```

The API will be available at:
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Using the API

#### Single Image Styling

```bash
curl -X POST "http://localhost:8000/style" \
  -F "car_image=@path/to/car.jpg" \
  -F "product_image=@path/to/product.jpg" \
  -F "product_description=matte black vinyl wrap on entire car body" \
  -F "preserve_background=true" \
  -F "match_lighting=true" \
  --output styled_car.png
```

#### Using the Python Client

```bash
python example_client.py \
  --car-image examples/car.jpg \
  --product-image examples/black_wrap.jpg \
  --description "matte black vinyl wrap on entire car body" \
  --output output/styled_car.png
```

#### Using Python Code

```python
import requests

# API endpoint
url = "http://localhost:8000/style"

# Prepare files
files = {
    'car_image': open('car.jpg', 'rb'),
    'product_image': open('product.jpg', 'rb')
}

data = {
    'product_description': 'chrome aftermarket rims and black side skirts',
    'preserve_background': True,
    'match_lighting': True
}

# Make request
response = requests.post(url, files=files, data=data)

# Save result
if response.status_code == 200:
    with open('styled_car.png', 'wb') as f:
        f.write(response.content)
```

### Example Product Descriptions

Good product descriptions lead to better results:

- **Wraps**: "matte black vinyl wrap on entire car body"
- **Rims**: "chrome 20-inch aftermarket wheels with low-profile tires"
- **Body Kits**: "aggressive front bumper lip spoiler and side skirts"
- **Paint**: "metallic blue pearl automotive paint finish"
- **Hood**: "carbon fiber hood with vented design"
- **Spoiler**: "large carbon fiber rear wing spoiler"
- **Windows**: "dark tinted windows all around"

## API Endpoints

### POST /style

Apply styling to a single car image.

**Parameters:**
- `car_image` (file): Original car image
- `product_image` (file): Reference image of the styling product
- `product_description` (string): Detailed description of the product
- `preserve_background` (boolean): Keep original background (default: true)
- `match_lighting` (boolean): Match lighting conditions (default: true)
- `return_format` (string): 'result', 'edges', or 'all' (default: 'result')

**Response:** PNG image of the styled car

### POST /style-batch

Apply styling to multiple car images.

**Parameters:**
- `car_images` (files): Multiple car images
- `product_image` (file): Reference image of the styling product
- `product_description` (string): Detailed description
- `preserve_background` (boolean): Default true
- `match_lighting` (boolean): Default true

**Response:** JSON with processing status

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": true
}
```

## Configuration

Edit `.env` file or `config.py` to customize:

```python
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Device
DEVICE=cuda  # or 'cpu'

# Model Settings
SD_MODEL=stabilityai/stable-diffusion-xl-base-1.0
CONTROLNET_MODEL=diffusers/controlnet-canny-sdxl-1.0

# Generation Parameters
IMAGE_SIZE=1024
NUM_INFERENCE_STEPS=30  # Higher = better quality but slower
GUIDANCE_SCALE=7.5      # Higher = more adherence to prompt
CONTROLNET_CONDITIONING_SCALE=0.8  # Higher = more structure preservation

# Edge Detection
CANNY_LOW_THRESHOLD=100
CANNY_HIGH_THRESHOLD=200
```

## Performance Tips

### GPU Optimization
- The service uses model offloading and VAE slicing for memory efficiency
- For faster processing, ensure CUDA is properly installed
- 8GB+ VRAM recommended for optimal performance

### CPU Mode
- Set `DEVICE=cpu` in `.env` for CPU-only mode
- Processing will be significantly slower (5-10 minutes per image)
- Consider reducing `NUM_INFERENCE_STEPS` to 20 for faster results

### Quality vs Speed Trade-offs
- `NUM_INFERENCE_STEPS`: 20 (fast), 30 (balanced), 50 (high quality)
- `IMAGE_SIZE`: 512 (fast), 1024 (balanced), 2048 (high quality, slow)

## Project Structure

```
carstyler/
├── main.py                 # FastAPI application
├── config.py              # Configuration management
├── run.py                 # Server startup script
├── example_client.py      # Example client code
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── services/
│   ├── __init__.py
│   ├── car_styler_service.py    # Main styling service
│   └── image_processor.py        # Image processing utilities
└── README.md
```

## How It Works

1. **Edge Detection**: Extracts edges from the car image using Canny edge detection
2. **ControlNet Conditioning**: Uses edges to guide the generation process
3. **Text-to-Image Generation**: Stable Diffusion XL generates the styled car based on the product description
4. **Lighting Analysis**: Analyzes lighting characteristics of the original image
5. **Post-Processing**:
   - Matches lighting conditions
   - Enhances contrast
   - Blends with original to preserve background
6. **Output**: Returns the final styled car image

## Limitations

- Best results with clear, well-lit car photos
- Product description quality significantly affects results
- Processing time varies based on hardware (30s-5min per image)
- Complex modifications may require multiple passes
- Background preservation works best with simple backgrounds

## Troubleshooting

### Out of Memory Errors
- Reduce `IMAGE_SIZE` in config
- Reduce `NUM_INFERENCE_STEPS`
- Switch to CPU mode
- Close other applications

### Poor Quality Results
- Improve product description (be more specific)
- Use higher quality input images
- Increase `NUM_INFERENCE_STEPS`
- Adjust `GUIDANCE_SCALE` and `CONTROLNET_CONDITIONING_SCALE`

### Slow Processing
- Enable GPU mode (set `DEVICE=cuda`)
- Reduce `NUM_INFERENCE_STEPS`
- Reduce `IMAGE_SIZE`
- Ensure CUDA is properly installed

## Future Improvements

- [ ] Add more advanced segmentation with SAM integration
- [ ] Support for depth-based ControlNet
- [ ] Multiple product application in one pass
- [ ] Video processing support
- [ ] Fine-tuned models for specific car brands
- [ ] Real-time preview with lower quality mode
- [ ] User gallery and result caching
- [ ] Support for more styling categories (interior, engine bay, etc.)

## License

This project is provided as-is for demonstration purposes.

## Acknowledgments

- Stable Diffusion by Stability AI
- ControlNet by Lvmin Zhang
- Hugging Face Diffusers library
- FastAPI framework

## Support

For issues and questions, please open an issue on the repository.

---

Built with Claude Code
