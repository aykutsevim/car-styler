# Quick Start Guide

Get the Car Styler API running in 5 minutes!

## 1. Prerequisites

- Python 3.10+
- At least 16GB RAM
- GPU with 8GB+ VRAM (optional but recommended)

## 2. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Configuration

```bash
# Copy environment file
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env if needed (default values work fine)
```

## 4. First Run

```bash
python run.py
```

**First startup will download AI models (~10GB). This takes 10-20 minutes.**

Once you see:
```
INFO:     Application startup complete.
```

You're ready!

## 5. Test the API

Open your browser and go to:
```
http://localhost:8000/docs
```

You'll see an interactive API documentation where you can test the endpoints.

### Test with curl

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", "device": "cuda", "models_loaded": false}
```

## 6. Style Your First Car

### Option A: Using the Python Client

```bash
python example_client.py \
  --car-image path/to/your/car.jpg \
  --product-image path/to/product.jpg \
  --description "matte black vinyl wrap" \
  --output styled_car.png
```

### Option B: Using curl

```bash
curl -X POST "http://localhost:8000/style" \
  -F "car_image=@car.jpg" \
  -F "product_image=@product.jpg" \
  -F "product_description=matte black vinyl wrap" \
  --output styled_car.png
```

### Option C: Using the Interactive Docs

1. Go to http://localhost:8000/docs
2. Click on `/style` endpoint
3. Click "Try it out"
4. Upload your images
5. Fill in the description
6. Click "Execute"
7. Download the result

## 7. Example Product Descriptions

Use detailed descriptions for best results:

**Good descriptions:**
- "matte black vinyl wrap covering entire car body"
- "chrome 20-inch aftermarket wheels with low-profile tires"
- "carbon fiber hood with aggressive vented design"
- "metallic blue automotive paint with pearl finish"

**Poor descriptions:**
- "black" (too vague)
- "new wheels" (not specific enough)
- "make it cool" (unclear)

## Troubleshooting

### "Out of memory" error
```bash
# Edit .env file:
IMAGE_SIZE=512
NUM_INFERENCE_STEPS=20
DEVICE=cpu  # if you don't have a GPU
```

### "Models not loading"
- Check your internet connection
- Make sure you have enough disk space (15GB free)
- Try deleting the `models/` folder and restart

### "API not responding"
- Make sure port 8000 is not in use by another application
- Try changing API_PORT in .env file

### "Slow processing"
- First run is always slower (loading models)
- CPU mode is 10x slower than GPU
- Reduce IMAGE_SIZE and NUM_INFERENCE_STEPS for faster results

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the API docs at http://localhost:8000/docs
- Experiment with different product descriptions
- Adjust settings in `.env` for quality vs speed trade-offs

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python run.py` | Start the server |
| `python example_client.py --help` | See client options |
| `curl http://localhost:8000/health` | Check API status |

## Support

- Check [README.md](README.md) for detailed docs
- Look at example code in `example_client.py`
- Open an issue on GitHub for bugs

---

Happy styling!
