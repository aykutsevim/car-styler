from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from typing import Optional

from services import CarStylerService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Car Styler API",
    description="AI-powered car styling visualization service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service (lazy loading on first request)
styler_service: Optional[CarStylerService] = None


def get_styler_service() -> CarStylerService:
    """Get or create the car styler service instance."""
    global styler_service
    if styler_service is None:
        styler_service = CarStylerService()
        styler_service.load_models()
    return styler_service


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Car Styler API...")
    logger.info(f"Using device: {settings.device}")
    # Models will be loaded lazily on first request to speed up startup


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global styler_service
    if styler_service is not None:
        styler_service.unload_models()
    logger.info("Car Styler API shutdown complete")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Car Styler API",
        "version": "1.0.0",
        "description": "AI-powered car styling visualization service",
        "endpoints": {
            "POST /style": "Apply styling to a car image",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": settings.device,
        "models_loaded": styler_service is not None
    }


@app.post("/style")
async def apply_car_styling(
    car_image: UploadFile = File(..., description="Image of the car"),
    product_image: UploadFile = File(..., description="Image of the styling product"),
    product_description: str = Form(
        ...,
        description="Description of how to apply the product (e.g., 'matte black wrap', 'chrome rims', 'carbon fiber hood')"
    ),
    preserve_background: bool = Form(
        True,
        description="Whether to preserve the original background"
    ),
    match_lighting: bool = Form(
        True,
        description="Whether to match lighting conditions"
    ),
    return_format: str = Form(
        "result",
        description="Which image to return: 'result', 'edges', 'all'"
    )
):
    """
    Apply a styling product to a car image.

    Args:
        car_image: The original car image
        product_image: Reference image of the styling product
        product_description: Text description of the product and application
        preserve_background: Keep original background
        match_lighting: Match lighting from original image
        return_format: 'result' for final image, 'edges' for edge map, 'all' for JSON with URLs

    Returns:
        Styled car image or processing results
    """
    try:
        logger.info(f"Received styling request: {product_description}")

        # Load images
        car_img = Image.open(io.BytesIO(await car_image.read())).convert("RGB")
        product_img = Image.open(io.BytesIO(await product_image.read())).convert("RGB")

        logger.info(f"Car image size: {car_img.size}")
        logger.info(f"Product image size: {product_img.size}")

        # Get service and apply styling
        service = get_styler_service()
        results = service.apply_styling(
            car_image=car_img,
            product_image=product_img,
            product_description=product_description,
            preserve_background=preserve_background,
            match_lighting=match_lighting
        )

        # Prepare response based on return format
        if return_format == "edges":
            output_image = results['edges']
        elif return_format == "all":
            # Return metadata about the processing
            return {
                "status": "success",
                "message": "Styling applied successfully",
                "product_description": product_description,
                "settings": {
                    "preserve_background": preserve_background,
                    "match_lighting": match_lighting
                },
                "note": "Use return_format='result' to get the styled image"
            }
        else:  # default to 'result'
            output_image = results['result']

        # Convert to bytes and return
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=styled_car.png"
            }
        )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/style-batch")
async def apply_car_styling_batch(
    car_images: list[UploadFile] = File(..., description="Multiple car images"),
    product_image: UploadFile = File(..., description="Image of the styling product"),
    product_description: str = Form(..., description="Description of the product"),
    preserve_background: bool = Form(True),
    match_lighting: bool = Form(True)
):
    """
    Apply styling to multiple car images in batch.

    Args:
        car_images: List of car images to process
        product_image: Reference image of the styling product
        product_description: Text description of the product
        preserve_background: Keep original backgrounds
        match_lighting: Match lighting from original images

    Returns:
        JSON with status and processing information
    """
    try:
        logger.info(f"Received batch styling request for {len(car_images)} images")

        # Load product image
        product_img = Image.open(io.BytesIO(await product_image.read())).convert("RGB")

        # Load all car images
        car_imgs = []
        for car_img_file in car_images:
            img = Image.open(io.BytesIO(await car_img_file.read())).convert("RGB")
            car_imgs.append(img)

        # Get service and apply styling to all images
        service = get_styler_service()
        results = service.batch_apply_styling(
            car_images=car_imgs,
            product_image=product_img,
            product_description=product_description,
            preserve_background=preserve_background,
            match_lighting=match_lighting
        )

        return {
            "status": "success",
            "message": f"Successfully processed {len(results)} images",
            "count": len(results),
            "product_description": product_description,
            "note": "Results are processed. Use single image endpoint for output."
        }

    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch images: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
