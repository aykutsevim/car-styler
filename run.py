"""
Simple script to run the Car Styler API server.
"""

import uvicorn
from config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Car Styler API")
    print("=" * 60)
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Device: {settings.device}")
    print(f"Model: {settings.sd_model}")
    print("=" * 60)
    print("\nAPI will be available at:")
    print(f"  http://localhost:{settings.api_port}")
    print(f"  http://localhost:{settings.api_port}/docs (API documentation)")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
