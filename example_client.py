"""
Example client script for the Car Styler API.
Demonstrates how to use the service to apply styling to car images.
"""

import requests
from pathlib import Path
import argparse


def style_car(
    car_image_path: str,
    product_image_path: str,
    product_description: str,
    output_path: str = "styled_car.png",
    api_url: str = "http://localhost:8000"
):
    """
    Apply styling to a car image using the Car Styler API.

    Args:
        car_image_path: Path to the car image
        product_image_path: Path to the product image
        product_description: Description of the styling product
        output_path: Where to save the result
        api_url: Base URL of the API
    """
    endpoint = f"{api_url}/style"

    # Prepare files
    with open(car_image_path, 'rb') as car_img, \
         open(product_image_path, 'rb') as product_img:

        files = {
            'car_image': ('car.jpg', car_img, 'image/jpeg'),
            'product_image': ('product.jpg', product_img, 'image/jpeg')
        }

        data = {
            'product_description': product_description,
            'preserve_background': True,
            'match_lighting': True,
            'return_format': 'result'
        }

        print(f"Sending request to {endpoint}...")
        print(f"Product description: {product_description}")

        response = requests.post(endpoint, files=files, data=data)

        if response.status_code == 200:
            # Save the result
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Success! Styled car saved to: {output_path}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())


def check_health(api_url: str = "http://localhost:8000"):
    """Check if the API is running."""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("API Status:", data['status'])
            print("Device:", data['device'])
            print("Models loaded:", data['models_loaded'])
            return True
        else:
            print("API not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Make sure it's running!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Car Styler API Client - Apply styling to car images"
    )

    parser.add_argument(
        '--car-image',
        type=str,
        required=True,
        help='Path to the car image'
    )

    parser.add_argument(
        '--product-image',
        type=str,
        required=True,
        help='Path to the product image'
    )

    parser.add_argument(
        '--description',
        type=str,
        required=True,
        help='Description of the styling product (e.g., "matte black wrap", "chrome rims")'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='styled_car.png',
        help='Output path for the styled car image'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the Car Styler API'
    )

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Just check if the API is running'
    )

    args = parser.parse_args()

    if args.health_check:
        check_health(args.api_url)
        return

    # Verify files exist
    if not Path(args.car_image).exists():
        print(f"Error: Car image not found at {args.car_image}")
        return

    if not Path(args.product_image).exists():
        print(f"Error: Product image not found at {args.product_image}")
        return

    # Check API health first
    print("Checking API status...")
    if not check_health(args.api_url):
        return

    print("\nApplying styling...")
    style_car(
        car_image_path=args.car_image,
        product_image_path=args.product_image,
        product_description=args.description,
        output_path=args.output,
        api_url=args.api_url
    )


if __name__ == "__main__":
    # Example usage without CLI args:
    # style_car(
    #     car_image_path="examples/car.jpg",
    #     product_image_path="examples/black_wrap.jpg",
    #     product_description="matte black vinyl wrap on entire car body",
    #     output_path="output/styled_car.png"
    # )

    main()
