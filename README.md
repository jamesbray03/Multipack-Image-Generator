# Multipack Image Creator for Product Listings

This project is designed to create multipack images for product listings. The script processes images by removing their backgrounds, resizing, and positioning them to create various multipack configurations. The output images can be used for displaying product bundles on e-commerce platforms.

## Features

- **Background Removal:** Automatically removes the background from product images.
- **Resizing and Positioning:** Resizes and positions the products in a predefined layout.
- **Multipack Generation:** Generates images with multiple products arranged in different configurations.
- **Output Management:** Saves the processed images in organized folders.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- rembg
- NumPy

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install opencv-python-headless numpy rembg
    ```

## Usage

1. **Prepare Input Images:**
   - Place your product images in the `Input Images` folder. Supported formats are `.jpg` and `.png`.

2. **Run the Script:**

    ```bash
    python <script_name>.py
    ```

3. **Output:**
   - The processed images will be saved in the `Output Images` folder.
   - Each product will have its own subfolder with different multipack configurations.
