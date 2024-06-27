import os
import shutil
import cv2
import numpy as np
import rembg

OUTPUT_RES_X = 1600
OUTPUT_RES_Y = 1600

# Read image safely
def read_image(filename):
    if os.path.exists(filename):
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return image
    else:
        return None

def strip_background(filename, output_path):
    # Read the image with alpha channel if it exists
    image = read_image(filename)
    if image is not None:
        # Check if the image has 3 channels (BGR)
        if image.shape[2] == 3:
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Add alpha channel
            image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
        elif image.shape[2] == 4:
            # The image already has an alpha channel
            image_bgra = image
            # Convert BGRA to RGBA
            image_rgba = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA)

        # Remove the background using rembg
        result = rembg.remove(image_rgba)

        # Convert the result from RGBA to BGRA for OpenCV compatibility
        result_bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)

        # Trim the image
        mask = result_bgra[:, :, 3]  # Extract the alpha channel
        coords = cv2.findNonZero(mask)  # Find the coordinates of non-zero pixels
        x, y, w, h = cv2.boundingRect(coords)  # Get the bounding rectangle
        cropped_image = result_bgra[y:y+h, x:x+w]  # Crop the image to the bounding rectangle

        # Save the cropped image as BGRA (OpenCV default)
        cv2.imwrite(output_path, cropped_image)
        return output_path
    else:
        print(f"Image not found at {filename}")
        return None


def place_product(stripped, position, size, output_path):
    position = (int(position[0] * OUTPUT_RES_X), int(position[1] * OUTPUT_RES_Y))  # Convert to integer coordinates
    size = (int(size[0] * OUTPUT_RES_X), int(size[1] * OUTPUT_RES_Y))  # Convert to integer size

    # Load the stripped image with alpha channel
    image = read_image(stripped)
    if image is not None:
        # Load the background image if it exists, otherwise create a blank white background
        background = read_image(output_path)
        if background is None:
            # Define output resolution
            background = 255 * np.ones((OUTPUT_RES_Y, OUTPUT_RES_X, 3), np.uint8)

        # Resize the stripped image to the specified size
        sticker = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        if sticker is not None:
            # Get the dimensions of the sticker and background
            sticker_height, sticker_width = sticker.shape[:2]
            background_height, background_width = background.shape[:2]

            # Separate the alpha channel from the sticker
            alpha_channel = sticker[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
            alpha_channel = np.stack([alpha_channel] * 3, axis=-1)  # Create 3-channel alpha

            # Define the region of interest (ROI) in the background
            y1, y2 = position[1], position[1] + sticker_height
            x1, x2 = position[0], position[0] + sticker_width

            # Blend the sticker with the background in the ROI
            background[y1:y2, x1:x2] = np.clip((1 - alpha_channel) * background[y1:y2, x1:x2] + alpha_channel * sticker[:, :, :3], 0, 255)
            
            # Ensure the ROI is within the bounds of the background
            if y2 > background_height or x2 > background_width:
                print("Sticker is out of bounds")

        else:
            print(f"Sticker not found at {stripped}")

        # Save the modified background image
        cv2.imwrite(output_path, background)

if __name__ == "__main__":
    folder_path = os.getcwd()
    output_folder = os.path.join(folder_path, 'Output Images')
    input_folder = os.path.join(folder_path, 'Input Images')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Get the file paths of all images in the input folder and store in an array
    images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and (f.endswith('.jpg') or f.endswith('.png'))]

    # Loop through all images in the directory
    for image in images:

        print(f"Processing {image}...")

        # Create a new folder named after the image
        image_folder = os.path.join(output_folder, image.split('.')[0])
        os.makedirs(image_folder, exist_ok=True)

        # Remove the background of the image
        stripped_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', '_stripped.png'))
        stripped_path = strip_background(os.path.join(input_folder, image), stripped_path)

        if stripped_path is None:
            continue

        # Get aspect ratio of the image
        img = read_image(stripped_path)
        ar = img.shape[1] / img.shape[0]

        # longest side as ratio of image
        border = 0.025
        long_size = 1 - 2 * border

        if ar < 0.6: # Portrait
            for num_products in range(2, 7):
                multi_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', f"_{num_products}.png"))
                
                y_size = long_size
                x_size = long_size * ar

                # distribute products vertically
                x_spacing = (1 - (2 * border) - x_size) / (num_products - 1)

                for i in range(num_products):
                    x_pos = border + x_spacing * i
                    place_product(stripped_path, (x_pos, border), (x_size, y_size), multi_path)

        elif ar < 1.4: # Square
            for num_products in range(2, 7):
                multi_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', f"_{num_products}.png"))
                
                if ar < 1: # tall square
                    y_size = (1 - 2 * border) / (1 + 0.1 * num_products)
                    x_size = y_size * ar

                else: # wide square
                    x_size = (1 - 2 * border) / (1 + 0.1 * num_products)
                    y_size = x_size / ar

                x_spacing = (1 - (2 * border) - x_size) / (num_products - 1)
                y_spacing = (1 - (2 * border) - y_size) / (num_products - 1)

                for i in range(num_products):
                    x_pos = border + x_spacing * i
                    y_pos = border + y_spacing * i
                    place_product(stripped_path, (x_pos, y_pos), (x_size, y_size), multi_path)

        else: # Landscape
            for num_products in range(2, 7):
                multi_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', f"_{num_products}.png"))

                x_size = long_size
                y_size = long_size / ar

                # distribute products vertically
                y_spacing = (1 - (2 * border) - y_size) / (num_products - 1)

                for i in range(num_products):
                    y_pos = border + y_spacing * i
                    place_product(stripped_path, (border, y_pos), (x_size, y_size), multi_path)

    print("Processing complete!")
