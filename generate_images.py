import os
import cv2
import numpy as np
import rembg

OUTPUT_RES_X = 1600
OUTPUT_RES_Y = 1600

def strip_background(filename, output_path):
    # Remove the background
    image = cv2.imread(filename)
    if image is not None:
        # Convert BGR image to RGBA
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # Remove the background using rembg
        result = rembg.remove(image_rgba)

        # Trim the image
        mask = result[:, :, 3]  # Extract the alpha channel
        coords = cv2.findNonZero(mask)  # Find the coordinates of non-zero pixels
        x, y, w, h = cv2.boundingRect(coords)  # Get the bounding rectangle
        cropped_image = result[y:y+h, x:x+w]  # Crop the image to the bounding rectangle

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        return output_path
    else:
        print(f"Image not found at {filename}")
        return None

def place_product(stripped, position, size, output_path):
    position = (int(position[0] * OUTPUT_RES_X), int(position[1] * OUTPUT_RES_Y))  # Convert to integer coordinates
    size = (int(size[0] * OUTPUT_RES_X), int(size[1] * OUTPUT_RES_Y))  # Convert to integer size

    # Load the stripped image with alpha channel
    image = cv2.imread(stripped, cv2.IMREAD_UNCHANGED)
    if image is not None:
        # Load the background image if it exists, otherwise create a blank white background
        background = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
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
    os.makedirs(output_folder, exist_ok=True)

    # Get the file paths of all images in the directory and store in an array
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and (f.endswith('.jpg') or f.endswith('.png'))]

    # Loop through all images in the directory
    for image in images:

        # Create a new folder named after the image
        image_folder = os.path.join(output_folder, image.split('.')[0])
        os.makedirs(image_folder, exist_ok=True)

        # Remove the background of the image
        stripped_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', '_stripped.png'))
        stripped_path = strip_background(os.path.join(folder_path, image), stripped_path)

        if stripped_path is None:
            continue

        # Get aspect ratio of the image
        img = cv2.imread(stripped_path)
        ar = img.shape[1] / img.shape[0]

        # longest side as ratio of image
        border = 0.025
        big_size = 1 - 2 * border
        small_size = 0.8

        if ar < 1: # Portrait
            for num_products in range(2, 7):
                multi_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', f"_{num_products}.png"))
                
                big_y_size = big_size
                big_x_size = big_y_size * ar
                small_y_size = small_size
                small_x_size = small_y_size * ar

                # centre vertically
                small_y_pos = 0.5 - small_y_size / 2

                # distribute small products
                x_spacing = (1 - (2 * border) - big_x_size) / num_products

                for i in range(num_products - 1):
                    small_x_pos = 1 - border - x_spacing * i - small_x_size
                    place_product(stripped_path, (small_x_pos, small_y_pos), (small_x_size, small_y_size), multi_path)

                # place big product
                place_product(stripped_path, (border, border), (big_x_size, big_y_size), multi_path)

        else: # Landscape
            for num_products in range(2, 7):
                multi_path = os.path.join(image_folder, image.replace('.jpg', '.png').replace('.png', f"_{num_products}.png"))

                big_x_size = big_size
                big_y_size = big_x_size / ar
                small_x_size = small_size
                small_y_size = small_x_size / ar

                # centre horizontally
                small_x_pos = 0.5 - small_x_size / 2

                # distribute small products
                y_spacing = (1 - (2 * border) - big_y_size) / num_products

                for i in range(num_products - 1):
                    small_y_pos = border + y_spacing * i
                    place_product(stripped_path, (small_x_pos, small_y_pos), (small_x_size, small_y_size), multi_path)
                    
                # place big product
                place_product(stripped_path, (border, 1 - border - big_y_size), (big_x_size, big_y_size), multi_path)

    print("Processing complete!")
