import os

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from albumentations import (
    Compose, Rotate, RandomScale, ShiftScaleRotate, GaussianBlur, CLAHE, RandomBrightnessContrast,
    RGBShift, GaussNoise, CoarseDropout, OpticalDistortion, Perspective, Equalize)
from albumentations.augmentations.geometric.transforms import Affine

# Source and destination directories
source_dir = r"D:\PycharmProjects\a_2\collected_data\3_340_augmented_sets\3_pre_augmented_cropped_frames"
destination_dir = r"D:\PycharmProjects\a_2\collected_data\3_340_augmented_sets\4_post_augmented_frames"
os.makedirs(destination_dir, exist_ok=True)


# Custom augmentations
def translate_image(image):
    image = image.copy()
    h, w = image.shape[:2]
    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)


def blur_or_sharpen(image):
    image = image.copy()
    if random.random() > 0.5:
        return cv2.GaussianBlur(image, (5, 5), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)


def occlusion_simulation(image):
    h, w = image.shape[:2]

    # Parameters for the size and number of blocks
    max_size = 4  # Maximum block size (in pixels)
    min_size = 3  # Minimum block size (in pixels)
    num_blocks = random.randint(1, 2)  # Random number of blocks (reduce to 1-3)

    occluded_image = image.copy()

    for _ in range(num_blocks):
        # Random block dimensions
        occ_w = random.randint(min_size, max_size)
        occ_h = random.randint(min_size, max_size)

        # Random block position
        x_start = random.randint(0, max(0, w - occ_w))
        y_start = random.randint(0, max(0, h - occ_h))

        # Add the block to the image
        cv2.rectangle(occluded_image, (x_start, y_start), (x_start + occ_w, y_start + occ_h), (0, 0, 0), -1)

    return occluded_image


def specular_highlights(image):
    h, w = image.shape[:2]
    max_size = 4
    min_size = 3
    num_highlights = random.randint(1, 3)
    highlighted_image = image.copy()
    for _ in range(num_highlights):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        size = random.randint(min_size, max_size)
        cv2.circle(highlighted_image, (x, y), size, (255, 255, 255), -1)
    return highlighted_image


def pixel_level_augmentation(image):
    corrupted_image = image.copy().astype(np.uint8)
    drop_prob = 0.02
    mask = np.random.choice([0, 1], size=image.shape[:2], p=[drop_prob, 1 - drop_prob]).astype(np.uint8)
    for c in range(image.shape[2]):
        corrupted_image[:, :, c] = corrupted_image[:, :, c] * mask
    return corrupted_image


# Albumentations augmentations
def get_augmentations():
    return Compose([
        Rotate(limit=5, p=0.5),  # Random rotation within ±5°
        RandomScale(scale_limit=0.1, p=0.5),  # Random scaling, 10% variation
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),  # Random translation, scaling, rotation
        GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Add Gaussian noise with variance in range [10, 50]
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Random brightness and contrast adjustment
        GaussianBlur(blur_limit=(3, 7), p=0.5),  # Apply Gaussian blur with kernel size in range [3, 7]
        CLAHE(p=1.0),  # Apply Contrast Limited Adaptive Histogram Equalization
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),  # Random shifts in RGB channels
        CoarseDropout(max_holes=1, max_height=5, max_width=5, p=0.5),  # Randomly drop rectangular areas in the image
        Affine(shear=10, p=0.5),  # Apply shear transformations with a max angle of 10°
        OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Distort image as if viewed through a lens
        Perspective(scale=(0.05, 0.1), p=0.5),  # Apply perspective transformation with scaling factor in range [0.05, 0.1]
        Equalize(p=1.0),  # Equalize image histogram for enhanced contrast
    ])


def apply_with_probability(augmentation_function, image, p):
    """Applies the augmentation function to the image with probability p."""
    if random.random() < p:
        return augmentation_function(image)
    return image


def resize_with_interpolation(image, target_size):
    """Resize image with appropriate interpolation based on size."""
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    # Check if resizing is necessary
    if (original_h, original_w) == (target_h, target_w):
        return image  # No resizing needed

    # Determine interpolation method
    if original_h > target_h or original_w > target_w:  # Downscaling
        interpolation = cv2.INTER_AREA
    else:  # Upscaling
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def process_frame(frame_path, dest_path, idx):
    original_size = (112, 80)  # (width, height)
    image = cv2.imread(frame_path)
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"Invalid frame: {frame_path}")

    # Define custom augmentations with their respective probabilities
    custom_augmentations = [
        (translate_image, 0.5),          # Apply translation with 50% probability
        (blur_or_sharpen, 0.5),          # Apply blur or sharpen with 50% probability
        (occlusion_simulation, 0.5),     # Apply occlusion with 50% probability
        (specular_highlights, 0.5),      # Apply specular highlights with 50% probability
        (pixel_level_augmentation, 0.5)  # Apply pixel-level augmentation with 50% probability
    ]

    if random.random() > 0.35:  # 35% chance for custom augmentations
        for aug_func, prob in custom_augmentations:
            image = apply_with_probability(aug_func, image, prob)
    else:  # 65% chance for Albumentations
        augmentations = get_augmentations()
        image = augmentations(image=image)["image"]

    # Resize to the original size with dynamic interpolation
    image = resize_with_interpolation(image, original_size)

    # Save the augmented image
    cv2.imwrite(os.path.join(dest_path, f"{idx + 1:02d}.png"), image)


# Process a word folder
def process_word_folder(source_word_path, dest_word_path, set_folder):
    """Process all frames in a word folder."""
    os.makedirs(dest_word_path, exist_ok=True)
    frame_paths = sorted(
        [os.path.join(source_word_path, f) for f in os.listdir(source_word_path) if f.endswith('.png')])

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_path, dest_word_path, idx) for idx, frame_path in
                   enumerate(frame_paths)]
        for future in as_completed(futures):
            future.result()

    # Print detailed output
    print(
        f"Finished processing '{os.path.basename(source_word_path)}' in '{os.path.basename(set_folder)}'. Images processed -> {len(frame_paths)}")


# Process a single set
def process_set(source_set_path, dest_set_path):
    """Process all word folders in a set folder."""
    word_folders = sorted([f for f in os.listdir(source_set_path) if os.path.isdir(os.path.join(source_set_path, f))])
    for word_folder in word_folders:
        source_word_path = os.path.join(source_set_path, word_folder)
        dest_word_path = os.path.join(dest_set_path, word_folder)
        process_word_folder(source_word_path, dest_word_path, source_set_path)


# Main function
def main():
    """Main function to process all sets with balanced source set usage."""
    set_folders = sorted(
        [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
    total_sets = 340
    source_set_usage = {os.path.basename(set_folder): 0 for set_folder in set_folders}  # Track usage by folder name

    for set_idx in range(1, total_sets + 1):
        # Find sets with the least usage
        min_usage = min(source_set_usage.values())
        least_used_sets = [set_folder for set_folder, usage in source_set_usage.items() if usage == min_usage]

        # Randomly select from least used sets
        selected_set = random.choice(least_used_sets)

        # Increment usage count
        source_set_usage[selected_set] += 1

        # Get the full path of the selected set
        source_set_path = \
        [os.path.join(source_dir, folder) for folder in source_set_usage.keys() if folder == selected_set][0]

        # Process the set
        dest_set_path = os.path.join(destination_dir, f"set_{set_idx:03d}")
        os.makedirs(dest_set_path, exist_ok=True)
        process_set(source_set_path, dest_set_path)

        # Print progress
        print(
            f"\nSet folder '{selected_set}' used for set_{set_idx:03d}. Current usage: {source_set_usage[selected_set]}\n")

    # Final summary
    print("\nFinal Usage Summary:")
    for set_folder, usage in source_set_usage.items():
        print(f"Set folder '{set_folder}' was used {usage} times.")


if __name__ == "__main__":
    main()