import os
import cv2
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Source and destination directories
source_dir = r"D:\PycharmProjects\a_2\collected_data\1_frames"
destination_dir = r"D:\PycharmProjects\a_2\collected_data\3_340_augmented_sets\2_pre_augmented_frames"
os.makedirs(destination_dir, exist_ok=True)


def temporal_shuffling(frames):
    """Slightly shuffle the order of consecutive frames."""
    shuffled_frames = frames.copy()
    shuffle_strength = 3
    for i in range(len(frames) - shuffle_strength):
        swap_idx = random.randint(i, min(i + shuffle_strength, len(frames) - 1))
        shuffled_frames[i], shuffled_frames[swap_idx] = shuffled_frames[swap_idx], shuffled_frames[i]
    return shuffled_frames


def morph_images(image1, image2, alpha):
    """Morph two images to create intermediate frames."""
    return cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)


def morph_sequence(frames):
    """Generate morphed frames between consecutive frames."""
    morphed_frames = []
    for i in range(len(frames) - 1):
        morphed_frames.append(frames[i])
        alpha = random.uniform(0.3, 0.7)  # Random alpha for variation
        morphed_frames.append(morph_images(frames[i], frames[i + 1], alpha))
    return morphed_frames[:len(frames)]


def process_word_folder(set_folder, word_folder):
    """Process all frames in a word folder."""
    source_path = os.path.join(set_folder, word_folder)
    dest_path = os.path.join(destination_dir, os.path.basename(set_folder), word_folder)
    os.makedirs(dest_path, exist_ok=True)

    # Load frames
    frame_paths = sorted([os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.png')])
    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(f"Invalid frame found: {path}")
        frames.append(frame)

    if len(frames) != 60:
        raise ValueError(f"Folder {source_path} does not contain exactly 60 valid frames.")

    # Decide augmentation type
    if random.random() < 0.7:  # 70% chance for no augmentation
        augmented_frames = frames  # No augmentation
    else:
        augmentations = [
            temporal_shuffling,
            morph_sequence
        ]
        augmented_frames = random.choice(augmentations)(frames)

    augmented_frames = augmented_frames[:60]  # Ensure exactly 60 frames

    # Save augmented frames
    for idx, frame in enumerate(augmented_frames):
        cv2.imwrite(os.path.join(dest_path, f"{idx:02d}.png"), frame)  # Updated naming format

    print(f"Finished processing '{word_folder}' in '{os.path.basename(set_folder)}'. Images processed -> {len(augmented_frames)}")

    return word_folder, len(augmented_frames)


def process_set_folder(set_folder):
    """Process all word folders in a set folder sequentially."""
    word_folders = sorted([f for f in os.listdir(set_folder) if os.path.isdir(os.path.join(set_folder, f))])
    results = []
    for word_folder in word_folders:
        try:
            result = process_word_folder(set_folder, word_folder)
            results.append(result)
        except Exception as e:
            print(f"Error processing '{word_folder}' in '{os.path.basename(set_folder)}': {e}")
    return os.path.basename(set_folder), results


def main():
    """Main function to process all sets folder by folder."""
    set_folders = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

    for set_folder in set_folders:
        print(f"\nProcessing set folder: {os.path.basename(set_folder)}")
        with ProcessPoolExecutor() as executor:
            word_folders = sorted([f for f in os.listdir(set_folder) if os.path.isdir(os.path.join(set_folder, f))])
            futures = {executor.submit(process_word_folder, set_folder, word_folder): word_folder for word_folder in word_folders}
            for future in as_completed(futures):
                word_folder = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error processing '{word_folder}' in '{os.path.basename(set_folder)}': {e}")

        print(f"Finished processing set folder: {os.path.basename(set_folder)}")


if __name__ == "__main__":
    main()