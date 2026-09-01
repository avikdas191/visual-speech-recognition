import cv2
import os
import dlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Define paths
source_dir = r"C:\Users\admin\Videos\project_videos\original_dataset_even_frames"
output_dir = r"C:\Users\admin\Videos\project_videos\original_dataset_DLib_cropped_frames"

# Load the detector and predictor (dlib models)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/PycharmProjects/source project files/models/shape_predictor_68_face_landmarks.dat")

# Mouth crop dimensions
LIP_HEIGHT = 80
LIP_WIDTH = 112


def process_frame(frame_file, word_path, output_word_path):
    frame_path = os.path.join(word_path, frame_file)
    try:
        # Load the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}. Skipping.")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray)

        if not faces:
            return False

        # Only process if a face is detected
        for face in faces:
            landmarks = predictor(gray, face)

            # Extract the mouth region by iterating over the landmarks (48 to 67)
            mouth_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]
            mouth_points_np = np.array(mouth_points)

            # Finding the bounding rectangle around the mouth points
            x, y, w, h = cv2.boundingRect(mouth_points_np)

            # Calculate padding to fit the target dimensions
            width_diff = LIP_WIDTH - w
            height_diff = LIP_HEIGHT - h
            pad_left = max(width_diff // 2, 0)
            pad_right = max(width_diff - pad_left, 0)
            pad_top = max(height_diff // 2, 0)
            pad_bottom = max(height_diff - pad_top, 0)

            # Padding to ensure it doesn’t exceed image boundaries
            pad_left = min(pad_left, x)
            pad_right = min(pad_right, frame.shape[1] - (x + w))
            pad_top = min(pad_top, y)
            pad_bottom = min(pad_bottom, frame.shape[0] - (y + h))

            # Crop and resize the mouth region
            lip_frame = frame[y - pad_top:y + h + pad_bottom, x - pad_left:x + w + pad_right]
            lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

            # Save the cropped mouth region to the output directory
            output_frame_path = os.path.join(output_word_path, frame_file)
            cv2.imwrite(output_frame_path, lip_frame)

            # print(f"Cropped and saved frame: {output_frame_path}")
            return True  # Exit after processing the first detected face
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}")
        return False


def process_word_folder(set_folder, word_folder):
    word_path = os.path.join(set_folder, word_folder)

    # Create output directory for cropped frames
    output_word_path = os.path.join(output_dir, os.path.basename(set_folder), word_folder)
    os.makedirs(output_word_path, exist_ok=True)

    frame_files = [f for f in os.listdir(word_path) if f.endswith('.png')]

    cropped_count = 0  # Counter for successfully cropped frames

    # Use ProcessPoolExecutor to process frames concurrently
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_file, word_path, output_word_path) for frame_file in frame_files]
        for future in futures:
            if future.result():  # Increment the counter if the frame was successfully cropped
                cropped_count += 1
    # Print the number of cropped images for the current word folder
    print(f"Finished processing '{word_folder}' in '{os.path.basename(set_folder)}'. Images cropped -> {cropped_count}")


if __name__ == "__main__":
    # Iterate over each set folder and process each word folder within
    set_folders = os.listdir(source_dir)
    for set_folder in set_folders:
        full_set_path = os.path.join(source_dir, set_folder)
        if os.path.isdir(full_set_path):
            word_folders = os.listdir(full_set_path)
            for word_folder in word_folders:
                if os.path.isdir(os.path.join(full_set_path, word_folder)):
                    print(f"Processing word folder: '{word_folder}' in: '{set_folder}'")
                    process_word_folder(full_set_path, word_folder)

    print("\nMouth cropping completed for all frames.")