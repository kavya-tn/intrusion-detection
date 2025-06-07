"""
Face Extractor Script

This script processes a video file to detect and extract unique human faces.
It identifies faces in video frames, compares them against already found unique faces,
and saves images of newly encountered unique faces to a specified output directory.
The script uses the face_recognition library for face detection and encoding,
OpenCV for video processing, and PIL (Pillow) for image manipulation.

Features:
- Extracts unique faces from a video.
- Saves each unique face as a JPEG image.
- Allows processing of every nth frame to speed up analysis.
- Provides logging for monitoring progress and errors.
- Customizable output directory for saved face images.
"""
import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_unique_face(known_encodings, new_encoding, tolerance=0.6):
    """
    Compares a new face encoding against a list of known encodings to determine uniqueness.

    Args:
        known_encodings (list): A list of numpy arrays, where each array is a face encoding.
        new_encoding (numpy.ndarray): The face encoding of the newly detected face.
        tolerance (float): The maximum distance between encodings for them to be considered
                           the same face. Lower values mean stricter comparison.

    Returns:
        bool: True if the new_encoding is considered unique (i.e., no known_encodings are
              within the tolerance distance), False otherwise.
    """
    # Check each known encoding to see if it matches the new encoding
    for encoding in known_encodings:
        # Calculate the Euclidean distance between the known encoding and the new encoding.
        # face_recognition typically uses a tolerance around 0.6.
        distance = np.linalg.norm(encoding - new_encoding)
        if distance <= tolerance:
            return False  # Not unique (i.e., a similar face is already known)
    return True # Unique face

def extract_unique_faces(video_path, output_dir="output_faces"):
    """
    Extracts unique faces from a video file and saves them as images.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str, optional): The directory where unique face images will be saved.
                                    Defaults to "output_faces".

    Side effects:
        - Creates the output directory if it doesn't exist.
        - Saves unique face images as JPEG files in the output directory.
        - Logs information about the process, errors, and results.
        - Prints the total number of unique faces found and their save location to the console
          via logging.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Successfully opened video: {video_path}, Frame count: {frame_count}")
    except Exception as e:
        logging.error(f"Failed to open or read video properties for {video_path}: {e}")
        return

    unique_faces = []
    unique_face_images = []  # List to store unique face image paths

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        with tqdm(total=frame_count, desc="Processing Frames") as pbar:
            frame_idx = 0
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logging.info("Reached end of video or failed to read frame.")
                        break  # Exit loop if no frame is returned

                    # Process every nth frame (e.g., every 5th frame) to reduce redundancy
                    if frame_idx % 5 == 0:
                        # Convert frame to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Find all face locations and encodings in the current frame
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        logging.debug(f"Frame {frame_idx}: Found {len(face_locations)} face(s).")

                        for encoding, location in zip(face_encodings, face_locations):
                            # Compare the current face's encoding with known unique encodings
                            if is_unique_face(unique_faces, encoding):
                                unique_faces.append(encoding) # Add new unique encoding to the list
                                # Extract the face image from the frame
                                top, right, bottom, left = location
                                unique_face_image = frame[top:bottom, left:right]

                                # Ensure the extracted face image is not empty
                                if unique_face_image.size == 0:
                                    logging.warning(f"Frame {frame_idx}: Detected face at {location} resulted in an empty image. Skipping.")
                                    continue

                                # Convert the OpenCV (BGR) image to a PIL Image (RGB)
                                pil_image = PILImage.fromarray(cv2.cvtColor(unique_face_image, cv2.COLOR_BGR2RGB))
                                # Construct the path to save the image
                                image_save_path = os.path.join(output_dir, f"unique_face_{len(unique_face_images) + 1}.jpg")
                                pil_image.save(image_save_path)
                                unique_face_images.append(image_save_path) # Store the path of the saved image
                                logging.info(f"Saved new unique face: {image_save_path}")

                except cv2.error as e: # Catch specific OpenCV errors
                    logging.error(f"OpenCV error processing frame {frame_idx}: {e}")
                    break # Example: break on critical OpenCV errors
                except Exception as e:
                    logging.error(f"Generic error processing frame {frame_idx}: {e}")
                    # Continue to next frame for generic errors
                    pass

                frame_idx += 1
                pbar.update(1)  # Update the progress bar
    finally:
        if cap.isOpened():
            cap.release()
            logging.info("Video capture released.")
        # cv2.destroyAllWindows() removed as per instruction

    # Display total unique faces found
    logging.info(f"Total unique faces found: {len(unique_face_images)}")
    if unique_face_images:
        logging.info(f"Unique faces saved in directory: {os.path.abspath(output_dir)}")
    else:
        logging.info("No unique faces were saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unique faces from a video.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--output_dir", default="output_faces",
                        help="Directory to save unique face images (default: output_faces).")

    args = parser.parse_args()

    logging.info(f"Starting face extraction for video: {args.video_path}")
    logging.info(f"Output directory set to: {args.output_dir}")

    extract_unique_faces(args.video_path, args.output_dir)

    logging.info("Script execution finished.")
