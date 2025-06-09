import cv2
import numpy as np

# Video properties
filename = "dummy_video.mp4"
width = 640
height = 480
fps = 30
duration_seconds = 3
num_frames = fps * duration_seconds

# Define a simple shape (a rectangle)
rect_size = 50
rect_color = (0, 255, 0)  # Green

# Initial position of the rectangle
x = width // 2
y = height // 2

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {filename}")
    exit()

print(f"Generating {filename}...")

for frame_num in range(num_frames):
    # Create a black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Update rectangle position (simple diagonal movement)
    current_x = x + int(50 * np.sin(2 * np.pi * frame_num / (fps * 1))) # Move horizontally
    current_y = y + int(50 * np.cos(2 * np.pi * frame_num / (fps * 1))) # Move vertically

    # Ensure the rectangle stays within bounds
    current_x = max(rect_size // 2, min(width - rect_size // 2, current_x))
    current_y = max(rect_size // 2, min(height - rect_size // 2, current_y))

    # Draw the rectangle
    cv2.rectangle(frame,
                  (current_x - rect_size // 2, current_y - rect_size // 2),
                  (current_x + rect_size // 2, current_y + rect_size // 2),
                  rect_color,
                  -1) # Filled rectangle

    # Add some random noise
    noise = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    out.write(frame)

out.release()
print(f"{filename} created successfully.")
