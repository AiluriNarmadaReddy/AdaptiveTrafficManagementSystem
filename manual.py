import sys
import cv2

# List of video filenames (passed as command-line arguments)
video_files = sys.argv[1:]

# Define specific frame positions for up to 4 videos
frame_positions = [(0, 0), (700, 0), (0, 400), (700, 400)]

# Open video capture objects for each video
video_captures = [cv2.VideoCapture(file) for file in video_files]

# Check if video capture objects were successfully opened
for i, capture in enumerate(video_captures):
    if not capture.isOpened():
        print(f"Error: Unable to open video file {video_files[i]}")
        exit(1)

# Define circle color constants
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Initialize the clicked video index (None means no video is clicked initially)
clicked_video_index = None

# Main loop to display videos and handle events
while True:
    for i, capture in enumerate(video_captures):
        if capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 360))

            # Check if this video frame should be highlighted (activated by keyboard)
            if clicked_video_index is not None and clicked_video_index == i:
                cv2.circle(frame, (600, 10), 15, GREEN, -1)  # Green circle for clicked video
            else:
                cv2.circle(frame, (600, 10), 15, RED, -1)  # Red circle for other videos

            cv2.imshow(f"Video {i+1}", frame)
            cv2.moveWindow(f"Video {i+1}", frame_positions[i][0], frame_positions[i][1])

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Handle keyboard input to activate specific video frames
    if key == ord('1'):
        clicked_video_index = 0  # Activate video 1 by pressing '1'
    elif key == ord('2'):
        clicked_video_index = 1  # Activate video 2 by pressing '2'
    elif key == ord('3'):
        clicked_video_index = 2  # Activate video 3 by pressing '3'
    elif key == ord('4'):
        clicked_video_index = 3  # Activate video 4 by pressing '4'

    # Check for exit key ('q')
    if key == ord('q'):
        break

# Release video capture objects and close windows
for capture in video_captures:
    capture.release()
cv2.destroyAllWindows()
