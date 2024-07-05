import sys
import cv2
import time

# List of video filenames (passed as command-line arguments, except the last one)
video_files = video_files = sys.argv[1:-1]#["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]#sys.argv[1:-1]
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Extract the initial time from the last command-line argument
try:
    initial_time = int(sys.argv[-1])  #  the last argument is a int representing time
except ValueError:
    print("initial time is take the value as")
    print(int(sys.argv[-1]))
    print("Error: Last argument should be a valid initial time.")
    exit(1)


# Define specific frame positions for up to 4 videos
frame_positions = [(0, 0), (700, 0), (0, 400), (700, 400)]

# Open video capture objects for each video
video_captures = [cv2.VideoCapture(file) for file in video_files]

# Check if video capture objects were successfully opened
for i, capture in enumerate(video_captures):
    if not capture.isOpened():
        print(f"Error: Unable to open video file {video_files[i]}")
        exit(1)

# Initialize variables for controlling video switching
current_video_index = 0
switch_video_time = time.time() + initial_time  # Set initial switch time based on initial_time

# Main loop to display videos and handle events
while True:
    # Calculate elapsed time since last video switch
    elapsed_time = switch_video_time - time.time()

    # Check if it's time to switch to the next video (every 5 seconds)
    if elapsed_time <= 0:
        current_video_index = (current_video_index + 1) % len(video_captures)
        switch_video_time = time.time() + initial_time  # Reset timer for the next switch

    # Display videos
    for i, capture in enumerate(video_captures):
        if capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 360))

            # Render circle with color and optional countdown timer
            if i == current_video_index:
                # Draw green circle for the current video with countdown timer
                cv2.circle(frame, (600, 10), 15, (0, 255, 0), -1)  # Green circle
                countdown_text = f"{int(elapsed_time)}"
                text_size, _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = 600 - text_size[0] // 2
                cv2.putText(frame, countdown_text, (text_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # Draw red circle for other videos without timer
                cv2.circle(frame, (600, 10), 15, (0, 0, 255), -1)  # Red circle

            cv2.imshow(f"Video {i+1}", frame)
            cv2.moveWindow(f"Video {i+1}", frame_positions[i][0], frame_positions[i][1])

    # Check for exit key ('q')
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release video capture objects and close windows
for capture in video_captures:
    capture.release()
cv2.destroyAllWindows()