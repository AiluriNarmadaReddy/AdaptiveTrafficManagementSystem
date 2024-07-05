import threading
import time
import datetime
import pandas as pd
from ultralytics import YOLO
import cv2
import sys

class VideoProcessor:
    def __init__(self, video_files, initial_delay=30):
        self.video_files = video_files
        self.initial_delay = initial_delay
        self.models = [YOLO('best.onnx', task='detect') for _ in video_files]
        self.video_captures = [cv2.VideoCapture(file) for file in video_files]
        self.data_frames = [pd.DataFrame(columns=['Time', 'Car', 'Motorcycle', 'Truck', 'Bus', 'Total']) for _ in video_files]
        self.traffic_light_colors = {f"Video {i+1}": (0, 255, 0) if i == 0 else (0, 0, 255) for i in range(len(video_files))}
        self.terminate = False  # Flag to indicate termination

    def start(self):
        # Start the update_traffic_light_colors thread
        traffic_light_thread = threading.Thread(target=self.update_traffic_light_colors)
        traffic_light_thread.start()

        # Start display_video threads for each video file
        display_threads = []
        for i, (capture, model, video_file, dataframe) in enumerate(zip(self.video_captures, self.models, self.video_files, self.data_frames)):
            thread = threading.Thread(target=self.display_video, args=(capture, model, f"Video {i+1}", dataframe))
            thread.start()
            display_threads.append(thread)

        # Wait for all display_video threads to complete
        for thread in display_threads:
            thread.join()

        # Set terminate flag to True after display_video threads complete
        self.terminate = True

        # Once all video processing is completed, print message
        print("All video processing completed.")

    def update_traffic_light_colors(self):
        index = 0
        while not self.terminate:
            active_video = f"Video {index + 1}"

            # Update traffic light colors
            for vid in self.traffic_light_colors:
                if vid == active_video:
                    self.traffic_light_colors[vid] = (0, 255, 0)  # Green for active video
                else:
                    self.traffic_light_colors[vid] = (0, 0, 255)  # Red for other videos

            # Debug: Print traffic light colors
            print(f"Updated Traffic Light Colors: {self.traffic_light_colors}")

            time.sleep(self.initial_delay)

            # Calculate new initial delay based on vehicle counts
            last_values = [df['Total'].iloc[-1] if not df.empty else 0 for df in self.data_frames]
            first_video_total_count = self.data_frames[0]['Total'].iloc[-1] if not self.data_frames[0].empty else 1

            try:
                if first_video_total_count != 0:
                    self.initial_delay = max(10, int((30 * last_values[(index + 1) % len(self.video_files)]) / first_video_total_count))
                else:
                    self.initial_delay = 10  # Default delay if first_video_total_count is zero
            except ZeroDivisionError:
                self.initial_delay = 10  # Default delay if division by zero occurs

            # Set orange color for active video
           # for _ in range(5):  # 5-second orange period
                #self.traffic_light_colors[active_video] = (0, 165, 255)
               # time.sleep(1)

            # Debug: Print current active video and initial delay
            print(f"Active Video: {active_video}, Initial Delay: {self.initial_delay}")

            # Move to the next video index
            index = (index + 1) % len(self.video_files)

    def display_video(self, capture, model, window_name, dataframe):
        while not self.terminate:
            ret, frame = capture.read()
            if not ret:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize frame to 600x400 for processing
            frame = cv2.resize(frame, (600, 400))

            # Perform object detection using YOLO model
            results = model(frame)
            vehicle_counts = self.count_objects(results[0].verbose())

            # Update dataframe with vehicle counts
            self.update_dataframe(dataframe, vehicle_counts)

            # Display annotated frame with traffic light color
            self.display_annotated_frame(frame, window_name, vehicle_counts)

    def count_objects(self, verbose_output):
        counts = {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0}

        output = verbose_output.split(',')
        for item in output:
            parts = item.strip().split()
            if len(parts) >= 2:
                count = parts[0]
                if count.isdigit():
                    obj_type = parts[-1][:-1] if count == '1' else parts[-1]
                    counts[obj_type] = int(count)

        return counts

    def update_dataframe(self, df, counts):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [current_time, counts['car'], counts['motorcycle'], counts['truck'], counts['bus'], sum(counts.values())]

    def display_annotated_frame(self, frame, window_name, vehicle_counts):
        annotated_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 2
        y_offset = 20

        for i, (vehicle_type, count) in enumerate(vehicle_counts.items()):
            cv2.putText(annotated_frame, f"{vehicle_type}: {count}", (10, y_offset*(i+1)), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Get traffic light color for the window name
        color = self.traffic_light_colors.get(window_name, (0, 0, 0))

        # Draw traffic light circle with color
        radius = 20
        center_coordinates = (annotated_frame.shape[1] - radius - 10, radius + 10)
        cv2.circle(annotated_frame, center_coordinates, radius, color, -1)

        delay_text = str(self.initial_delay)
        text_size = cv2.getTextSize(delay_text, font, font_scale, thickness)[0]
        text_x = center_coordinates[0] - text_size[0] // 2
        text_y = center_coordinates[1] + text_size[1] // 2
        cv2.putText(annotated_frame, delay_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Display annotated frame
        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            self.terminate=True
        


if __name__ == "__main__":
    # Specify video file paths
    #video_files = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
    video_files=sys.argv[1:]
    # Create and start video processor
    processor = VideoProcessor(video_files)
    processor.start()
