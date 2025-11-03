import threading
import queue
import time
import datetime
import pandas as pd
import numpy as np
import cv2
import json
import logging
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import pyodbc
from dotenv import load_dotenv
from ultralytics import YOLO
from colorama import init, Fore, Back, Style
from tabulate import tabulate

init(autoreset=True)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_traffic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrafficDetection:
    """Data model for traffic detection with both signal and density status"""
    timestamp: datetime.datetime
    source_id: str
    frame_number: int
    car_count: int
    motorcycle_count: int
    truck_count: int
    bus_count: int
    total_count: int
    confidence_score: float
    processing_time_ms: float
    traffic_signal_status: str  # Simulated signal: 'GREEN', 'RED', 'YELLOW'
    traffic_density_status: str  # Density based: 'LOW', 'MEDIUM', 'HIGH'
    green_time_seconds: int
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TrafficDensity:
    """Determines traffic density status based on vehicle count"""
    LOW = "LOW"      # Green overlay
    MEDIUM = "MEDIUM"  # Orange overlay
    HIGH = "HIGH"     # Red overlay
    
    @staticmethod
    def get_status(vehicle_count: int) -> str:
        """Get traffic density status based on vehicle count"""
        if vehicle_count <= 5:
            return TrafficDensity.LOW
        elif vehicle_count <= 15:
            return TrafficDensity.MEDIUM
        else:
            return TrafficDensity.HIGH
    
    @staticmethod
    def get_color(status: str) -> Tuple[int, int, int]:
        """Get BGR color for density status"""
        if status == TrafficDensity.LOW:
            return (0, 255, 0)  # Green
        elif status == TrafficDensity.MEDIUM:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red


class DatabaseManager:
    """Manages MS SQL Server operations"""
    
    def __init__(self):
        self.server = os.getenv('SQL_SERVER', 'localhost')
        self.database = os.getenv('SQL_DATABASE', 'TrafficAnalytics')
        self.username = os.getenv('SQL_USERNAME', 'sa')
        self.password = os.getenv('SQL_PASSWORD', '123456')
        
        # Try different server name formats
        server_options = [self.server, 'localhost', '127.0.0.1', '.\\SQLEXPRESS', '(local)']
        
        self.connection_string = None
        self.connected = False
        
        for server_opt in server_options:
            test_conn_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server_opt};"
                f"DATABASE=master;"
                f"UID={self.username};"
                f"PWD={self.password};"
                f"TrustServerCertificate=yes;"
            )
            
            try:
                conn = pyodbc.connect(test_conn_string, timeout=3)
                conn.close()
                self.server = server_opt
                self.connection_string = test_conn_string
                self.connected = True
                print(f"{Fore.GREEN}✔ Connected using server: {server_opt}")
                break
            except:
                continue
        
        if not self.connected:
            print(f"{Fore.RED}✗ Could not connect to SQL Server")
            print(f"{Fore.YELLOW}Pipeline will use CSV output only")
        else:
            self.init_database()
        
        self.total_inserted = 0
    
    def init_database(self):
        """Initialize database and tables"""
        if not self.connected:
            return
            
        try:
            conn = pyodbc.connect(self.connection_string, timeout=5)
            cursor = conn.cursor()
            
            # Create database if not exists
            cursor.execute(f"""
                IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = '{self.database}')
                CREATE DATABASE {self.database};
            """)
            conn.commit()
            
            # Update connection string
            self.connection_string = self.connection_string.replace("DATABASE=master", f"DATABASE={self.database}")
            
            conn.close()
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Drop and recreate tables
            cursor.execute("DROP TABLE IF EXISTS integrated_traffic_detections")
            cursor.execute("DROP TABLE IF EXISTS traffic_light_cycles")
            
            # Create integrated detection table
            cursor.execute("""
                CREATE TABLE integrated_traffic_detections (
                    id BIGINT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2(3) NOT NULL,
                    source_id NVARCHAR(100),
                    frame_number INT,
                    car_count INT DEFAULT 0,
                    motorcycle_count INT DEFAULT 0,
                    truck_count INT DEFAULT 0,
                    bus_count INT DEFAULT 0,
                    total_count INT DEFAULT 0,
                    confidence_score DECIMAL(5,4),
                    processing_time_ms DECIMAL(10,2),
                    traffic_signal_status NVARCHAR(20),
                    traffic_density_status NVARCHAR(20),
                    green_time_seconds INT,
                    
                    INDEX IX_timestamp (timestamp),
                    INDEX IX_source (source_id),
                    INDEX IX_signal_status (traffic_signal_status),
                    INDEX IX_density_status (traffic_density_status)
                );
            """)
            
            # Create traffic light cycles table
            cursor.execute("""
                CREATE TABLE traffic_light_cycles (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    cycle_start DATETIME2 DEFAULT GETDATE(),
                    cycle_end DATETIME2,
                    source_id NVARCHAR(200),
                    green_duration_seconds INT,
                    total_vehicles_passed INT,
                    avg_vehicle_rate DECIMAL(10,2),
                    avg_density_status NVARCHAR(20)
                );
            """)
            
            conn.commit()
            conn.close()
            print(f"{Fore.GREEN}✔ Database initialization complete")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Database initialization error: {e}")
            self.connected = False
    
    def bulk_insert(self, detections: List[TrafficDetection]):
        """Bulk insert detections into database"""
        if not detections or not self.connected:
            return
        
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            for detection in detections:
                formatted_timestamp = detection.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                cursor.execute("""
                    INSERT INTO integrated_traffic_detections 
                    (timestamp, source_id, frame_number, car_count, motorcycle_count, 
                     truck_count, bus_count, total_count, confidence_score, processing_time_ms,
                     traffic_signal_status, traffic_density_status, green_time_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    formatted_timestamp,
                    detection.source_id,
                    detection.frame_number,
                    detection.car_count,
                    detection.motorcycle_count,
                    detection.truck_count,
                    detection.bus_count,
                    detection.total_count,
                    float(detection.confidence_score) if detection.confidence_score else 0.0,
                    float(detection.processing_time_ms) if detection.processing_time_ms else 0.0,
                    detection.traffic_signal_status,
                    detection.traffic_density_status,
                    detection.green_time_seconds
                ))
            
            conn.commit()
            self.total_inserted += len(detections)
            conn.close()
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            self.connected = False
    
    def log_traffic_cycle(self, source_id: str, green_duration: int, vehicles_passed: int, avg_density: str):
        """Log completed traffic light cycle"""
        if not self.connected:
            return
            
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            avg_rate = vehicles_passed / green_duration if green_duration > 0 else 0
            
            cursor.execute("""
                INSERT INTO traffic_light_cycles 
                (source_id, green_duration_seconds, total_vehicles_passed, avg_vehicle_rate, avg_density_status, cycle_end)
                VALUES (?, ?, ?, ?, ?, GETDATE())
            """, (source_id, green_duration, vehicles_passed, avg_rate, avg_density))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging traffic cycle: {e}")


class VideoProcessor:
    """Processes video files with YOLO model"""
    
    def __init__(self, model_path: str = 'best.onnx'):
        try:
            self.model = YOLO(model_path, task='detect')
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            print(f"{Fore.GREEN}✔ YOLO model loaded from {model_path}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not load {model_path}, using default model")
            self.model = YOLO('yolov8n.pt')
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
    
    def process_frame(self, frame):
        """Process a single frame with YOLO and return detections"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        # Extract vehicle counts and bounding boxes
        counts = {'car': 0, 'motorcycle': 0, 'truck': 0, 'bus': 0}
        confidence_scores = []
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                if hasattr(boxes, 'cls') and hasattr(boxes, 'conf') and hasattr(boxes, 'xyxy'):
                    classes = boxes.cls.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    bboxes = boxes.xyxy.cpu().numpy()
                    
                    for cls_id, conf, bbox in zip(classes, confidences, bboxes):
                        class_name = self.class_names.get(int(cls_id), '').lower()
                        vehicle_type = None
                        
                        # Map to vehicle types
                        if 'car' in class_name or int(cls_id) == 2:
                            counts['car'] += 1
                            vehicle_type = 'car'
                        elif 'motorcycle' in class_name or 'bike' in class_name or int(cls_id) == 3:
                            counts['motorcycle'] += 1
                            vehicle_type = 'motorcycle'
                        elif 'truck' in class_name or int(cls_id) == 7:
                            counts['truck'] += 1
                            vehicle_type = 'truck'
                        elif 'bus' in class_name or int(cls_id) == 5:
                            counts['bus'] += 1
                            vehicle_type = 'bus'
                        
                        if vehicle_type:
                            detections.append({
                                'bbox': bbox,
                                'type': vehicle_type,
                                'confidence': conf
                            })
                        
                        confidence_scores.append(conf)
        
        processing_time = (time.time() - start_time) * 1000
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        total_count = sum(counts.values())
        
        return counts, total_count, avg_confidence, processing_time, detections


class IntegratedTrafficSystem:
    """Main integrated traffic control and monitoring system"""
    
    def __init__(self, video_paths: List[str], initial_delay: int = 30):
        self.video_paths = video_paths
        self.initial_delay = initial_delay
        self.current_delay = initial_delay
        self.num_videos = len(video_paths)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.video_processor = VideoProcessor()
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Video captures
        self.video_captures = [cv2.VideoCapture(file) for file in video_paths]
        
        # Source IDs
        self.source_ids = {
            video_paths[i]: f"cam_{i+1}_{Path(video_paths[i]).stem}"
            for i in range(len(video_paths))
        }
        
        # Traffic signal states (simulation)
        self.signal_status = {
            source_id: "GREEN" if i == 0 else "RED"
            for i, source_id in enumerate(self.source_ids.values())
        }
        
        # Traffic density states (real-time)
        self.density_status = {
            source_id: "LOW"
            for source_id in self.source_ids.values()
        }
        
        # Statistics
        self.stats = {
            source_id: {
                'frames_processed': 0,
                'vehicles_detected': 0,
                'density_history': [],
                'green_cycles': 0,
                'total_green_time': 0
            }
            for source_id in self.source_ids.values()
        }
        
        # Control flags
        self.is_running = False
        self.current_green_index = 0
        
        # Display settings
        self.display_width = 800
        self.display_height = 600
        
        # Create output directory
        Path('pipeline_output').mkdir(exist_ok=True)
        
        self._print_startup_info()
    
    def _print_startup_info(self):
        """Print startup information"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🚦 INTEGRATED TRAFFIC CONTROL & MONITORING SYSTEM")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"\n{Fore.YELLOW}Features:")
        print(f"  ✓ Adaptive Traffic Signal Control (Simulation)")
        print(f"  ✓ Real-time Traffic Density Visualization")
        print(f"  ✓ Vehicle Detection & Tracking")
        print(f"  ✓ Database Recording")
        print(f"\n{Fore.YELLOW}Configuration:")
        print(f"  📹 Video Sources: {self.num_videos}")
        print(f"  ⏱️  Initial Green Time: {self.initial_delay} seconds")
        print(f"  💾 Database: {'Connected' if self.db_manager.connected else 'CSV only'}")
        
        print(f"\n{Fore.YELLOW}Camera Sources:")
        for video_path, source_id in self.source_ids.items():
            print(f"  • {Path(video_path).name} → {Fore.GREEN}{source_id}")
        
        print(f"\n{Fore.YELLOW}Controls:")
        print(f"  • Press 'Q' in video window to quit")
        print(f"  • Press 'S' to save current frame")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def draw_integrated_overlay(self, frame, source_id, vehicle_data):
        """Draw both traffic signal and density visualization"""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Get current states
        signal_status = self.signal_status[source_id]
        density_status = self.density_status[source_id]
        density_color = TrafficDensity.get_color(density_status)
        
        # Draw density-based border (10px thick)
        cv2.rectangle(annotated, (0, 0), (width, height), density_color, 10)
        
        # Draw top panel with transparency
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
        
        # Title and source
        cv2.putText(annotated, f"{source_id}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Traffic Signal Simulation (left side)
        signal_x = 15
        signal_y = 50
        cv2.putText(annotated, "SIGNAL:", (signal_x, signal_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw simulated traffic light
        light_radius = 20
        light_x = signal_x + 80
        light_y = signal_y - 10
        
        signal_color = (0, 255, 0) if signal_status == "GREEN" else (0, 0, 255)
        cv2.circle(annotated, (light_x, light_y), light_radius + 2, (100, 100, 100), -1)
        cv2.circle(annotated, (light_x, light_y), light_radius, signal_color, -1)
        
        # Show timer on green
        if signal_status == "GREEN":
            cv2.putText(annotated, str(self.current_delay), (light_x - 8, light_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Traffic Density Status (right side)
        density_x = width - 200
        density_y = 50
        cv2.putText(annotated, "DENSITY:", (density_x, density_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, density_status, (density_x + 80, density_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, density_color, 2)
        
        # Vehicle counts
        cv2.putText(annotated, f"Vehicles: {vehicle_data['total_count']}", 
                   (15, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        counts_text = f"Cars:{vehicle_data['car_count']} Trucks:{vehicle_data['truck_count']} " \
                     f"Buses:{vehicle_data['bus_count']} Bikes:{vehicle_data['motorcycle_count']}"
        cv2.putText(annotated, counts_text, (15, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw bounding boxes for detected vehicles
        for det in vehicle_data.get('detections', []):
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Use density color for boxes
            cv2.rectangle(annotated, (x1, y1), (x2, y2), density_color, 2)
            
            # Label
            label = f"{det['type']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), density_color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom statistics panel
        stats = self.stats[source_id]
        cv2.putText(annotated, f"Total Detected: {stats['vehicles_detected']}", 
                   (15, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Frames: {stats['frames_processed']}", 
                   (15, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, timestamp, (width - 100, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Status indicator
        status_text = "🟢 REC" if self.is_running else "⏸️ PAUSE"
        cv2.putText(annotated, status_text, (width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return annotated
    
    def adaptive_signal_controller(self):
        """Adaptive traffic signal controller thread"""
        index = 0
        
        while self.is_running:
            # Get current source
            current_video_path = self.video_paths[index]
            active_source_id = self.source_ids[current_video_path]
            
            # Update signal states
            for source_id in self.source_ids.values():
                if source_id == active_source_id:
                    self.signal_status[source_id] = "GREEN"
                else:
                    self.signal_status[source_id] = "RED"
            
            print(f"\n{Fore.GREEN}🟢 GREEN: {active_source_id} for {self.current_delay}s")
            
            # Track cycle
            cycle_start = time.time()
            vehicles_start = self.stats[active_source_id]['vehicles_detected']
            
            # Green phase
            time.sleep(self.current_delay)
            
            # Calculate vehicles passed
            vehicles_passed = self.stats[active_source_id]['vehicles_detected'] - vehicles_start
            avg_density = self.density_status[active_source_id]
            
            # Log cycle
            self.db_manager.log_traffic_cycle(active_source_id, self.current_delay, vehicles_passed, avg_density)
            
            # Update stats
            self.stats[active_source_id]['green_cycles'] += 1
            self.stats[active_source_id]['total_green_time'] += self.current_delay
            
            # Calculate next delay based on density
            self._calculate_adaptive_delay()
            
            # Move to next
            index = (index + 1) % self.num_videos
            self.current_green_index = index
    
    def _calculate_adaptive_delay(self):
        """Calculate adaptive delay based on traffic density"""
        # Collect density scores
        density_scores = []
        
        for source_id in self.source_ids.values():
            recent_density = self.stats[source_id]['density_history'][-10:] if self.stats[source_id]['density_history'] else []
            
            if recent_density:
                # Convert density to numeric score
                score = sum(3 if d == "HIGH" else 2 if d == "MEDIUM" else 1 for d in recent_density) / len(recent_density)
                density_scores.append(score)
            else:
                density_scores.append(1)
        
        # Adjust delay based on average density
        avg_score = np.mean(density_scores) if density_scores else 1
        
        if avg_score > 2.5:  # High traffic
            self.current_delay = min(60, self.initial_delay + 20)
        elif avg_score > 1.5:  # Medium traffic
            self.current_delay = self.initial_delay
        else:  # Low traffic
            self.current_delay = max(15, self.initial_delay - 10)
    
    def process_video(self, video_index: int):
        """Process individual video stream"""
        capture = self.video_captures[video_index]
        video_path = self.video_paths[video_index]
        source_id = self.source_ids[video_path]
        window_name = f"Traffic Monitor - {source_id}"
        
        # Position window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        x_pos = (video_index % 2) * (self.display_width + 10)
        y_pos = (video_index // 2) * (self.display_height + 50)
        cv2.moveWindow(window_name, x_pos, y_pos)
        
        frame_number = 0
        
        while self.is_running:
            ret, frame = capture.read()
            
            if not ret:
                # Loop video
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_number = 0
                continue
            
            # Resize for processing
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Process with YOLO
            counts, total_count, confidence, processing_time, detections = self.video_processor.process_frame(frame_resized)
            
            # Update density status
            self.density_status[source_id] = TrafficDensity.get_status(total_count)
            self.stats[source_id]['density_history'].append(self.density_status[source_id])
            
            # Keep history limited
            if len(self.stats[source_id]['density_history']) > 100:
                self.stats[source_id]['density_history'].pop(0)
            
            # Update statistics
            self.stats[source_id]['frames_processed'] += 1
            self.stats[source_id]['vehicles_detected'] += total_count
            
            # Create detection record
            detection = TrafficDetection(
                timestamp=datetime.datetime.now(),
                source_id=source_id,
                frame_number=frame_number,
                car_count=counts['car'],
                motorcycle_count=counts['motorcycle'],
                truck_count=counts['truck'],
                bus_count=counts['bus'],
                total_count=total_count,
                confidence_score=confidence,
                processing_time_ms=processing_time,
                traffic_signal_status=self.signal_status[source_id],
                traffic_density_status=self.density_status[source_id],
                green_time_seconds=self.current_delay if self.signal_status[source_id] == "GREEN" else 0
            )
            
            # Queue for database
            try:
                self.data_queue.put(detection, timeout=0.1)
            except queue.Full:
                logger.warning(f"Queue full for {source_id}")
            
            # Prepare vehicle data for display
            vehicle_data = {
                'total_count': total_count,
                'car_count': counts['car'],
                'truck_count': counts['truck'],
                'bus_count': counts['bus'],
                'motorcycle_count': counts['motorcycle'],
                'detections': detections
            }
            
            # Draw overlay and display
            display_frame = self.draw_integrated_overlay(frame_resized, source_id, vehicle_data)
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"pipeline_output/{source_id}_frame_{frame_number}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"{Fore.GREEN}✔ Saved: {filename}")
            
            frame_number += 1
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
        
        capture.release()
        cv2.destroyWindow(window_name)
    
    def database_worker(self):
        """Worker thread for database operations"""
        batch = []
        batch_size = 50
        last_save = time.time()
        csv_batch_count = 0
        
        while self.is_running or not self.data_queue.empty():
            try:
                detection = self.data_queue.get(timeout=1)
                batch.append(detection)
                
                if len(batch) >= batch_size or (time.time() - last_save) > 5:
                    if batch:
                        # Save to database
                        self.db_manager.bulk_insert(batch)
                        
                        # Save to CSV
                        df = pd.DataFrame([d.to_dict() for d in batch])
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        csv_path = f'pipeline_output/detections_{timestamp}_batch{csv_batch_count}.csv'
                        df.to_csv(csv_path, index=False)
                        
                        csv_batch_count += 1
                        batch = []
                        last_save = time.time()
                
            except queue.Empty:
                if batch:
                    self.db_manager.bulk_insert(batch)
                    batch = []
            except Exception as e:
                logger.error(f"Database worker error: {e}")
    
    def monitor_worker(self):
        """Display system statistics"""
        while self.is_running:
            time.sleep(5)
            
            # Clear console
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n{Back.BLUE}{Fore.WHITE} INTEGRATED TRAFFIC SYSTEM MONITOR {Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}")
            
            # Current status
            current_video = self.video_paths[self.current_green_index]
            current_green = self.source_ids[current_video]
            
            print(f"\n{Fore.YELLOW}🚦 Signal Status:")
            for source_id in self.source_ids.values():
                signal = "🟢" if self.signal_status[source_id] == "GREEN" else "🔴"
                density = self.density_status[source_id]
                density_icon = "🟩" if density == "LOW" else "🟧" if density == "MEDIUM" else "🟥"
                print(f"  {signal} {source_id}: Signal={self.signal_status[source_id]}, Density={density_icon} {density}")
            
            print(f"\n{Fore.YELLOW}📊 Statistics:")
            total_vehicles = 0
            for source_id in self.source_ids.values():
                stats = self.stats[source_id]
                total_vehicles += stats['vehicles_detected']
                print(f"  {source_id}:")
                print(f"    Vehicles: {stats['vehicles_detected']:,}")
                print(f"    Frames: {stats['frames_processed']:,}")
                print(f"    Green Cycles: {stats['green_cycles']}")
            
            print(f"\n{Fore.YELLOW}💾 Database:")
            print(f"  Records: {self.db_manager.total_inserted:,}")
            print(f"  Queue: {self.data_queue.qsize()}/{self.data_queue.maxsize}")
            print(f"  Status: {'Connected' if self.db_manager.connected else 'CSV Only'}")
            
            print(f"\n{Fore.GREEN}✔ System Running | Press 'Q' to stop")
            print(f"{Fore.CYAN}{'='*80}")
    
    def start(self):
        """Start the integrated system"""
        self.is_running = True
        threads = []
        
        print(f"{Fore.GREEN}▶ Starting Integrated Traffic System...")
        
        # Start signal controller
        signal_thread = threading.Thread(
            target=self.adaptive_signal_controller,
            name="SignalController"
        )
        signal_thread.daemon = True
        signal_thread.start()
        threads.append(signal_thread)
        
        # Start video processors
        for i in range(self.num_videos):
            thread = threading.Thread(
                target=self.process_video,
                args=(i,),
                name=f"VideoProcessor-{i+1}"
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Start database worker
        db_thread = threading.Thread(
            target=self.database_worker,
            name="DatabaseWorker"
        )
        db_thread.daemon = True
        db_thread.start()
        threads.append(db_thread)
        
        # Start monitor
        monitor_thread = threading.Thread(
            target=self.monitor_worker,
            name="SystemMonitor"
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        threads.append(monitor_thread)
        
        print(f"{Fore.GREEN}✔ All systems active with {len(threads)} threads")
        print(f"{Fore.YELLOW}Press 'Q' in video window to stop, 'S' to save frame")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠ Shutting down...")
            self.is_running = False
        
        # Wait for threads
        for thread in threads:
            thread.join(timeout=2)
        
        self.display_final_stats()
    
    def display_final_stats(self):
        """Display final statistics"""
        print(f"\n{Back.GREEN}{Fore.BLACK} SYSTEM SUMMARY {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}")
        
        total_vehicles = sum(s['vehicles_detected'] for s in self.stats.values())
        total_frames = sum(s['frames_processed'] for s in self.stats.values())
        
        print(f"\n{Fore.YELLOW}Overall:")
        print(f"  Total Vehicles: {total_vehicles:,}")
        print(f"  Total Frames: {total_frames:,}")
        print(f"  Database Records: {self.db_manager.total_inserted:,}")
        
        print(f"\n{Fore.YELLOW}Per Camera:")
        for source_id in self.source_ids.values():
            stats = self.stats[source_id]
            print(f"  {source_id}:")
            print(f"    Vehicles: {stats['vehicles_detected']:,}")
            print(f"    Green Cycles: {stats['green_cycles']}")
            if stats['green_cycles'] > 0:
                avg_time = stats['total_green_time'] / stats['green_cycles']
                print(f"    Avg Green Time: {avg_time:.1f}s")
        
        print(f"\n{Fore.GREEN}✔ System shutdown complete")
        print(f"📁 Data saved to: pipeline_output/")
        print(f"{Fore.CYAN}{'='*80}\n")


def main():
    """Main entry point"""
    video_files = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Validate video files
    valid_files = []
    for file in video_files:
        if Path(file).exists():
            valid_files.append(file)
            print(f"{Fore.GREEN}Found: {file}")
        else:
            print(f"{Fore.RED}Not found: {file}")
    
    if not valid_files:
        print(f"{Fore.RED}No valid video files found!")
        print(f"{Fore.YELLOW}Usage: python integrated_system.py video1.mp4 video2.mp4 ...")
        sys.exit(1)
    
    print(f"\n{Fore.CYAN}Ready to start with {len(valid_files)} video(s)")
    print(f"{Fore.CYAN}Press Enter to begin...")
    input()
    
    # Start the system
    system = IntegratedTrafficSystem(valid_files, initial_delay=30)
    system.start()


if __name__ == "__main__":
    main()