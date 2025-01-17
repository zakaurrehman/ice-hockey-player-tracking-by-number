# import torch
# from ultralytics import YOLO
# import cv2
# import os
# import numpy as np
# import easyocr
# from .config import JERSEY_COLORS
# from .utils import is_jersey_color, enhance_number_region


# class EnhancedPlayerTracker:
#     def __init__(self, jersey_number, jersey_color, conf_threshold=0.5, min_clip_duration=1.5):
#         """
#         Initialize the EnhancedPlayerTracker.
#         :param jersey_number: Player's jersey number to track
#         :param jersey_color: Player's jersey color to track
#         :param conf_threshold: Confidence threshold for detection
#         :param min_clip_duration: Minimum clip duration to consider
#         """
#         # GPU setup
#         if torch.cuda.is_available():
#             device = 'cuda'
#         else:
#             device = 'cpu'
#             print("Warning: CUDA is not available. Falling back to CPU.")

#         print(f"Initializing Enhanced Player Tracker on {device.upper()}...")

#         # Load and optimize YOLO model
#         self.model = YOLO('yolov8s.pt')
#         self.model.to(device)
#         self.model.fuse()
#         print(f"YOLOv8 model loaded and optimized for {device.upper()}.")

#         # Tracker settings
#         self.device = device
#         self.jersey_number = str(jersey_number)
#         self.jersey_color = jersey_color
#         self.conf_threshold = conf_threshold
#         self.min_clip_duration = min_clip_duration
#         self.ocr = easyocr.Reader(['en'], gpu=(device == 'cuda'))

#         # Continuous tracking parameters
#         self.MISSED_FRAME_THRESHOLD = 30  # Allow more missed frames for continuity
#         self.FRAME_BUFFER = 15  # Frames to keep before/after detections
#         self.MAX_MERGE_GAP = 60  # Maximum frames between segments to merge

#         print(f"Tracker initialized:\n"
#               f" - Device: {device}\n"
#               f" - Jersey number: {jersey_number}\n"
#               f" - Jersey color: {jersey_color}\n"
#               f" - Confidence threshold: {conf_threshold}\n"
#               f" - Min clip duration: {min_clip_duration}s")

#     def detect_players(self, frames):
#         """
#         Detect players in a batch of frames using YOLO.
#         :param frames: List of input video frames
#         :return: List of detections for each frame
#         """
#         results = self.model(frames, conf=self.conf_threshold, imgsz=640)
#         all_detections = []

#         for result in results:
#             detections = []
#             if result.boxes:
#                 for box in result.boxes:
#                     bbox = box.xyxy.cpu().numpy().astype(int).flatten()
#                     conf = box.conf.cpu().numpy().item()
#                     detections.append((bbox, conf))
#             all_detections.append(detections)
#         return all_detections

#     def is_target_player(self, frame, bbox):
#         """
#         Check if a bounding box matches the target player's jersey color and number.
#         :param frame: Input frame
#         :param bbox: Bounding box coordinates
#         :return: True if it matches the target player, False otherwise
#         """
#         if is_jersey_color(frame, bbox, self.jersey_color):
#             detected_number = self.detect_jersey_number(frame, bbox)
#             if detected_number == self.jersey_number:
#                 return True
#         return False

#     def detect_jersey_number(self, frame, bbox):
#         """
#         Detect the jersey number using OCR.
#         :param frame: Input frame
#         :param bbox: Bounding box coordinates
#         :return: Detected jersey number as a string
#         """
#         try:
#             x1, y1, x2, y2 = map(int, bbox)
#             player_region = frame[y1:y2, x1:x2]
#             enhanced_region = enhance_number_region(player_region)

#             results = self.ocr.readtext(enhanced_region)
#             for _, text, conf in results:
#                 if text.isdigit() and conf > 0.6:  # High confidence threshold
#                     return text
#             return None
#         except Exception as e:
#             print(f"Error in jersey number detection: {str(e)}")
#             return None

#     def _merge_segments(self, segments):
#         """
#         Merge tracking segments that are close together.
#         :param segments: List of tracking segments
#         :return: List of merged segments
#         """
#         if not segments:
#             return []
        
#         merged = [segments[0]]
#         for current in segments[1:]:
#             previous = merged[-1]
            
#             if current['start'] - previous['end'] <= self.MAX_MERGE_GAP:
#                 previous['end'] = current['end']
#                 previous['frames'].extend(current['frames'])
#             else:
#                 merged.append(current)
        
#         return merged

#     def _create_continuous_video(self, input_path, output_path, segments, fps, width, height):
#         """
#         Create a continuous video from tracking segments.
#         :param input_path: Path to input video
#         :param output_path: Path to output video
#         :param segments: List of tracking segments
#         :param fps: Frames per second
#         :param width: Video width
#         :param height: Video height
#         """
#         cap = cv2.VideoCapture(input_path)
#         writer = cv2.VideoWriter(
#             output_path,
#             cv2.VideoWriter_fourcc(*'mp4v'),
#             fps,
#             (width, height)
#         )

#         for segment in segments:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start'])
            
#             for frame_number in range(segment['start'], segment['end'] + 1):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Add visualization for detected frames
#                 frame_info = next(
#                     (f for f in segment['frames'] if f['frame_number'] == frame_number),
#                     None
#                 )
#                 if frame_info:
#                     x1, y1, x2, y2 = map(int, frame_info['bbox'])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"#{self.jersey_number}", 
#                               (x1, y1 - 10), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 
#                               0.9, (0, 255, 0), 2)
                
#                 writer.write(frame)

#         cap.release()
#         writer.release()

#     def track_player(self, video_path, output_dir, progress_signal):
#         """
#         Track the player across all frames and create continuous video.
#         :param video_path: Path to the input video
#         :param output_dir: Directory to save the output video
#         :param progress_signal: Signal to update progress bar
#         :return: Path to the output video
#         """
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print("Error: Unable to open video file.")
#             return None

#         # Get video properties
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
#         output_video_path = os.path.join(output_dir, f"player_{self.jersey_number}_continuous.mp4")

#         # Initialize tracking variables
#         frame_count = 0
#         batch_size = 4  # Process frames in batches
#         frames = []
#         tracking_segments = []
#         current_segment = None
#         missed_frames = 0
#         last_detection_frame = None

#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame_count += 1
#                 if frame_count % 2 == 0:  # Skip alternate frames for speed
#                     continue

#                 frames.append(frame)

#                 # Process batch
#                 if len(frames) == batch_size or frame_count == total_frames:
#                     batch_detections = self.detect_players(frames)
                    
#                     for idx, detections in enumerate(batch_detections):
#                         current_frame = frames[idx]
#                         current_frame_number = frame_count - batch_size + idx + 1
#                         player_detected = False

#                         # Check each detection
#                         for bbox, conf in detections:
#                             if self.is_target_player(current_frame, bbox):
#                                 player_detected = True
#                                 last_detection_frame = current_frame_number
                                
#                                 # Start new segment or continue current
#                                 if current_segment is None:
#                                     current_segment = {
#                                         'start': max(0, current_frame_number - self.FRAME_BUFFER),
#                                         'frames': []
#                                     }
                                
#                                 current_segment['frames'].append({
#                                     'frame_number': current_frame_number,
#                                     'bbox': bbox
#                                 })
#                                 break

#                         if not player_detected:
#                             missed_frames += 1
                            
#                             if missed_frames >= self.MISSED_FRAME_THRESHOLD and current_segment is not None:
#                                 current_segment['end'] = min(
#                                     total_frames,
#                                     last_detection_frame + self.FRAME_BUFFER
#                                 )
#                                 if len(current_segment['frames']) > 0:
#                                     tracking_segments.append(current_segment)
#                                 current_segment = None
#                         else:
#                             missed_frames = 0

#                         # Update progress
#                         progress = int((current_frame_number / total_frames) * 100)
#                         progress_signal.emit(progress)

#                     frames = []

#             # Handle final segment
#             if current_segment is not None:
#                 current_segment['end'] = total_frames
#                 if len(current_segment['frames']) > 0:
#                     tracking_segments.append(current_segment)

#             # Merge segments and create final video
#             merged_segments = self._merge_segments(tracking_segments)
#             self._create_continuous_video(video_path, output_video_path, 
#                                        merged_segments, fps, width, height)

#         except Exception as e:
#             print(f"Error during tracking: {str(e)}")
#             return None

#         finally:
#             cap.release()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         return output_video_path if os.path.exists(output_video_path) else None

#     def __del__(self):
#         """Cleanup when tracker is destroyed"""
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
import easyocr
from collections import deque
from .config import JERSEY_COLORS
from .utils import is_jersey_color, enhance_number_region


class EnhancedPlayerTracker:
    def __init__(self, jersey_number, jersey_color, conf_threshold=0.5, min_clip_duration=1.5):
        """
        Initialize the EnhancedPlayerTracker with improved features.
        """
        # GPU setup with error handling
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            else:
                device = 'cpu'
                print("Warning: CUDA is not available. Falling back to CPU.")
        except Exception as e:
            device = 'cpu'
            print(f"Error initializing GPU: {e}. Falling back to CPU.")

        self.device = device
        print(f"Initializing Enhanced Player Tracker on {device.upper()}...")

        # Load and optimize YOLO model
        try:
            self.model = YOLO('yolov8s.pt')
            self.model.to(device)
            if device == 'cuda':
                self.model.fuse()  # Fuse layers for GPU optimization
            print(f"YOLOv8 model loaded and optimized for {device.upper()}.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

        # Tracker settings
        self.jersey_number = str(jersey_number)
        self.jersey_color = jersey_color
        self.conf_threshold = conf_threshold
        self.min_clip_duration = min_clip_duration
        
        # Initialize OCR with error handling
        try:
            self.ocr = easyocr.Reader(['en'], gpu=(device == 'cuda'))
        except Exception as e:
            print(f"Error initializing OCR: {e}")
            raise

        # Enhanced tracking parameters
        self.MISSED_FRAME_THRESHOLD = 45  # Increased for better continuity
        self.FRAME_BUFFER = 20  # Increased buffer for smoother transitions
        self.MAX_MERGE_GAP = 90  # Increased gap for merging segments
        self.SMOOTHING_ALPHA = 0.7  # Bbox smoothing factor
        self.tracking_history = deque(maxlen=30)  # Track history for smoothing
        
        # Dynamic batch size based on available memory
        self.batch_size = self._optimize_batch_size()

        print(f"Tracker initialized with enhanced settings:\n"
              f" - Device: {device}\n"
              f" - Jersey number: {jersey_number}\n"
              f" - Jersey color: {jersey_color}\n"
              f" - Batch size: {self.batch_size}\n"
              f" - Confidence threshold: {conf_threshold}\n"
              f" - Min clip duration: {min_clip_duration}s")

    def _optimize_batch_size(self):
        """Dynamically adjust batch size based on available memory"""
        try:
            if self.device == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = gpu_memory - torch.cuda.memory_allocated()
                optimal_batch = max(1, min(8, int((available_memory / 1024**3) * 2)))
                return optimal_batch
            return 4  # Default CPU batch size
        except Exception:
            return 4  # Fallback batch size

    def _smooth_bbox(self, prev_bbox, curr_bbox):
        """Smooth bounding box transitions"""
        if prev_bbox is None:
            return curr_bbox
        return tuple(int(self.SMOOTHING_ALPHA * p + (1 - self.SMOOTHING_ALPHA) * c) 
                    for p, c in zip(prev_bbox, curr_bbox))

    def _remove_duplicates(self, detections, iou_threshold=0.5):
        """Remove duplicate detections using NMS"""
        if not detections:
            return []
            
        # Convert to numpy arrays for easier processing
        boxes = np.array([d[0] for d in detections])
        scores = np.array([d[1] for d in detections])
        
        # Calculate areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence score
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            # Calculate IoU with rest of boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = np.where(ovr <= iou_threshold)[0]
            order = order[ids + 1]
            
        return [detections[i] for i in keep]

    def detect_players(self, frames):
        """Multi-scale player detection"""
        scales = [0.8, 1.0, 1.2]
        all_detections = []
        
        for frame in frames:
            frame_detections = []
            
            for scale in scales:
                scaled_size = int(640 * scale)
                try:
                    results = self.model(frame, conf=self.conf_threshold, imgsz=scaled_size)
                    
                    if results[0].boxes:
                        for box in results[0].boxes:
                            bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                            conf = box.conf.cpu().numpy().item()
                            frame_detections.append((bbox, conf))
                except Exception as e:
                    print(f"Error in detection at scale {scale}: {e}")
                    continue
            
            # Remove duplicates and add to final detections
            frame_detections = self._remove_duplicates(frame_detections)
            all_detections.append(frame_detections)
        
        return all_detections

    def detect_jersey_number(self, frame, bbox):
        """Enhanced jersey number detection with multiple preprocessing"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            player_region = frame[y1:y2, x1:x2]
            
            # Multiple preprocessing approaches
            preprocessed_versions = [
                enhance_number_region(player_region),
                cv2.convertScaleAbs(player_region, alpha=1.5, beta=30),
                cv2.GaussianBlur(player_region, (3,3), 0),
                cv2.detailEnhance(player_region, sigma_s=10, sigma_r=0.15)
            ]
            
            for enhanced_region in preprocessed_versions:
                results = self.ocr.readtext(enhanced_region)
                for _, text, conf in results:
                    if text.isdigit() and conf > 0.6:
                        # Check for exact match
                        if text == self.jersey_number:
                            return text
                        # Check for similar numbers (e.g., 18 vs 81)
                        if sorted(text) == sorted(self.jersey_number):
                            return text
            return None
            
        except Exception as e:
            print(f"Error in jersey number detection: {str(e)}")
            return None

    def _interpolate_missing_frames(self, prev_frame, next_frame, num_frames):
        """Interpolate bounding boxes for missing frames"""
        if prev_frame is None or next_frame is None:
            return []
            
        prev_bbox = prev_frame['bbox']
        next_bbox = next_frame['bbox']
        
        interpolated_frames = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            interpolated_bbox = tuple(
                int(prev_bbox[j] + t * (next_bbox[j] - prev_bbox[j]))
                for j in range(4)
            )
            interpolated_frames.append({
                'frame_number': prev_frame['frame_number'] + i + 1,
                'bbox': interpolated_bbox
            })
        
        return interpolated_frames

    def _add_visualization(self, frame, detection_info, thickness=2):
        """Enhanced visualization with smooth graphics"""
        if detection_info:
            x1, y1, x2, y2 = map(int, detection_info['bbox'])
            
            # Add dark overlay for better text visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1-2, y1-25), (x1+70, y1), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add bounding box with gradient effect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), thickness//2)
            
            # Add player number with shadow effect
            text = f"#{self.jersey_number}"
            cv2.putText(frame, text, (x1+2, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            return frame
        return frame

    def _merge_segments(self, segments):
        """Merge tracking segments with interpolation"""
        if not segments:
            return []
        
        merged = [segments[0]]
        for current in segments[1:]:
            previous = merged[-1]
            gap = current['start'] - previous['end']
            
            if gap <= self.MAX_MERGE_GAP:
                # Interpolate frames in the gap
                if gap > 1:
                    last_frame = previous['frames'][-1]
                    first_frame = current['frames'][0]
                    interpolated = self._interpolate_missing_frames(
                        last_frame, first_frame, gap-1)
                    previous['frames'].extend(interpolated)
                
                previous['end'] = current['end']
                previous['frames'].extend(current['frames'])
            else:
                merged.append(current)
        
        return merged

    def track_player(self, video_path, output_dir, progress_signal):
        """
        Enhanced player tracking with improved continuity and visualization.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output directory and video writer
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"player_{self.jersey_number}_enhanced.mp4")
        
        # Initialize tracking variables
        frame_count = 0
        frames = []
        tracking_segments = []
        current_segment = None
        missed_frames = 0
        last_detection_frame = None
        prev_bbox = None

        try:
            writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frames.append(frame)

                # Process in optimized batches
                if len(frames) == self.batch_size or frame_count == total_frames:
                    batch_detections = self.detect_players(frames)
                    
                    for idx, detections in enumerate(batch_detections):
                        current_frame = frames[idx]
                        current_frame_number = frame_count - len(frames) + idx + 1
                        player_detected = False

                        # Check each detection
                        for bbox, conf in detections:
                            if self.is_target_player(current_frame, bbox):
                                player_detected = True
                                last_detection_frame = current_frame_number
                                
                                # Smooth bbox transition
                                smoothed_bbox = self._smooth_bbox(prev_bbox, bbox)
                                prev_bbox = smoothed_bbox
                                
                                # Update or create segment
                                if current_segment is None:
                                    current_segment = {
                                        'start': max(0, current_frame_number - self.FRAME_BUFFER),
                                        'frames': []
                                    }
                                
                                current_segment['frames'].append({
                                    'frame_number': current_frame_number,
                                    'bbox': smoothed_bbox
                                })
                                break

                        if not player_detected:
                            missed_frames += 1
                            
                            if missed_frames >= self.MISSED_FRAME_THRESHOLD and current_segment is not None:
                                current_segment['end'] = min(
                                    total_frames,
                                    last_detection_frame + self.FRAME_BUFFER
                                )
                                if len(current_segment['frames']) > 0:
                                    tracking_segments.append(current_segment)
                                current_segment = None
                                prev_bbox = None
                        else:
                            missed_frames = 0

                        # Update progress
                        progress = int((current_frame_number / total_frames) * 100)
                        progress_signal.emit(progress)

                    frames = []

            # Handle final segment
            if current_segment is not None:
                current_segment['end'] = total_frames
                if len(current_segment['frames']) > 0:
                    tracking_segments.append(current_segment)

            # Merge segments and create final video
            merged_segments = self._merge_segments(tracking_segments)
            
            # Reset video capture for final video creation
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            
            print("\nCreating enhanced continuous video...")
            for segment in merged_segments:
                # Skip to segment start
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start'])
                
                for frame_number in range(segment['start'], segment['end'] + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Find detection info for current frame
                    detection_info = next(
                        (f for f in segment['frames'] if f['frame_number'] == frame_number),
                        None
                    )
                    
                    # Add visualization if frame has detection
                    frame = self._add_visualization(frame, detection_info)
                    writer.write(frame)
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"Processing frame {frame_count}/{total_frames}")
                        
            print(f"\nVideo processing complete. Output saved to: {output_video_path}")

        except Exception as e:
            print(f"Error during tracking: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            cap.release()
            writer.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return output_video_path if os.path.exists(output_video_path) else None

    def is_target_player(self, frame, bbox):
        """
        Enhanced target player detection with color and number verification.
        """
        try:
            if is_jersey_color(frame, bbox, self.jersey_color):
                detected_number = self.detect_jersey_number(frame, bbox)
                if detected_number == self.jersey_number:
                    return True
            return False
        except Exception as e:
            print(f"Error in target player detection: {str(e)}")
            return False

    def __del__(self):
        """Cleanup resources when tracker is destroyed"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

            
            #