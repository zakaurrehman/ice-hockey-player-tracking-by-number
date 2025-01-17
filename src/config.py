"""Configuration settings for hockey player tracking"""
import torch

# GPU Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(DEVICE)
CUDA_AVAILABLE = torch.cuda.is_available()

# Early GPU memory allocation settings
GPU_MEMORY_FRACTION = 0.8
TORCH_BACKENDS_CUDNN_BENCHMARK = True

# Model settings
YOLO_MODEL = 'yolov8s.pt'  # Smaller model for better speed
YOLO_CONF = 0.5  # Confidence threshold for detection

# Video processing settings
BATCH_SIZE = 4  # Process frames in batches
SKIP_FRAMES = 1  # Process every frame for accuracy
MIN_CLIP_DURATION = 0.5  # Minimum clip duration in seconds
MAX_MISSED_FRAMES = 10  # Max consecutive frames without detection

# Detection settings
PERSON_CONF_THRESHOLD = 0.3  # Threshold for detecting "person" class
OCR_CONFIDENCE = 0.5  # Threshold for OCR confidence
COLOR_THRESHOLD = 300  # Minimum area for a jersey color match

# Debug settings
SAVE_DEBUG_FRAMES = True  # Save frames for debugging
DEBUG_FRAME_INTERVAL = 100  # Save every 100th frame during debugging
DEBUG_OUTPUT_DIR = "debug_frames"

# Jersey colors in HSV format
JERSEY_COLORS = {
    'black-jersey': [
        ((0, 0, 0), (180, 255, 50)),      # Pure black
        ((0, 0, 0), (180, 100, 40))       # Dark black with some reflection
    ],
    'white-red-jersey': [
        ((0, 0, 180), (180, 30, 255)),    # White base
        ((0, 150, 150), (10, 255, 255)),  # Red accents
        ((170, 150, 150), (180, 255, 255))  # Wrapped red
    ],
    'white-jersey': [
        ((0, 0, 180), (180, 30, 255)),    # Pure white
        ((0, 0, 150), (180, 40, 255))     # Off-white
    ],
    'black-white-numbers': [
        ((0, 0, 0), (180, 255, 30)),      # Black base
        ((0, 0, 200), (180, 30, 255))     # White numbers
    ],
    'game-red': [
        ((0, 100, 100), (10, 255, 255)),    # Bright red
        ((170, 100, 100), (180, 255, 255))  # Wrapped red
    ],
    'game-blue': [
        ((85, 30, 30), (110, 255, 255)),    # Teal/Light blue
        ((100, 20, 30), (130, 255, 255)),   # Blue
        ((80, 20, 30), (120, 255, 255))     # Wider blue range
    ],
    'game-black': [
        ((0, 0, 0), (180, 100, 30)),        # Black with arena lighting
        ((0, 0, 0), (180, 150, 40))         # Black with reflections
    ],
    'black-light-blue': [  # Black with light blue accents
        ((0, 0, 0), (180, 255, 50)),        # Black base
        ((85, 60, 60), (110, 255, 255))     # Light blue accents
    ],
    'white-red-blue': [  # New color: White with red and blue accents
        ((0, 0, 180), (180, 30, 255)),      # White base
        ((0, 150, 150), (10, 255, 255)),    # Red accents
        ((170, 150, 150), (180, 255, 255)), # Wrapped red
        ((100, 50, 50), (140, 255, 255))    # Blue accents
    ]
}

# OCR settings
OCR_SETTINGS = {
    'batch_size': 1,
    'min_size': 10,
    'text_threshold': 0.7,
    'link_threshold': 0.4,
    'low_text': 0.4,
    'canvas_size': 2560,
    'mag_ratio': 1.5
}

# Image processing settings
IMAGE_PROCESSING = {
    'resize_factor': 4,
    'contrast_alpha': 1.3,
    'contrast_beta': 20,
    'blur_kernel': (3, 3),
    'threshold_block_size': 11,
    'threshold_C': 2
}

# Padding settings for bounding boxes
BBOX_PADDING = {
    'top': 20,
    'bottom': 20,
    'left': 10,
    'right': 10
}

# Video output settings
VIDEO_OUTPUT = {
    'codec': 'mp4v',
    'fps_factor': 1.0,
    'min_clip_frames': 15,
    'max_clip_gap': 10
}

# Debug visualization colors (BGR format)
DEBUG_COLORS = {
    'detected': (0, 255, 0),     # Green
    'color_match': (0, 165, 255),  # Orange
    'no_match': (0, 0, 255),     # Red
    'text_color': (255, 255, 255)  # White
}
