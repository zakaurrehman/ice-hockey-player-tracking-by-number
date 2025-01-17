# src/__init__.py

"""Hockey player tracking package"""
import torch
from .config import DEVICE, TORCH_DEVICE, CUDA_AVAILABLE

def initialize_gpu():
    """Initialize GPU settings if available."""
    if CUDA_AVAILABLE:
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print(f"GPU initialized: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")

initialize_gpu()

from .detector import PlayerDetector
from .enhanced_tracker import EnhancedPlayerTracker  # Correct import here
from .utils import print_gpu_utilization
from .video import extract_video_clip, get_video_info

__version__ = '1.0.0'
