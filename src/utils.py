"""Utility functions for the player tracker"""
import cv2
import numpy as np
import os
from .config import JERSEY_COLORS, DEBUG_COLORS, DEBUG_OUTPUT_DIR, SAVE_DEBUG_FRAMES


def print_gpu_utilization():
    """Print GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory allocated: {allocated:.2f} MB")
            print(f"GPU Memory reserved: {reserved:.2f} MB")
    except Exception as e:
        print(f"Error checking GPU utilization: {str(e)}")


def is_jersey_color(frame, bbox, color_name, color_threshold=200):
    """
    Check if the bounding box contains the specified jersey color.
    """
    try:
        # Verify JERSEY_COLORS is properly imported
        if color_name not in JERSEY_COLORS:
            print(f"Warning: Unsupported jersey color '{color_name}'")
            return False

        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)

        if x2 <= x1 or y2 <= y1:  # Invalid bbox dimensions
            return False

        cropped = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        for lower, upper in JERSEY_COLORS[color_name]:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            match_percentage = (np.sum(mask) / mask.size) * 100
            if match_percentage > 5:  # Adjust threshold as needed
                return True

        return False
    except Exception as e:
        print(f"Error in jersey color detection: {str(e)}")
        return False


def enhance_number_region(bbox):
    """
    Enhanced number region processing with multiple techniques.
    Returns the enhanced version of the bounding box.
    """
    try:
        if bbox is None or bbox.size == 0:
            return None

        # Basic enhancement: Resize and increase contrast
        resized = cv2.resize(bbox, (bbox.shape[1] * 4, bbox.shape[0] * 4))
        enhanced = cv2.convertScaleAbs(resized, alpha=1.5, beta=30)

        return enhanced
    except Exception as e:
        print(f"Error enhancing number region: {str(e)}")
        return bbox


def save_debug_image(image, path, label=""):
    """
    Save debug image with an optional label.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if label:
            cv2.putText(
                image,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                DEBUG_COLORS['text_color'],
                2,
            )
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving debug image: {str(e)}")


def create_debug_visualization(frame, bbox, color_match, number_match, jersey_number):
    """
    Create a debug visualization of the detection.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        debug_frame = frame.copy()

        # Determine color based on match type
        if number_match:
            color = DEBUG_COLORS['detected']
            label = f"#{jersey_number}"
        elif color_match:
            color = DEBUG_COLORS['color_match']
            label = "Color Match"
        else:
            color = DEBUG_COLORS['no_match']
            label = "No Match"

        # Draw bounding box and add label
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return debug_frame
    except Exception as e:
        print(f"Error creating debug visualization: {str(e)}")
        return frame
