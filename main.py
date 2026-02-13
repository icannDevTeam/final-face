# Standard library imports
import os
import pickle
import random
import time
import tempfile
import hashlib
import logging
from datetime import datetime, time as dt_time
import argparse
from dotenv import load_dotenv
load_dotenv()
import threading
from collections import deque
import queue
import math

# Optional Firebase imports (attendance uploads)
try:
    from firebase_dataset_sync import initialize_firebase_app
    from firebase_admin import firestore as fb_firestore
    FIREBASE_ENABLED = True
except Exception:
    FIREBASE_ENABLED = False

# Optional Binus School API integration (attendance uploads)
try:
    import api_integrate
    API_INTEGRATE_ENABLED = True
except Exception:
    API_INTEGRATE_ENABLED = False

# Third-party imports
import cv2
import numpy as np
import json
import concurrent.futures
# SciPy removed — using numpy for distance calculations

# Try to import dlib for 68-point landmarks (HRNet removed)
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# Try to import GPU acceleration libraries
try:
    # this is only if you have NVIDIA gpu, then you can install then use cupy
    import cupy as cp 
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available, using CPU only")

# Firebase optional: local logging always, Firebase upload when env + SDK available

# Configuration settings with performance optimizations and best practices
CONFIG = {
    # Recognition settings - optimized for accuracy and security
    "tolerance": 0.4,           # Stricter tolerance for better security
    "frame_resize": 0.7,        # Higher quality for better accuracy
    "skip_frames": 1,           # Process more frames for better tracking
    "enhanced_facial_recognition": True,
    "use_landmark_embeddings": False,      # Landmark embeddings disabled for performance
    "use_cnn_embeddings": True,
    "embedding_mode": "cnn",            # landmarks | cnn | hybrid
    "embedding_component_weights": {"cnn": 1.0},
    "min_face_size": (70, 70),    # Minimum face size for detection
    "max_face_size": (400, 400),  # Maximum face size for processing
    "face_quality_threshold": 0.4, # Minimum quality score for face processing
    "brightness_threshold": (50, 200), # Acceptable brightness range
    "blur_threshold": 100,         # Minimum sharpness threshold
    
    # Multi-model ensemble settings
    "use_ensemble_models": True,     # Use multiple models for better accuracy
    "ensemble_threshold": 0.7,       # Confidence threshold for ensemble
    # Set to a dict to customize, or the string "default" to use built-ins
    "model_weights": "default",
    
    # Display settings
    "display_fps": True,
    "show_all_faces": True,
    "flip_camera": 1,
    "corner_display": True,       # Disabled for performance
    "show_landmarks": False,       # Disabled by default for speed
    
    # Attendance settings
    "latest_login_time": "07:30",
    "duplicate_detection_window": 300,  # 5 minutes window to prevent duplicates
    "upload_attendance_to_firebase": True,  # Upload daily JSON to Firebase Firestore when available
    "upload_attendance_to_api": True,         # Upload attendance to Binus School API when available
    
    # Performance settings
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "first_run_warning": True,
    "use_gpu_acceleration": GPU_AVAILABLE,
    "use_dlib_detector": DLIB_AVAILABLE,
    "detector_scale_factor": 1.1,  # More conservative for accuracy
    "detector_min_neighbors": 5,   # Higher for better accuracy
    
    # Advanced Performance settings
    "face_detection_threads": 2,
    "encoding_threads": 4,
    "recognition_cache_size": 500,  # Larger cache for better hit rate
    "preload_known_faces": True,
    "dynamic_quality_adjustment": True,
    "performance_monitoring": True,
    "target_fps": 25,                # More realistic target
    
    # PIN Authentication removed
    
    # Advanced Recognition settings - optimized for security
    "min_recognition_threshold": 0.60,     # Higher for better security
    "confident_recognition_threshold": 0.61, # Much higher for confident matches
    "use_gpu_if_available": True,
    "adaptive_processing": True,
    "max_parallel_recognitions": 2,        # Conservative for stability
    "face_tracking_enabled": True,
    "tracking_quality_threshold": 15,      # Higher for better tracking
    "max_tracking_age": 20,                # Conservative tracking age
    "recognition_voting_threshold": 2,     # Require multiple confirmations (was 5, 2 is better for ensemble)
    
    # Security and anti-spoofing
    "require_multiple_angles": True,       # Require face from multiple angles
    "liveness_detection": True,            # Enable liveness detection
    "texture_analysis": True,              # Analyze face texture for spoofing
    "motion_detection": True,              # Detect natural head motion
    "eye_movement_detection": True,        # Track eye movements
    "depth_analysis": True,                # Analyze face depth (if available)
    
    # Face quality assessment
    "enable_quality_assessment": True,     # Enable comprehensive quality checks
    "symmetry_threshold": 0.8,             # Face symmetry requirement
    "pose_angle_threshold": 30,            # Maximum pose angle in degrees
    "illumination_consistency": True,      # Check lighting consistency
    "resolution_check": True,              # Ensure adequate resolution
    
    # debugging
    "show_face_boxes": True,               # Enable for debugging
    "show_quality_metrics": True,          # Show quality assessment
    "log_recognition_details": True,       # Detailed logging

    # Enhanced high-performance settings
    "batch_processing_size": 8,            # Optimized batch size
    "use_fast_detector": True,
    "roi_optimization": True,
    "memory_optimization": True,
    "cascade_optimization": True,          # Use optimized cascade detection
    "parallel_face_detection": True,       # Parallel detection
    "smart_frame_skipping": True,          # Intelligent frame skipping
    "adaptive_roi": True,                  # Adaptive region of interest
    "temporal_consistency": True,          # Use temporal information
    
    # Anti-spoofing settings - enhanced
    "enable_blink_detection": True,        # Enable blink detection
    "ear_threshold": 0.25,                 # Eye aspect ratio threshold for blink
    "min_blinks_required": 3,              # More blinks required for security
    "blink_detection_time": 8,             # Longer time window
    "show_eye_landmarks": False,           # Show eye landmarks for debugging
    "micro_expression_detection": True,    # Detect micro-expressions
    "pulse_detection": True,               # Detect pulse from face
    "challenge_response": True,            # Random challenges (smile, nod, etc.)
}

# Stop the app after 8 hours (in seconds)
APP_TIMEOUT_SECONDS = 8 * 60 * 60

# Motivational quotes based on BINUS Values and IB Learner Profile
MOTIVATIONAL_QUOTES = [
"Strive for excellence.", "Embrace innovation", "Persevere daily.",
"Grow every day.", "Be honest.", "Respect others.", "Stay curious.",
"Think deeply.", "Be creative.", "Communicate well.", "Act with integrity.",
"Stay open-minded.", "Care for others.", "Take risks.", "Stay balanced.",
"Reflect often",
]

# Global variables
attendance = {}  # Track attendance to prevent duplicate entries
attendance_timestamps = {}  # Track when people last logged in
thank_you_message = {
    'active': False, 'name': '', 'time': 0,
    'duration': 3.0, 'quote': '', 'status': ''
}

# Security and quality assessment globals
face_quality_assessor = None
liveness_detector = None
challenge_response_active = {
    'active': False, 'challenge': '', 'start_time': 0,
    'person_name': '', 'class_name': '', 'required_action': '', 'completed': False
}

# Enhanced face tracking with quality metrics
face_quality_history = {}  # Track quality over time for each person

# Security monitoring and logging
security_monitor = {
    'failed_attempts': deque(maxlen=100),
    'suspicious_activity': deque(maxlen=50),
    'recognition_logs': deque(maxlen=1000),
    'system_alerts': deque(maxlen=100)
}

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_recognition_security.log'),
        logging.StreamHandler()
    ]
)
security_logger = logging.getLogger('FacialRecognitionSecurity')


def upload_attendance_to_firebase(local_path, date_only):
    """Upload the daily attendance JSON to Firebase Firestore."""
    if not (CONFIG.get("upload_attendance_to_firebase", False) and FIREBASE_ENABLED):
        return
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            day_data = json.load(f) or {}
        app = initialize_firebase_app()
        db = fb_firestore.client(app=app)
        db.collection("attendance").document(date_only).set(day_data, merge=True)
        print(f"☁️ Uploaded attendance to Firestore: attendance/{date_only}")
    except Exception as e:
        print(f"⚠️ Failed to upload attendance to Firestore: {e}")


def _upload_attendance_to_api(payload):
    """Upload a single attendance record to the Binus School API (runs in background thread)."""
    try:
        success = api_integrate.insert_student_attendance(payload)
        if success:
            print(f"☁️ API attendance uploaded: {payload.get('studentName', '?')}")
        else:
            print(f"⚠️ API attendance upload returned failure for {payload.get('studentName', '?')}")
    except Exception as e:
        print(f"⚠️ API attendance upload error: {e}")


# JSON serialization helper to handle NumPy types cleanly
def _json_fallback(obj):
    try:
        import numpy as _np  # local import to avoid global side effects
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        # Some numpy scalars expose .item()
        return obj.item()
    except Exception:
        return str(obj)

def log_security_event(event_type, details, severity='INFO'):
    """Log security events with structured data."""
    timestamp = datetime.now().isoformat()
    event_data = {
        'timestamp': timestamp,
        'event_type': event_type,
        'details': details,
        'severity': severity
    }
    
    security_monitor['system_alerts'].append(event_data)
    
    if severity == 'WARNING':
        security_logger.warning(f"{event_type}: {details}")
    elif severity == 'ERROR':
        security_logger.error(f"{event_type}: {details}")
    elif severity == 'CRITICAL':
        security_logger.critical(f"{event_type}: {details}")
    else:
        security_logger.info(f"{event_type}: {details}")

def detect_suspicious_activity(face_locations, face_names, face_confidences):
    """Detect patterns that might indicate spoofing or suspicious behavior."""
    current_time = time.time()
    
    # Multiple unknown faces might indicate spoofing attempts
    unknown_count = sum(1 for name in face_names if name == "Unknown")
    if unknown_count > 3:
        log_security_event(
            "MULTIPLE_UNKNOWN_FACES",
            f"Detected {unknown_count} unknown faces simultaneously",
            "WARNING"
        )
    
    # Very low confidence across multiple faces
    if face_confidences and len(face_confidences) > 1:
        avg_confidence = np.mean([c for c in face_confidences if c > 0])
        if avg_confidence < 0.3:
            log_security_event(
                "LOW_CONFIDENCE_PATTERN",
                f"Average confidence {avg_confidence:.2f} across {len(face_confidences)} faces",
                "WARNING"
            )
    
    # Faces appearing and disappearing rapidly
    if hasattr(detect_suspicious_activity, 'last_face_count'):
        face_count_change = abs(len(face_locations) - detect_suspicious_activity.last_face_count)
        if face_count_change > 2:
            log_security_event(
                "RAPID_FACE_CHANGES",
                f"Face count changed by {face_count_change} in one frame",
                "INFO"
            )
    
    detect_suspicious_activity.last_face_count = len(face_locations)

def comprehensive_system_health_check():
    """Perform comprehensive system health and security checks."""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'security_status': 'OK',
        'performance_status': 'OK',
        'alerts': []
    }
    
    # Check critical components
    health_status['components']['gpu_available'] = GPU_AVAILABLE
    health_status['components']['dlib_available'] = DLIB_AVAILABLE
    # HRNet/face_alignment removed for performance; using dlib 68 instead
    # Firebase removed; report local storage availability instead
    health_status['components']['local_storage_available'] = os.access('.', os.W_OK)
    
    # Check performance metrics
    if performance_metrics['frame_times']:
        avg_frame_time = np.mean(performance_metrics['frame_times'])
        if avg_frame_time > 0.1:  # More than 100ms per frame
            health_status['performance_status'] = 'DEGRADED'
            health_status['alerts'].append(f"High frame processing time: {avg_frame_time*1000:.1f}ms")
    
    # Check recognition cache health
    total_cache_requests = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
    if total_cache_requests > 0:
        cache_hit_rate = performance_metrics['cache_hits'] / total_cache_requests
        if cache_hit_rate < 0.5:  # Less than 50% hit rate
            health_status['alerts'].append(f"Low cache hit rate: {cache_hit_rate*100:.1f}%")
    
    # Check recent security events
    recent_alerts = [alert for alert in security_monitor['system_alerts'] 
                     if alert['severity'] in ['WARNING', 'ERROR', 'CRITICAL']]
    if len(recent_alerts) > 10:  # Many recent alerts
        health_status['security_status'] = 'ELEVATED'
        health_status['alerts'].append(f"{len(recent_alerts)} recent security alerts")
    
    return health_status

# PIN features removed

# Performance monitoring globals
performance_metrics = {
    'frame_times': deque(maxlen=50),
    'detection_times': deque(maxlen=50),
    'recognition_times': deque(maxlen=50),
    'encoding_times': deque(maxlen=50),
    'total_faces_detected': 0,
    'total_faces_recognized': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'avg_fps': 0,
    'peak_fps': 0
}

# Global variables for blink detection
blink_detection = {
    'active': False, 'start_time': 0, 'blink_count': 0,
    'last_blink_time': 0, 'ear_values': deque(maxlen=10),
    'person_name': '', 'class_name': '', 'status_message': '',
    'previous_ear': 1.0,  # Initialize with open eye value
}

# HRNet facial landmark predictor removed; using dlib 68-point predictor only.

class FaceQualityAssessor:
    """Comprehensive face quality assessment for security and accuracy."""
    
    def __init__(self):
        self.blur_threshold = CONFIG.get("blur_threshold", 100)
        self.brightness_range = CONFIG.get("brightness_threshold", (50, 200))
        self.symmetry_threshold = CONFIG.get("symmetry_threshold", 0.8)
        self.pose_threshold = CONFIG.get("pose_angle_threshold", 30)
        
    def assess_face_quality(self, face_image, landmarks=None):
        """
        Comprehensive face quality assessment.
        Returns:
            dict: Quality metrics and overall score
        """
        if face_image is None or face_image.size == 0:
            return {"overall_score": 0.0, "reason": "Invalid image"}
        
        quality_metrics = {
            "sharpness": self._assess_sharpness(face_image),
            "brightness": self._assess_brightness(face_image),
            "contrast": self._assess_contrast(face_image),
            "symmetry": self._assess_symmetry(face_image, landmarks),
            "pose_angle": self._assess_pose_angle(landmarks),
            "resolution": self._assess_resolution(face_image),
            "noise_level": self._assess_noise(face_image),
            "overall_score": 0.0
        }
        
        # Calculate weighted overall score
        weights = {
            "sharpness": 0.25, "brightness": 0.15, "contrast": 0.15,
            "symmetry": 0.15, "pose_angle": 0.15, "resolution": 0.10,
            "noise_level": 0.05
        }
        
        overall_score = sum(quality_metrics[metric] * weights[metric] 
                            for metric in weights.keys())
        quality_metrics["overall_score"] = overall_score
        
        return quality_metrics
    
    def _assess_sharpness(self, image):
        """Assess image sharpness using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return min(1.0, laplacian_var / self.blur_threshold)
        except Exception:
            return 0.0
    
    def _assess_brightness(self, image):
        """Assess image brightness."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            
            min_bright, max_bright = self.brightness_range
            if min_bright <= mean_brightness <= max_bright:
                return 1.0
            elif mean_brightness < min_bright:
                return max(0.0, mean_brightness / min_bright)
            else:
                return max(0.0, (255 - mean_brightness) / (255 - max_bright))
        except Exception:
            return 0.0
    
    def _assess_contrast(self, image):
        """Assess image contrast using standard deviation."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = np.std(gray)
            return min(1.0, contrast / 50.0)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _assess_symmetry(self, image, landmarks=None):
        """Assess face symmetry."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default score when landmarks unavailable
        
        try:
            left_points = landmarks[0:17]  # Face outline left
            right_points = landmarks[16:0:-1]  # Face outline right (reversed)
            nose_tip = landmarks[30]
            center_x = nose_tip[0]
            
            left_distances = [abs(point[0] - center_x) for point in left_points]
            right_distances = [abs(point[0] - center_x) for point in right_points]
            
            symmetry_diffs = [abs(l - r) for l, r in zip(left_distances, right_distances)]
            avg_diff = np.mean(symmetry_diffs)
            
            symmetry_score = max(0.0, 1.0 - (avg_diff / 20.0))
            return symmetry_score
        except Exception:
            return 0.5
    
    def _assess_pose_angle(self, landmarks=None):
        """Assess head pose angle."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default score
        
        try:
            nose_tip = landmarks[30]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            angle_degrees = abs(np.degrees(angle))
            
            pose_score = max(0.0, 1.0 - (angle_degrees / self.pose_threshold))
            return pose_score
        except Exception:
            return 0.5
    
    def _assess_resolution(self, image):
        """Assess if image resolution is adequate."""
        try:
            height, width = image.shape[:2]
            min_size = min(height, width)
            
            if min_size >= CONFIG["min_face_size"][0]:
                return 1.0
            else:
                return max(0.0, min_size / CONFIG["min_face_size"][0])
        except Exception:
            return 0.0
    
    def _assess_noise(self, image):
        """Assess image noise level."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            noise_estimate = np.var(gradient_magnitude)
            noise_score = max(0.0, 1.0 - (noise_estimate / 10000.0))
            return noise_score
        except Exception:
            return 0.5

class LivenessDetector:
    """Advanced liveness detection to prevent spoofing attacks."""
    
    def __init__(self):
        self.texture_patterns = []
        self.motion_history = deque(maxlen=30)
        self.challenge_responses = {
            'blink': False, 'smile': False, 'nod': False,
            'turn_left': False, 'turn_right': False
        }
    
    def detect_liveness(self, face_image, landmarks=None, frame_history=None):
        """
        Comprehensive liveness detection.
        Returns:
            dict: Liveness assessment results
        """
        liveness_score = {
            "texture_analysis": self._analyze_texture(face_image),
            "motion_detection": self._detect_motion(frame_history),
            "depth_estimation": self._estimate_depth(face_image, landmarks),
            "micro_expressions": self._detect_micro_expressions(landmarks),
            "pulse_detection": self._detect_pulse(face_image),
            "overall_liveness": 0.0
        }
        
        weights = {
            "texture_analysis": 0.3, "motion_detection": 0.25, "depth_estimation": 0.2,
            "micro_expressions": 0.15, "pulse_detection": 0.1
        }
        
        overall_score = sum(liveness_score[metric] * weights[metric] 
                            for metric in weights.keys())
        liveness_score["overall_liveness"] = overall_score
        
        return liveness_score
    
    def _analyze_texture(self, face_image):
        """Analyze face texture for signs of spoofing."""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            texture_variance = np.var(sobel_x) + np.var(sobel_y)
            
            texture_score = min(1.0, texture_variance / 50000.0)
            return texture_score
        except Exception:
            return 0.5
    
    def _detect_motion(self, frame_history):
        """Detect natural head motion indicative of a live person."""
        # This is a placeholder. Real optical flow is complex.
        return 0.5 # Placeholder
    
    def _estimate_depth(self, face_image, landmarks=None):
        """Estimate face depth to detect flat surfaces."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5
        
        try:
            nose_tip = landmarks[30]
            nose_bridge = landmarks[27]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            eye_line_y = (left_eye[1] + right_eye[1]) / 2
            nose_depth_ratio = abs(nose_bridge[1] - eye_line_y) / abs(nose_tip[1] - eye_line_y)
            
            depth_score = min(1.0, nose_depth_ratio / 0.5)
            return depth_score
        except Exception:
            return 0.5
    
    def _detect_micro_expressions(self, landmarks):
        """Detect subtle micro-expressions that indicate a live person."""
        if landmarks is None:
            return 0.5
        
        try:
            if hasattr(self, 'previous_landmarks') and self.previous_landmarks is not None:
                movement_scores = []
                
                for eye_idx in range(36, 48):  # Eye landmark indices
                    if eye_idx < len(landmarks) and eye_idx < len(self.previous_landmarks):
                        movement = np.linalg.norm(
                            np.array(landmarks[eye_idx]) - np.array(self.previous_landmarks[eye_idx])
                        )
                        movement_scores.append(movement)
                
                if movement_scores:
                    avg_movement = np.mean(movement_scores)
                    micro_expr_score = min(1.0, avg_movement / 2.0) if avg_movement < 10 else 0.5
                    self.previous_landmarks = landmarks
                    return micro_expr_score
            
            self.previous_landmarks = landmarks
            return 0.5
        except Exception:
            return 0.5
    
    def _detect_pulse(self, face_image):
        """Detect pulse from facial color variations (simplified)."""
        try:
            height, width = face_image.shape[:2]
            forehead_region = face_image[int(height*0.1):int(height*0.3), 
                                         int(width*0.3):int(width*0.7)]
            
            if forehead_region.size > 0:
                avg_intensity = np.mean(forehead_region)
                
                if hasattr(self, 'pulse_history'):
                    self.pulse_history.append(avg_intensity)
                    if len(self.pulse_history) > 100:
                        self.pulse_history.pop(0)
                    
                    if len(self.pulse_history) > 30:
                        pulse_variance = np.var(self.pulse_history)
                        pulse_score = min(1.0, pulse_variance / 10.0)
                        return pulse_score
                else:
                    self.pulse_history = [avg_intensity]
            
            return 0.5
        except Exception:
            return 0.5

# Initialize quality assessment components
face_quality_assessor = FaceQualityAssessor()
liveness_detector = LivenessDetector()

# Optional: dlib 68-landmark shape predictor for landmark embeddings
dlib_shape_predictor = None
cnn_face_rec_model = None

def initialize_dlib_shape_predictor(model_path=None):
    """Initialize dlib 68-landmark shape predictor if available.
    Search order: provided path -> $DLIB_LANDMARK_MODEL -> ./shape_predictor_68_face_landmarks.dat
    """
    global dlib_shape_predictor
    try:
        if not 'DLIB_AVAILABLE' in globals() or not DLIB_AVAILABLE:
            return False
        if dlib_shape_predictor is not None:
            return True
        if model_path is None:
            model_path = os.environ.get('DLIB_LANDMARK_MODEL', 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(model_path):
            print(f"⚠️ dlib 68-landmark model not found at '{model_path}'. Set DLIB_LANDMARK_MODEL.")
            return False
        import dlib  # lazy import
        dlib_shape_predictor = dlib.shape_predictor(model_path)
        print("✅ Loaded dlib 68-landmark predictor")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load dlib shape predictor: {e}")
        return False

def initialize_dlib_cnn_model(model_path=None):
    """Initialize dlib face recognition ResNet model if available."""
    global cnn_face_rec_model
    try:
        if not DLIB_AVAILABLE:
            return False
        if cnn_face_rec_model is not None:
            return True
        if model_path is None:
            model_path = (
                os.environ.get('DLIB_FACE_REC_MODEL') or
                os.environ.get('DLIB_FACE_RECOGNITION_MODEL') or
                'dlib_face_recognition_resnet_model_v1.dat'
            )
        if not os.path.exists(model_path):
            print(f"⚠️ dlib face recognition model not found at '{model_path}'. Set DLIB_FACE_REC_MODEL.")
            return False
        import dlib
        cnn_face_rec_model = dlib.face_recognition_model_v1(model_path)
        print("✅ Loaded dlib face recognition model")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load dlib face recognition model: {e}")
        return False

def _shape_to_points(shape):
    if shape is None:
        return None
    try:
        return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    except Exception:
        return None

def _compute_landmark_embedding(rgb_frame, face_location, shape=None, gray_image=None):
    """Return 136-d normalized embedding from dlib 68 landmarks; None on failure."""
    try:
        if dlib_shape_predictor is None and shape is None:
            return None
        import dlib
        top, right, bottom, left = face_location
        if shape is None:
            rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
            if gray_image is None:
                gray_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            shape = dlib_shape_predictor(gray_image, rect)
        if shape is None:
            return None
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
        # Center and scale by inter-ocular distance (fallback to bbox diag)
        mean = pts.mean(axis=0)
        pts_c = pts - mean
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)
        scale = float(np.linalg.norm(right_eye - left_eye))
        if scale <= 1e-6:
            w = max(1.0, right - left)
            h = max(1.0, bottom - top)
            scale = float(np.hypot(w, h))
        emb = (pts_c / scale).flatten().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm
        return emb
    except Exception:
        return None

def _compute_cnn_embedding(rgb_frame, face_location, shape=None, gray_image=None):
    """Return 128-d embedding from dlib ResNet face descriptor; None on failure."""
    try:
        if cnn_face_rec_model is None:
            return None
        import dlib
        top, right, bottom, left = face_location
        if shape is None:
            if dlib_shape_predictor is None:
                return None
            rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
            if gray_image is None:
                gray_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            shape = dlib_shape_predictor(gray_image, rect)
        if shape is None:
            return None
        face_chip = dlib.get_face_chip(rgb_frame, shape, size=150)
        descriptor = cnn_face_rec_model.compute_face_descriptor(face_chip)
        vec = np.array(descriptor, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec
    except Exception:
        return None

def _normalize_embedding(vec):
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm > 1e-6:
        arr = arr / norm
    return arr

def fuse_face_embeddings(landmark_vec, cnn_vec):
    """Fuse embeddings according to CONFIG['embedding_mode']."""
    mode = str(CONFIG.get("embedding_mode", "landmarks")).lower()
    weights = CONFIG.get("embedding_component_weights", {"cnn": 0.7, "landmarks": 0.3})
    lm = _normalize_embedding(landmark_vec)
    cnn = _normalize_embedding(cnn_vec)

    if mode == "cnn":
        return cnn if cnn is not None else lm
    if mode == "landmarks":
        return lm if lm is not None else cnn

    # Hybrid default: concatenate available components with weighting
    fused_parts = []
    if cnn is not None:
        fused_parts.append(cnn * float(weights.get("cnn", 0.7)))
    if lm is not None:
        fused_parts.append(lm * float(weights.get("landmarks", 0.3)))
    if not fused_parts:
        return None
    fused = np.concatenate(fused_parts).astype(np.float32)
    norm = float(np.linalg.norm(fused))
    if norm > 1e-6:
        fused /= norm
    return fused

def validate_face_region(image, face_location):
    """Validate that a detected face region meets quality standards."""
    try:
        top, right, bottom, left = face_location
        
        if top < 0 or left < 0 or bottom >= image.shape[0] or right >= image.shape[1]:
            return False, 0.0, "Face region out of bounds"
        
        if right <= left or bottom <= top:
            return False, 0.0, "Invalid face dimensions"
        
        face_image = image[top:bottom, left:right]
        
        if face_image.size == 0:
            return False, 0.0, "Empty face region"
        
        face_width = right - left
        face_height = bottom - top
        
        min_size = CONFIG["min_face_size"]
        max_size = CONFIG["max_face_size"]
        
        if face_width < min_size[0] or face_height < min_size[1]:
            return False, 0.2, f"Face too small: {face_width}x{face_height}"
        
        if face_width > max_size[0] or face_height > max_size[1]:
            return False, 0.3, f"Face too large: {face_width}x{face_height}"
        
        quality_metrics = face_quality_assessor.assess_face_quality(face_image)
        overall_quality = quality_metrics["overall_score"]
        
        threshold = CONFIG.get("face_quality_threshold", 0.3)
        
        if overall_quality < threshold:
            reasons = []
            if quality_metrics["sharpness"] < 0.3: reasons.append("blurry")
            if quality_metrics["brightness"] < 0.3: reasons.append("poor lighting")
            if quality_metrics["contrast"] < 0.3: reasons.append("low contrast")
            return False, overall_quality, f"Low quality: {', '.join(reasons)}"
        
        return True, overall_quality, "Valid face"
        
    except Exception as e:
        return False, 0.0, f"Validation error: {str(e)}"

def enhanced_face_preprocessing(face_image):
    """Apply advanced preprocessing to improve face recognition accuracy."""
    try:
        if face_image is None or face_image.size == 0:
            return face_image
        
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        if len(face_image.shape) == 3:
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            face_image = cv2.equalizeHist(face_image)
        
        if len(face_image.shape) == 3:
            face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
        else:
            face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        face_image = cv2.filter2D(face_image, -1, kernel)
        
        face_image = np.clip(face_image, 0, 255).astype(np.uint8)
        
        return face_image
        
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        return face_image

# PIN loading removed


def sync_student_metadata_from_api(student_ids, grade="1", homeroom="1A", token=None, write_to_storage=False):
    """
    (This function is unchanged)
    For a list of student IDs, call the sandbox API to retrieve student metadata
    and store it in Firestore (collection: 'students'). 
    """
    results = {}
    try:
        try:
            import api_integrate
        except Exception:
            api_integrate = None

        if token is None and api_integrate is not None and hasattr(api_integrate, 'get_auth_token'):
            try:
                token = api_integrate.get_auth_token()
            except Exception as e:
                print(f"⚠️ Failed to get token from api_integrate: {e}")

        if api_integrate is None:
            print("❌ api_integrate module not available. Aborting sync.")
            return results

        print(f"🔁 Querying sandbox API for {len(student_ids)} students (grade={grade} homeroom={homeroom})")
        students = api_integrate.get_student_photos(grade=grade, homeroom=homeroom, student_ids=student_ids, token=token)

        if not students:
            print("ℹ️ API returned no students for the provided ids")
            return results

    # Firebase disabled: we'll optionally write local metadata

        for s in students:
            idStudent = str(s.get('idStudent') or s.get('IdStudent') or s.get('id') or '')
            idBinusian = s.get('idBinusian') or s.get('idBinusian') or ''
            name = s.get('studentName') or s.get('name') or s.get('fullName') or s.get('studentFullName') or s.get('nama') or ''
            filePath = s.get('filePath') or s.get('fileName') or ''

            if idStudent:
                doc_data = {
                    'idStudent': idStudent, 'idBinusian': idBinusian, 'name': name,
                    'filePath': filePath, 'grade': grade, 'homeroom': homeroom,
                    'synced_at': datetime.utcnow().isoformat()
                }
                # Optionally write a local metadata JSON alongside dataset
                if write_to_storage and filePath:
                    try:
                        class_name = homeroom or "unknown"
                        student_folder = idStudent or name or "unknown"
                        local_meta_dir = os.path.join("face_dataset", class_name, student_folder)
                        os.makedirs(local_meta_dir, exist_ok=True)
                        metadata_path = os.path.join(local_meta_dir, "metadata.json")
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(doc_data, f, ensure_ascii=False, indent=2)
                        print(f"✅ Wrote local metadata: {metadata_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to write local metadata for {idStudent}: {e}")

            results[idStudent] = s

        return results

    except Exception as e:
        print(f"❌ sync_student_metadata_from_api failed: {e}")
        return results


def lookup_student_via_api(student_id, grade=None, homeroom=None, token=None):
    """Lookup a single student by ID and return (name, class_name).

    Behavior:
    - If grade & homeroom are provided, use class-based API (batch endpoint) to fetch.
    - Otherwise, prefer the Part C2-style ID endpoint (env-configured) for direct lookup.
    """
    try:
        try:
            import api_integrate  # type: ignore
        except Exception:
            api_integrate = None

        if api_integrate is None:
            print("❌ API module not available. Cannot lookup student by ID.")
            return None, None

        if not student_id:
            print("❌ No student_id provided")
            return None, None

        # If grade & homeroom provided, use the class-based batch API
        if grade and homeroom:
            if token is None and hasattr(api_integrate, 'get_auth_token'):
                try:
                    token = api_integrate.get_auth_token()
                except Exception as e:
                    print(f"⚠️ Failed to get token: {e}")
            resp = api_integrate.get_student_photos(grade=grade, homeroom=homeroom, student_ids=[str(student_id)], token=token)
            if resp:
                for s in resp:
                    sid = str(s.get('idStudent') or s.get('IdStudent') or '')
                    if sid == str(student_id):
                        name = s.get('studentName') or s.get('name') or s.get('fullName') or s.get('studentFullName') or s.get('nama') or ''
                        return name, homeroom
            return None, None

        # Otherwise, prefer C2 direct lookup if available
        if hasattr(api_integrate, 'get_student_by_id_c2'):
            student = api_integrate.get_student_by_id_c2(student_id, token=token)
            if isinstance(student, dict):
                name = student.get('studentName') or student.get('name') or student.get('fullName') or student.get('studentFullName') or student.get('nama') or ''
                # Try to infer class/homeroom if present; else None
                class_name = student.get('homeroom') or student.get('class') or student.get('className') or None
                return (name or None), class_name
        
        return None, None
    except Exception as e:
        print(f"❌ lookup_student_via_api failed: {e}")
        return None, None


# PIN UI and handlers removed


def handle_mouse_click(event, x, y, flags, param):
    return  # No-op; PIN UI removed


def log_attendance(name, class_name=None, confidence=0.0, quality_score=0.0, security_score=0.0):
    """Record attendance locally (JSON), with duplicate prevention and basic checks."""
    attendance_key = name
    if class_name:
        attendance_key = f"{class_name}/{name}" 
    
    current_time = datetime.now()
    current_timestamp = current_time.timestamp()
    
    if attendance_key in attendance_timestamps:
        last_login_time = attendance_timestamps[attendance_key]
        time_diff = current_timestamp - last_login_time
        
        if time_diff < CONFIG.get("duplicate_detection_window", 300):
            minutes_remaining = int((CONFIG.get("duplicate_detection_window", 300) - time_diff) / 60)
            print(f"⚠️ {name} already logged in recently. Please wait {minutes_remaining} more minutes.")
            return False
    
    min_confidence = CONFIG.get("min_recognition_threshold", 0.65)
    min_quality = CONFIG.get("face_quality_threshold", 0.3)
    min_security = 0.4  # Minimum liveness score
    
    if confidence < min_confidence:
        print(f"⚠️ Attendance rejected for {name}: Low confidence ({confidence:.2f} < {min_confidence})")
        return False
    
    if quality_score > 0 and quality_score < min_quality:
        print(f"⚠️ Attendance rejected for {name}: Poor image quality ({quality_score:.2f} < {min_quality})")
        return False
    
    if security_score > 0 and security_score < min_security:
        print(f"⚠️ Attendance rejected for {name}: Failed liveness detection ({security_score:.2f} < {min_security})")
        return False
        
    if attendance_key in attendance:
        return False  # Already logged today
    
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    date_only = current_time.strftime("%Y-%m-%d")

    latest_time = dt_time(
        int(CONFIG["latest_login_time"].split(":")[0]), 
        int(CONFIG["latest_login_time"].split(":")[1])
    )
    
    is_late = current_time.time() > latest_time
    status = "Late" if is_late else "Present"

    # Ensure primitive Python types for JSON
    py_conf = float(round(float(confidence), 3))
    py_qual = float(round(float(quality_score), 3)) if quality_score > 0 else None
    py_sec = float(round(float(security_score), 3)) if security_score > 0 else None

    attendance_data = {
        name: {
            'timestamp': timestamp, 'status': status, 'late': is_late,
            'confidence': py_conf,
            'quality_score': py_qual,
            'security_score': py_sec,
            'ip_address': 'localhost',
            'system_info': {
                'gpu_used': GPU_AVAILABLE and CONFIG["use_gpu_acceleration"],
                'detection_method': 'ensemble' if CONFIG.get("use_ensemble_models", True) else 'standard'
            }
        }
    }
    
    if class_name:
        attendance_data[name]['class'] = class_name

    try:
        # Ensure local directory exists
        local_dir = os.path.join("data", "attendance")
        os.makedirs(local_dir, exist_ok=True)
        daily_path = os.path.join(local_dir, f"{date_only}.json")

        # Load current day's log
        day_data = {}
        if os.path.isfile(daily_path):
            try:
                with open(daily_path, "r", encoding="utf-8") as f:
                    day_data = json.load(f) or {}
            except Exception:
                day_data = {}

        # Merge attendance entry
        day_data.update(attendance_data)

        # Write back to file
        with open(daily_path, "w", encoding="utf-8") as f:
            json.dump(day_data, f, ensure_ascii=False, indent=2, default=_json_fallback)

        if CONFIG.get("upload_attendance_to_firebase", False):
            upload_attendance_to_firebase(daily_path, date_only)

        # Upload to Binus School API (non-blocking)
        if CONFIG.get("upload_attendance_to_api", False) and API_INTEGRATE_ENABLED:
            try:
                api_payload = {
                    "studentName": name,
                    "class": class_name or "",
                    "timestamp": timestamp,
                    "status": status,
                    "late": is_late,
                    "confidence": py_conf,
                }
                threading.Thread(
                    target=_upload_attendance_to_api,
                    args=(api_payload,),
                    daemon=True,
                ).start()
            except Exception as api_err:
                print(f"⚠️ Failed to queue API attendance upload: {api_err}")

        # Update local tracking
        attendance[attendance_key] = True
        attendance_timestamps[attendance_key] = current_timestamp

        # Console message + thank you overlay
        log_message = f"✅ Attendance logged: {name}"
        if class_name: log_message += f" ({class_name})"
        log_message += f" - {status} (conf: {confidence:.2f}"
        if quality_score > 0: log_message += f", qual: {quality_score:.2f}"
        if security_score > 0: log_message += f", sec: {security_score:.2f}"
        log_message += ")"
        print(log_message)

        global thank_you_message
        thank_you_message = {
            'active': True,
            'name': name if not class_name else f"{name} ({class_name})",
            'time': time.time(),
            'duration': 3.0,
            'quote': random.choice(MOTIVATIONAL_QUOTES),
            'status': status
        }

        return True
    except Exception as e:
        print(f"❌ Error logging attendance locally for {name}: {e}")
        return False

#
# --- THIS IS THE MODIFIED SECTION ---
#
# We have replaced the old, complex `load_face_encodings` and
# `load_face_encodings_source` with this single, simple function
# that loads metrics from your local `encodings.pickle` file.

def load_face_encodings(cache_file="encodings.pickle"):
    """
    Load known face encodings from the local cache file.
    
    Returns:
        tuple: (landmark_encodings, cnn_encodings, names, classes)
    """
    print(f"🔄 Loading face encodings from local cache file: {cache_file}...")
    
    if not os.path.exists(cache_file):
        print(f"❌ FATAL ERROR: Cache file '{cache_file}' not found.")
        print(f"  Please run the 'enroll_local.py' script first to build it.")
        print("  The application will run but will not recognize anyone.")
        return [], [], [], []

    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            known_face_landmarks = data.get("landmark_encodings") or data.get("encodings", [])
            known_face_cnn = data.get("cnn_encodings", [])
            known_face_names = data.get("names", [])
            known_face_classes = data.get("classes", [])
            
        if not known_face_landmarks and not known_face_cnn:
            print(f"⚠️ Warning: Cache file '{cache_file}' is empty.")
            print("  Please run 'enroll_local.py' to add students.")
            return [], [], [], []

        print(f"✅ Loaded {len(known_face_names)} students from cache")
        return known_face_landmarks, known_face_cnn, known_face_names, known_face_classes
        
    except Exception as e:
        print(f"❌ Error loading cache file '{cache_file}': {e}")
        print(f"  The file might be corrupt. Try deleting it and re-running 'enroll_local.py'.")
        return [], [], [], []

#
# --- END OF MODIFIED SECTION ---
#

def display_thank_you_message(frame):
    """Display thank you message overlay on frame."""
    if not thank_you_message['active']:
        return frame
        
    if time.time() - thank_you_message['time'] < thank_you_message['duration']:
        bg_color = (0, 200, 0) if thank_you_message['status'] == "Present" else (0, 165, 255)
        
        cv2.rectangle(frame, (50, frame.shape[0]//2 - 70), 
                      (frame.shape[1]-50, frame.shape[0]//2 + 70), 
                      bg_color, -1)
        
        cv2.putText(frame, f"Thank you, {thank_you_message['name']}!", 
                    (100, frame.shape[0]//2 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"You are {thank_you_message['status']}", 
                    (100, frame.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        quote = thank_you_message['quote']
        if len(quote) > 50:
            split_idx = quote.find(' ', 40)
            if split_idx == -1: split_idx = 50
            line1 = quote[:split_idx]
            line2 = quote[split_idx:]
            cv2.putText(frame, f'"{line1}', 
                        (100, frame.shape[0]//2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f'{line2}"', 
                        (100, frame.shape[0]//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f'"{quote}"', 
                        (100, frame.shape[0]//2 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        thank_you_message['active'] = False
        
    return frame

 # HRNet landmark drawing removed; dlib-based blink UI handles visualization.

# Class for face tracking to improve performance
class FaceTracker:
    # This class is unchanged
    def __init__(self, max_disappeared=30, min_quality=7):
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.min_quality = min_quality
        
    def register(self, face_rect, face_encoding, name="Unknown", class_name="", confidence=0.0):
        track_id = self.next_id
        self.tracks[track_id] = {
            "rect": face_rect, "encoding": face_encoding, "name": name,
            "class_name": class_name, "confidence": confidence, "quality": 10
        }
        self.disappeared[track_id] = 0
        self.next_id += 1
        return track_id
        
    def update(self, face_locations, face_encodings, face_names=None, face_classes=None, face_confidences=None):
        if face_names is None: face_names = ["Unknown"] * len(face_locations)
        if face_classes is None: face_classes = [""] * len(face_locations)
        if face_confidences is None: face_confidences = [0.0] * len(face_locations)
            
        if len(face_locations) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
            return self.tracks
        
        if len(self.tracks) == 0:
            for i, (rect, encoding, name, class_name, confidence) in enumerate(
                zip(face_locations, face_encodings, face_names, face_classes, face_confidences)):
                self.register(rect, encoding, name, class_name, confidence)
            return self.tracks
            
        track_ids = list(self.tracks.keys())
        track_encodings = [self.tracks[tid]["encoding"] for tid in track_ids]
        
        matched_tracks = {}
        unmatched_detections = list(range(len(face_locations)))
        
        for i, face_encoding in enumerate(face_encodings):
            if len(track_encodings) == 0:
                continue
                
            # Compute Euclidean distances between track encodings and the current face encoding
            try:
                track_encodings_arr = np.array(track_encodings)
                face_encoding_arr = np.array(face_encoding)
                face_distances = np.linalg.norm(track_encodings_arr - face_encoding_arr, axis=1)
            except Exception:
                face_distances = np.array([1.0] * len(track_encodings))
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            if min_distance < 0.6:
                track_id = track_ids[best_match_index]
                matched_tracks[track_id] = i
                if i in unmatched_detections:
                    unmatched_detections.remove(i)
                    
                self.tracks[track_id]["rect"] = face_locations[i]
                
                if face_names[i] != "Unknown" and face_confidences[i] > self.tracks[track_id]["confidence"]:
                    self.tracks[track_id]["name"] = face_names[i]
                    self.tracks[track_id]["class_name"] = face_classes[i]
                    self.tracks[track_id]["confidence"] = face_confidences[i]
                    
                self.tracks[track_id]["quality"] = min(10, self.tracks[track_id]["quality"] + 1)
                self.disappeared[track_id] = 0
        
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.disappeared[track_id] += 1
                self.tracks[track_id]["quality"] = max(0, self.tracks[track_id]["quality"] - 1)
                
                if self.disappeared[track_id] > self.max_disappeared or self.tracks[track_id]["quality"] < self.min_quality:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
        
        for i in unmatched_detections:
            self.register(face_locations[i], face_encodings[i], 
                          face_names[i], face_classes[i], face_confidences[i])
        
        return self.tracks

# Advanced caching system
class LRUCache:
    # This class is unchanged
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                oldest = self.order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)

# High-performance recognition cache
recognition_cache = LRUCache(CONFIG["recognition_cache_size"])

# Preloaded face data for faster access
preloaded_face_data = {
    'encodings': None,
    'names': None,
    'classes': None,
    'loaded': False,
    'gpu_encodings': None,
    'landmark_encodings': None,
    'cnn_encodings': None
}

# Face detection optimization
class OptimizedFaceDetector:
    # This class is unchanged
    def __init__(self):
        self.detectors = {}
        self.initialize_detectors()
        self.detection_history = deque(maxlen=10)
        
    def initialize_detectors(self):
        # Load dlib HOG detector only
        if DLIB_AVAILABLE:
            try:
                hog_detector = dlib.get_frontal_face_detector()
                self.detectors['dlib_hog'] = hog_detector
                print("ℹ️ dlib HOG detector loaded")
            except Exception as e:
                print(f"⚠️ dlib HOG detector failed: {e}")
    
    def detect_faces_optimized(self, frame):
        start_time = time.time()
        # Use dlib HOG only
        faces = self._detect_with_dlib_hog(frame) if 'dlib_hog' in self.detectors else []
        
        self.detection_history.append(len(faces))
        
        detection_time = time.time() - start_time
        performance_metrics['detection_times'].append(detection_time)
        performance_metrics['total_faces_detected'] += len(faces)
        
        return faces
    
    # Haar detection removed per requirement
    
    def _detect_with_dlib_hog(self, frame):
        if 'dlib_hog' not in self.detectors:
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            drects = self.detectors['dlib_hog'](gray, 0)
            faces = []
            h, w = gray.shape[:2]
            for r in drects:
                x = max(0, r.left())
                y = max(0, r.top())
                x2 = min(w - 1, r.right())
                y2 = min(h - 1, r.bottom())
                faces.append([x, y, x2 - x, y2 - y])
            return faces
        except Exception as e:
            print(f"⚠️ dlib HOG detection error: {e}")
            return self._detect_with_haar(frame)

# Initialize optimized detector
face_detector = OptimizedFaceDetector()

def preload_face_encodings():
    """
    Preload all face encodings for maximum performance.
    This function now calls our new, simple load_face_encodings()
    """
    global preloaded_face_data
    
    if preloaded_face_data['loaded']:
        return
    
    print("🚀 Preloading face encodings for maximum performance...")
    
    # This now calls your new, fast, local-file-only function
    landmark_encodings, cnn_encodings, names, classes = load_face_encodings()

    landmark_encodings = landmark_encodings or []
    cnn_encodings = cnn_encodings or []
    classes = classes or []

    fused_encodings = []
    filtered_names = []
    filtered_classes = []

    for idx, name in enumerate(names):
        lm = landmark_encodings[idx] if idx < len(landmark_encodings) else None
        cnn = cnn_encodings[idx] if idx < len(cnn_encodings) else None
        fused = fuse_face_embeddings(lm, cnn)
        if fused is not None:
            fused_encodings.append(fused)
            filtered_names.append(name)
            filtered_classes.append(classes[idx] if idx < len(classes) else "")

    if fused_encodings:
        preloaded_face_data['encodings'] = np.array(fused_encodings, dtype=np.float32)
        preloaded_face_data['names'] = filtered_names
        preloaded_face_data['classes'] = filtered_classes
        preloaded_face_data['landmark_encodings'] = landmark_encodings
        preloaded_face_data['cnn_encodings'] = cnn_encodings
        
        # Preload to GPU if available
        if GPU_AVAILABLE and CONFIG["use_gpu_acceleration"]:
            try:
                preloaded_face_data['gpu_encodings'] = cp.asarray(preloaded_face_data['encodings'])
                print("✅ Face encodings loaded to GPU")
            except Exception as e:
                print(f"⚠️ GPU loading failed: {e}")
        
        preloaded_face_data['loaded'] = True
        print(f"✅ Preloaded {len(fused_encodings)} face encodings")
    else:
        print("⚠️ No face encodings to preload. Did you run 'enroll_local.py'?")

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) based on facial landmarks."""
    if len(eye_landmarks) != 6:
        return 1.0
    
    try:
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        ear = (A + B) / (2.0 * C) if C > 0 else 1.0
        return ear
    except Exception as e:
        print(f"⚠️ Error calculating EAR: {str(e)}")
        return 1.0

def get_eye_landmarks(landmarks):
    """Extract eye landmarks from the full set of facial landmarks."""
    if landmarks is None or len(landmarks) < 68:
        return None, None
    
    try:
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        return left_eye, right_eye
    except Exception as e:
        print(f"⚠️ Error extracting eye landmarks: {str(e)}")
        return None, None

def detect_blink(ear_value):
    """Detect a blink based on EAR values."""
    global blink_detection
    
    blink_detection['ear_values'].append(ear_value)
    
    blink_start = False
    if (blink_detection['previous_ear'] > CONFIG['ear_threshold'] and 
        ear_value <= CONFIG['ear_threshold']):
        blink_start = True
    
    if (blink_detection['previous_ear'] <= CONFIG['ear_threshold'] and 
        ear_value > CONFIG['ear_threshold']):
        if time.time() - blink_detection['last_blink_time'] > 0.2:
            blink_detection['blink_count'] += 1
            blink_detection['last_blink_time'] = time.time()
            blink_detection['status_message'] = f"Blink detected! ({blink_detection['blink_count']}/{CONFIG['min_blinks_required']})"
            print(f"👁️ Blink detected! Count: {blink_detection['blink_count']}")
            
            if blink_detection['blink_count'] >= CONFIG['min_blinks_required']:
                return True
    
    blink_detection['previous_ear'] = ear_value
    return False

def draw_eye_landmarks(frame, landmarks, left_eye, right_eye, ear_value):
    """Draw eye landmarks and EAR value on the frame for debugging."""
    if not CONFIG["show_eye_landmarks"]:
        return
    
    try:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
            
        for (x, y) in left_eye:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        for (x, y) in right_eye:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except Exception:
        pass

def start_blink_detection(name, class_name):
    """Start the blink detection process for anti-spoofing."""
    global blink_detection
    
    blink_detection = {
        'active': True, 'start_time': time.time(), 'blink_count': 0,
        'last_blink_time': 0, 'ear_values': deque(maxlen=10),
        'person_name': name, 'class_name': class_name,
        'status_message': "Please blink naturally...", 'previous_ear': 1.0
    }
    
    print(f"👁️ Anti-spoofing activated: Waiting for {CONFIG['min_blinks_required']} blinks from {name}")

def display_blink_detection_ui(frame):
    """Display blink detection UI overlay on frame."""
    if not blink_detection['active']:
        return frame
    
    elapsed_time = time.time() - blink_detection['start_time']
    time_remaining = max(0, CONFIG['blink_detection_time'] - elapsed_time)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 100, 200), 3)
    
    cv2.putText(frame, "ANTI-SPOOFING VERIFICATION", 
                (frame.shape[1]//2 - 200, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    
    cv2.putText(frame, f"Hello, {blink_detection['person_name']}!", 
                (frame.shape[1]//2 - 150, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.putText(frame, "Please blink naturally to verify you are a real person", 
                (frame.shape[1]//2 - 250, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    cv2.putText(frame, blink_detection['status_message'], 
                (frame.shape[1]//2 - 150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    progress_width = int((time_remaining / CONFIG['blink_detection_time']) * (frame.shape[1] - 200))
    cv2.rectangle(frame, (100, frame.shape[0] - 100), 
                  (frame.shape[1] - 100, frame.shape[0] - 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (100, frame.shape[0] - 100), 
                  (100 + progress_width, frame.shape[0] - 80), (0, 200, 0), -1)
    cv2.putText(frame, f"Time remaining: {int(time_remaining)}s", 
                (100, frame.shape[0] - 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if time_remaining <= 0:
        blink_detection['active'] = False
        print(f"⏱️ Blink detection timed out for {blink_detection['person_name']}")
    
    return frame

def enhanced_face_recognition_ensemble(face_encoding, face_image=None, landmarks=None, use_cache=True):
    """Enhanced face recognition with ensemble models, security checks, and quality assessment."""
    start_time = time.time()
    
    if use_cache:
        try:
            encoding_hash = hash(face_encoding.tobytes())
            cached_result = recognition_cache.get(encoding_hash)
            if cached_result is not None:
                performance_metrics['cache_hits'] += 1
                return cached_result
        except Exception:
            pass
    
    if not preloaded_face_data['loaded']:
        preload_face_encodings()
    
    name = "Unknown"
    class_name = ""
    base_confidence = 0.0
    quality_metrics = {}
    security_score = 0.0
    
    if face_image is not None and CONFIG.get("enable_quality_assessment", True):
        quality_metrics = face_quality_assessor.assess_face_quality(face_image, landmarks)
        
        if quality_metrics["overall_score"] < CONFIG.get("face_quality_threshold", 0.3):
            result = (name, class_name, 0.0, quality_metrics, 0.0)
            if use_cache:
                try: recognition_cache.put(encoding_hash, result)
                except Exception: pass
            return result
    
    if face_image is not None and CONFIG.get("liveness_detection", True):
        liveness_metrics = liveness_detector.detect_liveness(face_image, landmarks)
        security_score = liveness_metrics["overall_liveness"]
        
        if security_score < 0.4:
            result = (name, class_name, 0.0, quality_metrics, security_score)
            if use_cache:
                try: recognition_cache.put(encoding_hash, result)
                except Exception: pass
            return result
    
    if preloaded_face_data['encodings'] is not None and len(preloaded_face_data['encodings']) > 0:
        try:
            recognition_results = []
            # Resolve model weights (allow "default")
            weights_cfg = CONFIG.get("model_weights", "default")
            if isinstance(weights_cfg, str) and weights_cfg.lower() == "default":
                weights = {"euclidean": 0.5, "cosine": 0.3, "manhattan": 0.2}
            else:
                # Merge over defaults for safety
                defaults = {"euclidean": 0.5, "cosine": 0.3, "manhattan": 0.2}
                weights = {**defaults, **(weights_cfg or {})}
            
            # Method 1: Standard Euclidean distance
            euclidean_distances = np.linalg.norm(preloaded_face_data['encodings'] - face_encoding, axis=1)
            euclidean_best_idx = np.argmin(euclidean_distances)
            euclidean_confidence = max(0, 1 - euclidean_distances[euclidean_best_idx])
            recognition_results.append({
                'method': 'euclidean', 'index': euclidean_best_idx,
                'confidence': euclidean_confidence,
                'weight': weights.get("euclidean", 0.5)
            })
            
            # Method 2: Cosine similarity
            try:
                norm_encoding = face_encoding / np.linalg.norm(face_encoding)
                norm_stored = preloaded_face_data['encodings'] / np.linalg.norm(preloaded_face_data['encodings'], axis=1, keepdims=True)
                
                cosine_similarities = np.dot(norm_stored, norm_encoding)
                cosine_best_idx = np.argmax(cosine_similarities)
                cosine_confidence = max(0, cosine_similarities[cosine_best_idx])
                recognition_results.append({
                    'method': 'cosine', 'index': cosine_best_idx,
                    'confidence': cosine_confidence,
                    'weight': weights.get("cosine", 0.3)
                })
            except Exception as e:
                print(f"⚠️ Cosine similarity failed: {e}")
            
            # Method 3: Manhattan distance
            try:
                manhattan_distances = np.sum(np.abs(preloaded_face_data['encodings'] - face_encoding), axis=1)
                manhattan_best_idx = np.argmin(manhattan_distances)
                manhattan_confidence = max(0, 1 - (manhattan_distances[manhattan_best_idx] / 100))
                recognition_results.append({
                    'method': 'manhattan', 'index': manhattan_best_idx,
                    'confidence': manhattan_confidence,
                    'weight': weights.get("manhattan", 0.2)
                })
            except Exception as e:
                print(f"⚠️ Manhattan distance failed: {e}")
            
            if recognition_results:
                candidate_scores = {}
                
                for result in recognition_results:
                    idx = result['index']
                    confidence = result['confidence']
                    weight = result['weight']
                    
                    if idx not in candidate_scores:
                        candidate_scores[idx] = {'total_score': 0.0, 'vote_count': 0, 'confidence_sum': 0.0}
                    
                    candidate_scores[idx]['total_score'] += confidence * weight
                    candidate_scores[idx]['vote_count'] += 1
                    candidate_scores[idx]['confidence_sum'] += confidence
                
                best_candidate = None
                best_score = 0.0
                
                for idx, scores in candidate_scores.items():
                    if scores['vote_count'] >= CONFIG.get("recognition_voting_threshold", 2):
                        avg_confidence = scores['confidence_sum'] / scores['vote_count']
                        final_score = scores['total_score'] * (scores['vote_count'] / len(recognition_results))
                        
                        if final_score > best_score and avg_confidence >= CONFIG["min_recognition_threshold"]:
                            best_score = final_score
                            best_candidate = idx
                
                if best_candidate is not None:
                    base_confidence = best_score
                    security_passed = True
                    
                    if base_confidence < CONFIG["min_recognition_threshold"]:
                        security_passed = False
                    
                    if base_confidence >= CONFIG["confident_recognition_threshold"]:
                        security_passed = True
                    
                    if quality_metrics:
                        quality_factor = quality_metrics["overall_score"]
                        base_confidence *= (0.5 + 0.5 * quality_factor)
                    
                    if security_score > 0:
                        base_confidence *= (0.6 + 0.4 * security_score)
                    
                    if security_passed and base_confidence >= CONFIG["min_recognition_threshold"]:
                        name = preloaded_face_data['names'][best_candidate]
                        class_name = preloaded_face_data['classes'][best_candidate] if best_candidate < len(preloaded_face_data['classes']) else ""
                        
                        if CONFIG.get("log_recognition_details", False):
                            print(f"✅ Ensemble recognition: {name} (conf: {base_confidence:.3f}, qual: {quality_metrics.get('overall_score', 0):.3f}, sec: {security_score:.3f})")
                
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
    
    result = (name, class_name, base_confidence, quality_metrics, security_score)
    
    if use_cache:
        try:
            recognition_cache.put(encoding_hash, result)
            performance_metrics['cache_misses'] += 1
        except Exception:
            pass
    
    recognition_time = time.time() - start_time
    performance_metrics['recognition_times'].append(recognition_time)
    if name != "Unknown":
        performance_metrics['total_faces_recognized'] += 1
    
    return result

def ultra_fast_face_recognition(face_encoding, use_cache=True):
    """Backward compatibility wrapper for the enhanced recognition function."""
    name, class_name, confidence, _, _ = enhanced_face_recognition_ensemble(
        face_encoding, None, None, use_cache
    )
    return (name, class_name, confidence)

def parallel_face_encoding(rgb_frame, face_locations):
    """Process face encodings (landmark, CNN, or hybrid) for detected faces."""
    start_time = time.time()

    if not face_locations:
        performance_metrics['encoding_times'].append(time.time() - start_time)
        return [], [], []

    embeddings = []
    landmarks_output = []
    encoded_indices = []

    use_landmarks = CONFIG.get("use_landmark_embeddings", True)
    use_cnn = CONFIG.get("use_cnn_embeddings", False)

    if (use_landmarks or use_cnn) and dlib_shape_predictor is None:
        initialize_dlib_shape_predictor()
    if use_cnn and cnn_face_rec_model is None:
        initialize_dlib_cnn_model()

    gray_frame = None
    if dlib_shape_predictor is not None and (use_landmarks or use_cnn):
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    for idx, loc in enumerate(face_locations):
        shape = None
        shape_points = None

        if dlib_shape_predictor is not None:
            try:
                import dlib
                top, right, bottom, left = loc
                rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
                shape = dlib_shape_predictor(gray_frame, rect) if gray_frame is not None else None
                shape_points = _shape_to_points(shape)
            except Exception:
                shape = None
                shape_points = None

        landmark_vec = None
        if use_landmarks:
            landmark_vec = _compute_landmark_embedding(rgb_frame, loc, shape, gray_frame)

        cnn_vec = None
        if use_cnn:
            cnn_vec = _compute_cnn_embedding(rgb_frame, loc, shape, gray_frame)

        fused_vec = fuse_face_embeddings(landmark_vec, cnn_vec)
        if fused_vec is not None:
            embeddings.append(fused_vec)
            landmarks_output.append(shape_points)
            encoded_indices.append(idx)

    encoding_time = time.time() - start_time
    performance_metrics['encoding_times'].append(encoding_time)
    return embeddings, landmarks_output, encoded_indices

def adaptive_quality_control(fps):
    """Dynamically adjust processing parameters based on performance."""
    if not CONFIG["dynamic_quality_adjustment"]:
        return
    
    target_fps = CONFIG["target_fps"]
    
    if fps < target_fps * 0.6:  # Significantly below target
        CONFIG["frame_resize"] = max(0.25, CONFIG["frame_resize"] - 0.05)
        CONFIG["skip_frames"] = min(4, CONFIG["skip_frames"] + 1)
        CONFIG["batch_processing_size"] = max(4, CONFIG["batch_processing_size"] - 2)
        CONFIG["max_parallel_recognitions"] = max(2, CONFIG["max_parallel_recognitions"] - 1)
    elif fps > target_fps * 1.3:  # Significantly above target
        CONFIG["frame_resize"] = min(0.5, CONFIG["frame_resize"] + 0.02)
        CONFIG["skip_frames"] = max(1, CONFIG["skip_frames"] - 1)
        CONFIG["batch_processing_size"] = min(16, CONFIG["batch_processing_size"] + 1)
        CONFIG["max_parallel_recognitions"] = min(8, CONFIG["max_parallel_recognitions"] + 1)

def display_enhanced_performance_metrics(frame):
    """Display comprehensive performance and security metrics."""
    if not CONFIG["performance_monitoring"]:
        return frame
    
    avg_frame_time = np.mean(performance_metrics['frame_times']) if performance_metrics['frame_times'] else 0
    avg_detection_time = np.mean(performance_metrics['detection_times']) if performance_metrics['detection_times'] else 0
    avg_recognition_time = np.mean(performance_metrics['recognition_times']) if performance_metrics['recognition_times'] else 0
    avg_encoding_time = np.mean(performance_metrics['encoding_times']) if performance_metrics['encoding_times'] else 0
    
    total_cache_requests = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
    cache_hit_rate = (performance_metrics['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
    
    # This check is simplified as it's intensive
    # recent_alerts = len(security_monitor['system_alerts']) 
    health_status = 'OK' # Placeholder for speed
    
    y_offset = 50
    
    metrics = [
        f"Frame: {avg_frame_time*1000:.1f}ms",
        f"Detection: {avg_detection_time*1000:.1f}ms",
        f"Recognition: {avg_recognition_time*1000:.1f}ms",
        f"Cache: {cache_hit_rate:.0f}% hit rate",
        f"Quality: {'ON' if CONFIG.get('enable_quality_assessment', True) else 'OFF'}",
        f"Liveness: {'ON' if CONFIG.get('liveness_detection', True) else 'OFF'}"
    ]
    
    for i, metric in enumerate(metrics):
        color = (0, 255, 255) if i < 3 else (255, 255, 0) if i == 3 else (255, 255, 255)
        cv2.putText(frame, metric, (10, y_offset + i * 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    return frame

def display_face_quality_info(frame, face_location, quality_metrics, security_score, confidence):
    """Display quality and security information for a detected face."""
    if not CONFIG.get("show_quality_metrics", False):
        return frame
    
    top, right, bottom, left = face_location
    
    info_x = left
    info_y = top - 60
    
    if info_y < 0:
        info_y = bottom + 20
    
    quality_score = quality_metrics.get("overall_score", 0.0) if quality_metrics else 0.0
    
    if quality_score > 0.7 and security_score > 0.7 and confidence > 0.8:
        status_color = (0, 255, 0); status_text = "EXCELLENT"
    elif quality_score > 0.5 and security_score > 0.5 and confidence > 0.6:
        status_color = (0, 255, 255); status_text = "GOOD"
    elif quality_score > 0.3 and confidence > 0.4:
        status_color = (0, 165, 255); status_text = "FAIR"
    else:
        status_color = (0, 0, 255); status_text = "POOR"
    
    box_height = 50
    cv2.rectangle(frame, (info_x, info_y - box_height), 
                  (info_x + 200, info_y), (0, 0, 0), -1)
    cv2.rectangle(frame, (info_x, info_y - box_height), 
                  (info_x + 200, info_y), status_color, 1)
    
    cv2.putText(frame, f"Quality: {status_text}", 
                (info_x + 5, info_y - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    cv2.putText(frame, f"Q:{quality_score:.2f} S:{security_score:.2f} C:{confidence:.2f}", 
                (info_x + 5, info_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    if quality_metrics:
        detail_text = f"Sharp:{quality_metrics.get('sharpness', 0):.1f} Bright:{quality_metrics.get('brightness', 0):.1f}"
        cv2.putText(frame, detail_text, 
                    (info_x + 5, info_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    return frame

# Ultra-fast face tracker with vectorized operations
class UltraFastFaceTracker:
    # This class is unchanged
    def __init__(self, max_disappeared=15, min_quality=3):
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.min_quality = min_quality
        self.frame_count = 0
        
    def register(self, face_rect, face_encoding, name="Unknown", class_name="", confidence=0.0):
        track_id = self.next_id
        self.tracks[track_id] = {
            "rect": face_rect, "encoding": face_encoding, "name": name,
            "class_name": class_name, "confidence": confidence, "quality": 8,
            "last_update": self.frame_count
        }
        self.disappeared[track_id] = 0
        self.next_id += 1
        return track_id
    
    def update(self, face_locations, face_encodings, face_names=None, face_classes=None, face_confidences=None):
        self.frame_count += 1
        
        if face_names is None: face_names = ["Unknown"] * len(face_locations)
        if face_classes is None: face_classes = [""] * len(face_locations)
        if face_confidences is None: face_confidences = [0.0] * len(face_locations)
        
        if len(face_locations) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
            return self.tracks
        
        if len(self.tracks) == 0:
            for i, (rect, encoding, name, class_name, confidence) in enumerate(
                zip(face_locations, face_encodings, face_names, face_classes, face_confidences)):
                self.register(rect, encoding, name, class_name, confidence)
            return self.tracks
        
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(face_encodings) > 0:
            track_encodings = np.array([self.tracks[tid]["encoding"] for tid in track_ids])
            face_encodings_array = np.array(face_encodings)
            
            distances_matrix = np.linalg.norm(
                track_encodings[:, np.newaxis] - face_encodings_array[np.newaxis, :], 
                axis=2
            )
            
            matched_tracks = {}
            unmatched_detections = set(range(len(face_locations)))
            
            # Create a list of (distance, track_idx, face_idx)
            matches = []
            for t_idx in range(distances_matrix.shape[0]):
                for f_idx in range(distances_matrix.shape[1]):
                    matches.append((distances_matrix[t_idx, f_idx], t_idx, f_idx))
            
            # Sort by distance (greedy matching)
            matches.sort()
            
            matched_t = set()
            matched_f = set()

            for dist, track_idx, face_idx in matches:
                if dist < 0.6 and track_idx not in matched_t and face_idx not in matched_f:
                    track_id = track_ids[track_idx]
                    matched_tracks[track_id] = face_idx
                    unmatched_detections.discard(face_idx)
                    matched_t.add(track_idx)
                    matched_f.add(face_idx)

                    self.tracks[track_id]["rect"] = face_locations[face_idx]
                    if face_names[face_idx] != "Unknown" and face_confidences[face_idx] > self.tracks[track_id]["confidence"]:
                        self.tracks[track_id]["name"] = face_names[face_idx]
                        self.tracks[track_id]["class_name"] = face_classes[face_idx]
                        self.tracks[track_id]["confidence"] = face_confidences[face_idx]
                    
                    self.tracks[track_id]["quality"] = min(10, self.tracks[track_id]["quality"] + 3)
                    self.tracks[track_id]["last_update"] = self.frame_count
                    self.disappeared[track_id] = 0

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.disappeared[track_id] += 1
                self.tracks[track_id]["quality"] = max(0, self.tracks[track_id]["quality"] - 2)
                
                if (self.disappeared[track_id] > self.max_disappeared or 
                    self.tracks[track_id]["quality"] < self.min_quality):
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
        
        for face_idx in unmatched_detections:
            self.register(face_locations[face_idx], face_encodings[face_idx],
                          face_names[face_idx], face_classes[face_idx], face_confidences[face_idx])
        
        return self.tracks

"""
Lighting checks removed per requirements. We keep brightness as a quality metric only.
"""

def main():
    """Main function with ultra-high performance optimizations."""
    global blink_detection
    
    # Optional CLI for ID -> name/class lookup via API (Part C2)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--lookup-id', dest='lookup_id', type=str, default=None,
                        help='Lookup a student by ID using the API and print name/class, then exit')
    parser.add_argument('--grade', dest='grade', type=str, default=None, help='(Optional) Grade for class-based API lookup (e.g., 1, 2, EY1)')
    parser.add_argument('--homeroom', dest='homeroom', type=str, default=None, help='(Optional) Homeroom for class-based API lookup (e.g., 1A, 2B)')
    args, _ = parser.parse_known_args()

    if args.lookup_id:
        name, class_name = lookup_student_via_api(args.lookup_id, args.grade, args.homeroom)
        if name or class_name:
            print(f"✅ Lookup result: ID={args.lookup_id} → Name={name or 'N/A'}, Class={class_name or 'N/A'}")
        else:
            print(f"❌ No match found for ID={args.lookup_id}")
        return

    print("🚀 Starting Ultra-High Performance Facial Recognition System...")
    
    # Initialize embedding backends
    if CONFIG.get("use_landmark_embeddings", True) or CONFIG.get("use_cnn_embeddings", True):
        initialize_dlib_shape_predictor()
    if CONFIG.get("use_cnn_embeddings", True):
        initialize_dlib_cnn_model()
    
    if CONFIG["preload_known_faces"]:
        preload_face_encodings()
    
    # PIN features removed: no PIN loading
    
    rtsp_url = os.getenv("HIKVISION_RTSP_URL", "0")
    source = rtsp_url if rtsp_url != "0" else 0
    print(f"📷 Opening video source: {source}")
    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    except Exception:
        pass 
    
    print("🎥 Ultra-high performance camera initialized.")
    
    cv2.namedWindow('Ultra-Fast Face Recognition', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Ultra-Fast Face Recognition', handle_mouse_click,
                         (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    face_tracker = UltraFastFaceTracker(
        max_disappeared=CONFIG["max_tracking_age"],
        min_quality=CONFIG["tracking_quality_threshold"]
    )
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    skip_counter = 0
    
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONFIG["encoding_threads"]
    )
    
    print("✅ Ultra-fast facial recognition system ready!")

    runtime_start = time.time()
    
    try:
        while True:
            if time.time() - runtime_start >= APP_TIMEOUT_SECONDS:
                print("⏲️ Runtime limit reached (8 hours). Shutting down.")
                break
            frame_start_time = time.time()
            
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, CONFIG["flip_camera"])
            
            # Lighting checks removed
            if blink_detection['active']:
                # Handle blink detection mode
                small_frame = cv2.resize(frame, (0, 0), 
                                         fx=CONFIG["frame_resize"], 
                                         fy=CONFIG["frame_resize"],
                                         interpolation=cv2.INTER_LINEAR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations_raw = face_detector.detect_faces_optimized(small_frame)
                
                if face_locations_raw and dlib_shape_predictor is not None:
                    # Use the largest detected face; get dlib 68 landmarks on the resized RGB frame
                    largest_face = max(face_locations_raw, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    try:
                        import dlib
                        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                        gray_small = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
                        shape = dlib_shape_predictor(gray_small, rect)
                        if shape is not None:
                            pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                            left_eye = pts[36:42]
                            right_eye = pts[42:48]
                            left_ear = calculate_ear(left_eye)
                            right_ear = calculate_ear(right_eye)
                            avg_ear = (left_ear + right_ear) / 2.0

                            # Scale points back to the original frame size
                            scale = 1.0 / CONFIG["frame_resize"]
                            scaled_landmarks = [(int(px * scale), int(py * scale)) for (px, py) in pts]
                            scaled_left_eye = [(int(px * scale), int(py * scale)) for (px, py) in left_eye]
                            scaled_right_eye = [(int(px * scale), int(py * scale)) for (px, py) in right_eye]

                            draw_eye_landmarks(frame, scaled_landmarks, scaled_left_eye, scaled_right_eye, avg_ear)

                            if detect_blink(avg_ear):
                                print(f"✅ Blink verification successful for {blink_detection['person_name']}")
                                blink_detection['status_message'] = "Verification successful!"
                                # Log attendance after successful blink
                                log_attendance(
                                    blink_detection['person_name'],
                                    blink_detection['class_name'],
                                    confidence=0.9,
                                    security_score=0.9
                                )
                                time.sleep(1)
                                blink_detection['active'] = False
                    except Exception as e:
                        print(f"⚠️ Error processing dlib landmarks: {e}")
                
                frame = display_blink_detection_ui(frame)
            else:
                # Regular face recognition mode
                skip_counter += 1
                process_this_frame = skip_counter % CONFIG["skip_frames"] == 0
                frame_count += 1
                
                if time.time() - fps_start_time >= 1.0:
                    fps = frame_count / (time.time() - fps_start_time)
                    performance_metrics['avg_fps'] = fps
                    performance_metrics['peak_fps'] = max(performance_metrics.get('peak_fps', 0), fps)
                    frame_count = 0
                    fps_start_time = time.time()
                    
                    adaptive_quality_control(fps)
                
                active_faces = {}
                
                if process_this_frame:
                    small_frame = cv2.resize(frame, (0, 0), 
                                             fx=CONFIG["frame_resize"], 
                                             fy=CONFIG["frame_resize"],
                                             interpolation=cv2.INTER_LINEAR)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    face_locations_raw = face_detector.detect_faces_optimized(small_frame)
                    
                    if face_locations_raw:
                        face_locations = []
                        valid_face_regions = []
                        
                        for face_data in face_locations_raw:
                            if len(face_data) >= 4:
                                x, y, w, h = face_data[:4]
                                top, right, bottom, left = y, x + w, y + h, x
                                face_location = (top, right, bottom, left)
                                
                                is_valid, quality_score, reason = validate_face_region(rgb_small_frame, face_location)
                                
                                if is_valid:
                                    face_locations.append(face_location)
                                    face_region = rgb_small_frame[top:bottom, left:right]
                                    face_region = enhanced_face_preprocessing(face_region)
                                    valid_face_regions.append(face_region)
                                elif CONFIG.get("show_quality_metrics", False):
                                    print(f"⚠️ Face rejected: {reason} (quality: {quality_score:.2f})")
                        
                        if face_locations:
                            face_encodings, landmarks_list, encoded_indices = parallel_face_encoding(
                                rgb_small_frame, face_locations
                            )

                            if face_encodings:
                                recognition_results = []

                                encoded_face_locations = [face_locations[i] for i in encoded_indices]
                                encoded_face_regions = [valid_face_regions[i] for i in encoded_indices]

                                futures = []
                                for i, encoding in enumerate(face_encodings):
                                    face_image = encoded_face_regions[i] if i < len(encoded_face_regions) else None
                                    landmarks = landmarks_list[i] if i < len(landmarks_list) else None

                                    future = executor.submit(
                                        enhanced_face_recognition_ensemble,
                                        encoding, face_image, landmarks, True
                                    )
                                    futures.append(future)
                                
                                for future in concurrent.futures.as_completed(futures):
                                    try:
                                        result = future.result()
                                        recognition_results.append(result)
                                    except Exception as e:
                                        print(f"⚠️ Enhanced recognition future error: {e}")
                                        recognition_results.append(("Unknown", "", 0.0, {}, 0.0))
                                
                                face_names = [result[0] for result in recognition_results]
                                face_classes = [result[1] for result in recognition_results]
                                face_confidences = [result[2] for result in recognition_results]
                                face_qualities = [result[3] for result in recognition_results]
                                face_securities = [result[4] for result in recognition_results]
                                
                                for i, (name, class_name, confidence, quality_metrics, security_score) in enumerate(recognition_results):
                                    if name != "Unknown" and confidence >= CONFIG["min_recognition_threshold"]:
                                        attendance_key = f"{class_name}/{name}" if class_name else name
                                        
                                        should_log = True
                                        if attendance_key in attendance_timestamps:
                                            last_time = attendance_timestamps[attendance_key]
                                            time_diff = time.time() - last_time
                                            if time_diff < CONFIG.get("duplicate_detection_window", 300):
                                                should_log = False
                                        
                                        if should_log:
                                            quality_score = quality_metrics.get("overall_score", 0.0) if quality_metrics else 0.0
                                            
                                            if confidence >= CONFIG["confident_recognition_threshold"]:
                                                success = log_attendance(name, class_name, confidence, quality_score, security_score)
                                                if success:
                                                    print(f"⚡ High-confidence attendance: {name} (conf: {confidence:.2f})")
                                            else:
                                                if CONFIG.get("log_recognition_details", True):
                                                    print(f"❌ Low confidence rejected: {name} (conf: {confidence:.2f}, qual: {quality_score:.2f}, sec: {security_score:.2f})")
                                
                                face_locations = encoded_face_locations

                                active_faces = face_tracker.update(
                                    face_locations, face_encodings, 
                                    face_names, face_classes, face_confidences
                                )
                else:
                    active_faces = face_tracker.tracks
                
                scale = 1.0 / CONFIG["frame_resize"]
                for face_id, face_data in list(active_faces.items()): # Use list to allow modification during iteration
                    top, right, bottom, left = face_data["rect"]
                    
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
                    
                    name = face_data["name"]
                    class_name = face_data.get("class_name", "")
                    confidence = face_data.get("confidence", 0.0)
                    
                    if name == "Unknown":
                        color = (0, 0, 255); label = f"Unknown ({confidence:.2f})"
                    else:
                        attendance_key = f"{class_name}/{name}" if class_name else name
                        if attendance_key in attendance:
                            color = (255, 165, 0); label = f"{name} - Logged"
                        else:
                            color = (0, 255, 0); label = f"{name} ({confidence:.2f})"
                        
                        if class_name and name != "Unknown":
                            label = f"{name} ({class_name}) ({confidence:.2f})"
                    
                    if CONFIG.get("show_face_boxes", True):
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, label, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Find the original recognition result for quality info
                        # This is a simplification; in a real scenario, you'd store this in the tracker
                        quality_metrics = {}
                        security_score = 0.0
                        if 'recognition_results' in locals():
                             for res in recognition_results:
                                 if res[0] == name: # Simple match by name
                                     quality_metrics = res[3]
                                     security_score = res[4]
                                     break

                        frame = display_face_quality_info(
                            frame, (top, right, bottom, left), 
                            quality_metrics, security_score, confidence
                        )
            
            if CONFIG["display_fps"]:
                fps_color = (0, 255, 0) if fps >= CONFIG["target_fps"] * 0.8 else (0, 255, 255)
                cv2.putText(frame, f"FPS: {fps:.1f} (Peak: {performance_metrics.get('peak_fps', 0):.1f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            
            frame = display_enhanced_performance_metrics(frame)
            frame = display_thank_you_message(frame)
            
            frame_time = time.time() - frame_start_time
            performance_metrics['frame_times'].append(frame_time)
            
            cv2.imshow('Ultra-Fast Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
                
    except KeyboardInterrupt:
        print("\n⚡ System stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        video_capture.release()
        cv2.destroyAllWindows()
        
        if CONFIG["performance_monitoring"]:
            print("\n📊 Ultra-Performance Summary:")
            print(f"Peak FPS: {performance_metrics.get('peak_fps', 0):.1f}")
            print(f"Average FPS: {performance_metrics.get('avg_fps', 0):.1f}")
            print(f"Total faces detected: {performance_metrics['total_faces_detected']}")
            print(f"Total faces recognized: {performance_metrics['total_faces_recognized']}")
            cache_total = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
            cache_rate = (performance_metrics['cache_hits'] / max(1, cache_total) * 100)
            print(f"Cache hit rate: {cache_rate:.1f}%")
            print(f"GPU acceleration: {'ON' if GPU_AVAILABLE and CONFIG['use_gpu_acceleration'] else 'OFF'}")
        
        print("🔴 Ultra-High Performance Face Recognition System closed.")

if __name__ == "__main__":
    main()
