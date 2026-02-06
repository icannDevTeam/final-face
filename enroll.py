# enroll_local.py
# Run this script ONCE to create your metrics file.

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

try:
    import dlib  
    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False
DLIB_MODEL_PATH = os.environ.get('DLIB_LANDMARK_MODEL', 'shape_predictor_68_face_landmarks.dat')
DLIB_FACE_REC_MODEL_PATH = os.environ.get('DLIB_FACE_REC_MODEL', 'dlib_face_recognition_resnet_model_v1.dat')
_CNN_WARNING_EMITTED = False

# --- CONFIGURATION ---
# 1. SET THIS to the name of the metrics file you want to create
ENCODINGS_FILE = "encodings.pickle"


def _resolve_class_and_student(dirpath: str, dataset_root: Path) -> tuple[str, str]:
    """Infer class & student names from a directory path.

    Supports both legacy layouts (`face_dataset/StudentName`) and the newer
    Firebase layout (`face_dataset/ClassName/StudentName/pics`).
    """

    path = Path(dirpath)
    class_name = ""
    student_name = ""

    try:
        rel_parts = path.resolve().relative_to(dataset_root.resolve()).parts
    except Exception:
        rel_parts = path.parts

    rel_parts = [part for part in rel_parts if part]
    if not rel_parts:
        return class_name, student_name

    if rel_parts[-1].lower() == "pics":
        rel_parts = rel_parts[:-1]
        if not rel_parts:
            return class_name, student_name

    student_name = rel_parts[-1]
    if len(rel_parts) >= 2:
        class_name = rel_parts[-2]

    return class_name, student_name

def _compute_landmark_embedding_from_bbox(rgb_image, bbox, predictor, shape=None):
    """Compute 136-d normalized embedding from dlib 68 landmarks."""
    try:
        top, right, bottom, left = bbox
        if shape is None:
            rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            shape = predictor(gray, rect)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
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
        n = float(np.linalg.norm(emb))
        if n > 1e-6:
            emb /= n
        return emb
    except Exception:
        return None

def _compute_cnn_embedding_from_shape(rgb_image, shape, cnn_model):
    """Compute 128-d embedding from dlib face recognition model using an existing shape."""
    try:
        if cnn_model is None or shape is None:
            return None
        chip = dlib.get_face_chip(rgb_image, shape, size=150)
        descriptor = cnn_model.compute_face_descriptor(chip)
        vec = np.array(descriptor, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec
    except Exception:
        return None

def enroll_local_students(
    dataset_path: str | Path,
    encodings_file: str | Path = ENCODINGS_FILE,
):
    """
    Scans the local dataset path, generates encodings for the
    single photo of each student, and saves them to a pickle file.
    """
    global _CNN_WARNING_EMITTED

    dataset_root = Path(dataset_path).expanduser().resolve()
    encodings_file = Path(encodings_file).expanduser()

    print(f"Starting enrollment from '{dataset_root}'...")
    
    known_face_encodings = []
    known_face_names = []
    known_face_classes = []
    known_face_cnn_encodings = []
    
    # Walk through the directory structure
    # os.walk gives us:
    # dirpath (e.g., "local_dataset/10A/John_Doe")
    # dirnames (subfolders, e.g., [])
    # filenames (files in that folder, e.g., ["photo.jpg"])
    
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        # Check if we are in a student's folder (it will contain files)
        if not filenames:
            continue
            
        class_name, student_name = _resolve_class_and_student(dirpath, dataset_root)
        if not student_name:
            continue
        
        # Find the first valid image file
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                print(f"Processing: Class={class_name}, Student={student_name}, File={image_path}")

                try:
                    # Load the image
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  ⚠️ Warning: Unable to read image {image_path}, skipping.")
                        continue

                    # Initialize predictor and HOG detector once
                    global _ENROLL_PREDICTOR, _ENROLL_HOG_DET
                    if '_ENROLL_PREDICTOR' not in globals():
                        _ENROLL_PREDICTOR = None
                    if '_ENROLL_HOG_DET' not in globals():
                        _ENROLL_HOG_DET = None
                    if '_ENROLL_CNN_MODEL' not in globals():
                        _ENROLL_CNN_MODEL = None

                    if DLIB_AVAILABLE and _ENROLL_PREDICTOR is None:
                        if os.path.exists(DLIB_MODEL_PATH):
                            _ENROLL_PREDICTOR = dlib.shape_predictor(DLIB_MODEL_PATH)
                        else:
                            print(f"  ❌ Missing dlib landmark model at '{DLIB_MODEL_PATH}'. Cannot compute embeddings.")

                    if DLIB_AVAILABLE and _ENROLL_HOG_DET is None:
                        try:
                            _ENROLL_HOG_DET = dlib.get_frontal_face_detector()
                        except Exception:
                            _ENROLL_HOG_DET = None

                    if DLIB_AVAILABLE and _ENROLL_CNN_MODEL is None:
                        if os.path.exists(DLIB_FACE_REC_MODEL_PATH):
                            try:
                                _ENROLL_CNN_MODEL = dlib.face_recognition_model_v1(DLIB_FACE_REC_MODEL_PATH)
                            except Exception as cnn_ex:
                                print(f"  ⚠️ Unable to load face recognition model: {cnn_ex}")
                                _ENROLL_CNN_MODEL = None
                        else:
                            if not _CNN_WARNING_EMITTED:
                                print(f"  ⚠️ Face recognition model not found at '{DLIB_FACE_REC_MODEL_PATH}'. CNN embeddings will be skipped.")
                                _CNN_WARNING_EMITTED = True

                    # Detect faces using HOG only
                    face_locations = []
                    if _ENROLL_HOG_DET is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        dets = _ENROLL_HOG_DET(gray, 0)
                        for r in dets:
                            face_locations.append((r.top(), r.right(), r.bottom(), r.left()))
                    else:
                        print("  ❌ No dlib HOG detector available. Install dlib to proceed.")

                    if len(face_locations) == 1 and _ENROLL_PREDICTOR is not None:
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        top, right, bottom, left = face_locations[0]
                        rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
                        shape = _ENROLL_PREDICTOR(gray, rect)

                        encoding = _compute_landmark_embedding_from_bbox(rgb, face_locations[0], _ENROLL_PREDICTOR, shape)
                        if encoding is None:
                            print("  ⚠️ Could not compute landmark embedding. Skipping.")
                            continue

                        cnn_encoding = _compute_cnn_embedding_from_shape(rgb, shape, _ENROLL_CNN_MODEL)

                        known_face_encodings.append(encoding)
                        known_face_cnn_encodings.append(cnn_encoding)
                        known_face_names.append(student_name)
                        known_face_classes.append(class_name)
                        print(f"  ✅ Successfully encoded {student_name}.")
                    elif len(face_locations) > 1:
                        print(f"  ⚠️ Warning: Found {len(face_locations)} faces. Only use photos with ONE face. Skipping.")
                    else:
                        print(f"  ⚠️ Warning: No faces detected or landmark model missing for {image_path}. Skipping.")
                except Exception as e:
                    print(f"  ❌ Error processing {image_path}: {e}")
                
                # Since multiple images may exist, we use only the first valid one per student
                break 

    # --- Save the metrics to the pickle file ---
    print(f"Enrollment complete. Found {len(known_face_encodings)} students.")

    # Resolve final encodings path and print diagnostic info so callers can find the file
    encodings_file = encodings_file if isinstance(encodings_file, Path) else Path(encodings_file)
    encodings_file_parent = encodings_file.parent.resolve()
    encodings_file_parent.mkdir(parents=True, exist_ok=True)
    encodings_abs = encodings_file.resolve()
    print(f"Saving metrics to '{encodings_abs}'...")

    # Ensure cnn list aligns with others
    if len(known_face_cnn_encodings) < len(known_face_encodings):
        known_face_cnn_encodings.extend([None] * (len(known_face_encodings) - len(known_face_cnn_encodings)))

    data = {
        "encodings": known_face_encodings,  # Legacy key
        "landmark_encodings": known_face_encodings,
        "cnn_encodings": known_face_cnn_encodings,
        "names": known_face_names,
        "classes": known_face_classes,
    }

    # Write and validate
    with open(encodings_abs, 'wb') as f:
        pickle.dump(data, f)

    if encodings_abs.exists():
        try:
            size = encodings_abs.stat().st_size
        except Exception:
            size = -1
        print(f"✅ Encodings file written: {encodings_abs} (size={size} bytes)")
    else:
        print("❌ Failed to write encodings file — file not found after write attempt.")

    print("✅ Done. You can now run your main application.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download photos from Firebase Storage and generate face encodings."
    )
    parser.add_argument(
        "--encodings",
        default=ENCODINGS_FILE,
        help="Output pickle file for encodings (default: %(default)s)",
    )
    parser.add_argument(
        "--firebase-prefix",
        default=None,
        help="Remote folder prefix inside the bucket (default: FIREBASE_DATASET_PREFIX or 'face_dataset').",
    )
    parser.add_argument(
        "--firebase-credentials",
        default=None,
        help="Path to the Firebase service-account JSON (overrides FIREBASE_CREDENTIALS).",
    )
    parser.add_argument(
        "--firebase-bucket",
        default=None,
        help="Firebase Storage bucket name override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from firebase_dataset_sync import sync_face_dataset_from_firebase
    except ImportError:
        print("❌ firebase_admin is required for Firebase syncing. Install dependencies and retry.")
        raise

    with tempfile.TemporaryDirectory(prefix="firebase_face_dataset_") as temp_dir:
        dataset_root = Path(temp_dir) / "face_dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)

        stats = sync_face_dataset_from_firebase(
            destination_root=dataset_root,
            remote_prefix=args.firebase_prefix,
            credentials_path=args.firebase_credentials,
            storage_bucket=args.firebase_bucket,
            skip_existing=True,
        )
        print(f"📥 Firebase sync complete ({stats}).")

        enroll_local_students(dataset_path=dataset_root, encodings_file=args.encodings)


if __name__ == "__main__":
    main()