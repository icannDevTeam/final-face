Instructions:
1. Enroll from Firebase (script automatically downloads photos to a temporary location, builds `encodings.pickle`, and cleans up):

	```bash
	python3 enroll_local.py
	```

	Use the optional arguments to point at custom Firebase folders or credentials:

	- `--firebase-prefix <remote/subfolder>`
	- `--firebase-credentials <path/to/service_account.json>`
	- `--firebase-bucket <bucket-name>`

2. Collect metrics (+ encodings)
3. Run main file

Model assets (place beside the scripts or set env vars):
- `shape_predictor_68_face_landmarks.dat` (`DLIB_LANDMARK_MODEL`)
- `dlib_face_recognition_resnet_model_v1.dat` (`DLIB_FACE_REC_MODEL`)

The enrollment step still generates both 136-D landmark embeddings and 128-D CNN embeddings for compatibility, but the runtime now consumes only the CNN descriptors for maximum throughput. Keep the CNN weights available; landmark embeddings are retained solely for future experimentation and liveness overlays.

---

API lookup (Part C2) — ID → Name/Class

You can look up a student’s name and class by ID using the API.

Option A: Direct C2 (ID-only)
- Set environment variables per your API doc:
	- BSS_C2_URL: Full endpoint URL for the ID lookup
	- BSS_C2_BODY_KEY: Body key for the student ID (default: IdStudent)
	- BSS_C2_BODY_MODE: 'single' (default) to send {IdStudent: "..."} or 'list' to send {IdStudent: ["..."]}
- Then run:

	python3 main.py --lookup-id <STUDENT_ID>

This prints the Name and Class (if class/homeroom field is present in the response).

Option B: Class-based lookup (requires grade/homeroom)

	python3 main.py --lookup-id <STUDENT_ID> --grade <GRADE> --homeroom <HOMEROOM>

Notes
- Token retrieval is automatic.
- Detection pipeline is HOG-only with dlib 68-landmarks.
- Recognition now runs in CNN-only mode by default (`CONFIG["embedding_mode"] = "cnn"`). You can re-enable landmark embeddings in CONFIG if you want to experiment with fusion, but the fastest setup keeps them disabled.
- Ensemble distance weights can be left as "default" or customized in CONFIG.
- Firebase sync is configured through `firebase_dataset_sync.py`. Override defaults with:
	- `FIREBASE_CREDENTIALS` → path to the service-account JSON (defaults to `facial-attendance-binus-firebase-adminsdk.json`).
	- `FIREBASE_STORAGE_BUCKET` → bucket name (defaults to `<project_id>.appspot.com`).
	- `FIREBASE_DATASET_PREFIX` → folder prefix (defaults to `face_dataset`).# final
