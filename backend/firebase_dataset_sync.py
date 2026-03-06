"""Utilities for synchronizing the face dataset from Firebase Storage.

This module pulls photos stored in Firebase Storage and writes them to the local
`face_dataset/Class/Student/pics` hierarchy expected by `enroll_local.py`.
The Firebase service account JSON file is read from the path defined by the
`FIREBASE_CREDENTIALS` environment variable (defaults to the repository service
account file). The target storage bucket can be overridden with
`FIREBASE_STORAGE_BUCKET`; otherwise it falls back to `<project_id>.appspot.com`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import firebase_admin
from firebase_admin import credentials, storage

DEFAULT_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS",
    "facial-attendance-binus-firebase-adminsdk.json",
)
DEFAULT_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
DEFAULT_REMOTE_PREFIX = os.getenv("FIREBASE_DATASET_PREFIX", "face_dataset")

_firebase_app: Optional[firebase_admin.App] = None


@dataclass
class SyncStats:
    """Simple struct summarizing a Firebase sync operation."""

    downloaded: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        return f"downloaded={self.downloaded}, skipped={self.skipped}"


def _normalize_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    cleaned = prefix.strip().strip("/")
    return cleaned


def initialize_firebase_app(
    credentials_path: Optional[str] = None,
    storage_bucket: Optional[str] = None,
) -> firebase_admin.App:
    """Initialise (or reuse) the Firebase Admin SDK application."""

    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    cred_path = credentials_path or DEFAULT_CREDENTIALS_PATH
    if not cred_path:
        raise ValueError("No Firebase credentials path provided.")

    cred_path = os.path.abspath(cred_path)
    if not os.path.isfile(cred_path):
        raise FileNotFoundError(
            f"Firebase credentials not found at '{cred_path}'. Set FIREBASE_CREDENTIALS or pass the path explicitly."
        )

    cred = credentials.Certificate(cred_path)
    bucket_name = storage_bucket or DEFAULT_STORAGE_BUCKET
    if not bucket_name:
        project_id = getattr(cred, "project_id", None)
        if project_id:
            bucket_name = f"{project_id}.firebasestorage.app"

    options = {"storageBucket": bucket_name} if bucket_name else None
    _firebase_app = firebase_admin.initialize_app(cred, options)
    return _firebase_app


def _split_relative_path(remote_path: str, prefix: str) -> Path:
    if prefix:
        prefix_with_sep = prefix + "/"
        if remote_path.startswith(prefix_with_sep):
            remote_path = remote_path[len(prefix_with_sep) :]
        elif remote_path == prefix:
            return Path()
    return Path(remote_path)


def _flatten_to_student_path(rel_path: Path) -> Path:
    """Normalize a remote-relative path into `StudentName/...`.

    If blobs are stored as `Class/Student/pics/file.jpg` or
    `Class/Student/file.jpg` this will map them to `Student/file.jpg` so
    the local layout becomes `face_dataset/Student/...` (no class folders).

    If the remote path is already `Student/file.jpg` it is returned as-is.
    Empty or invalid paths return an empty Path().
    """
    parts = [p for p in rel_path.parts if p]
    if not parts:
        return Path()

    # Remove any 'pics' segments that are used as containers
    parts = [p for p in parts if p.lower() != "pics"]

    # If the last part looks like a filename, pick the preceding part as student
    if len(parts) >= 2 and "." in parts[-1]:
        student = parts[-2]
        tail = parts[-1:]
        return Path(student, *tail)

    # If there is a single part, treat it as student or file
    if len(parts) == 1:
        return Path(parts[0])

    # Fallback: use the last two elements as student/filename
    student = parts[-2]
    filename = parts[-1]
    return Path(student, filename)


def sync_face_dataset_from_firebase(
    destination_root: str | Path = "face_dataset",
    remote_prefix: Optional[str] = None,
    credentials_path: Optional[str] = None,
    storage_bucket: Optional[str] = None,
    skip_existing: bool = True,
) -> SyncStats:
    """Download photos from Firebase Storage to the local face dataset tree.

    Args:
        destination_root: Local folder (typically `face_dataset`).
        remote_prefix: Folder prefix inside the bucket (defaults to `face_dataset`).
        credentials_path: Optional override for the service-account JSON file.
        storage_bucket: Optional override for the Firebase Storage bucket name.
        skip_existing: If True, files already present locally are not downloaded
            again (helps speed up incremental syncs).
    Returns:
        SyncStats with counts of downloaded and skipped files.
    """

    app = initialize_firebase_app(credentials_path, storage_bucket)
    bucket = storage.bucket(app=app)

    dest_root = Path(destination_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    prefix = _normalize_prefix(remote_prefix or DEFAULT_REMOTE_PREFIX)

    stats = SyncStats()
    blobs = bucket.list_blobs(prefix=prefix if prefix else None)

    for blob in blobs:
        # Skip "directories" (Firebase Storage represents them as zero-length placeholders)
        if blob.name.endswith("/"):
            continue

        rel_path = _split_relative_path(blob.name, prefix)
        if not rel_path or rel_path.name == "":
            continue

        # Map remote layout into flat student-first layout: face_dataset/<Student>/file
        mapped = _flatten_to_student_path(rel_path)
        if not mapped or mapped.name == "":
            continue

        local_path = dest_root / mapped
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
            stats.skipped += 1
            continue

        blob.download_to_filename(str(local_path))
        stats.downloaded += 1

    return stats


__all__ = [
    "SyncStats",
    "initialize_firebase_app",
    "sync_face_dataset_from_firebase",
]
