#!/usr/bin/env python3
"""
Migrate root-level Firestore collections + Storage objects into the
tenant-scoped layout — Phase A6.

Source (legacy)              →  Destination (tenant-scoped)
──────────────────────────────────────────────────────────────────────
students/{id}                →  tenants/{tid}/students/{id}
student_metadata/{id}        →  tenants/{tid}/student_metadata/{id}
face_descriptors/{id}        →  tenants/{tid}/face_descriptors/{id}
attendance/{date}            →  tenants/{tid}/attendance/{date}
attendance/{date}/records/*  →  tenants/{tid}/attendance/{date}/records/*
spoof_attempts/{date}/logs/* →  tenants/{tid}/spoof_attempts/{date}/logs/*
access_logs/*                →  tenants/{tid}/access_logs/*
face_dataset/...             →  tenants/{tid}/face_dataset/...

Usage:
    python3 migrate_to_tenants.py --dry-run                 # default; counts only
    python3 migrate_to_tenants.py --execute                 # actually copy
    python3 migrate_to_tenants.py --execute --skip-storage  # firestore only
    python3 migrate_to_tenants.py --collection students     # only one collection

Safety:
  * Default mode is DRY-RUN — prints planned writes without touching anything.
  * Source documents are NEVER deleted by this script. Run a separate cleanup
    after the dual-read window with `--cleanup-legacy` (also dry-run by default).
  * Subcollections are walked recursively.
  * Idempotent: re-running copies the same doc to the same destination
    (overwriting), so a partial run can be resumed.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Iterable

import firebase_admin
from firebase_admin import credentials, firestore, storage

import tenancy

SERVICE_ACCOUNT_PATH = os.path.join(
    os.path.dirname(__file__),
    "facial-attendance-binus-firebase-adminsdk.json",
)
DEFAULT_BUCKET = "facial-attendance-binus.firebasestorage.app"

LEGACY_COLLECTIONS = {
    "students": "students",
    "student_metadata": "student_metadata",
    "face_descriptors": "face_descriptors",
    "attendance": "attendance",
    "spoof_attempts": "spoof_attempts",
    "access_logs": "access_logs",
}


def init_firebase() -> tuple[firestore.Client, "storage.bucket.Bucket"]:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"storageBucket": DEFAULT_BUCKET})
    return firestore.client(), storage.bucket()


# ─── Recursive Firestore copy ──────────────────────────────────────────
def copy_doc_recursive(
    src_doc_ref: firestore.DocumentReference,
    dst_doc_ref: firestore.DocumentReference,
    *,
    dry_run: bool,
    counters: dict,
    indent: int = 0,
) -> None:
    snap = src_doc_ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        counters["docs"] += 1
        if not dry_run:
            dst_doc_ref.set(data, merge=False)
        if dry_run:
            print(f"{'  ' * indent}[dry] copy {src_doc_ref.path}  →  {dst_doc_ref.path}  ({len(data)} fields)")
    # Recurse into subcollections regardless of parent existence
    for sub in src_doc_ref.collections():
        for child in sub.stream():
            child_dst = dst_doc_ref.collection(sub.id).document(child.id)
            child_src = src_doc_ref.collection(sub.id).document(child.id)
            copy_doc_recursive(child_src, child_dst, dry_run=dry_run, counters=counters, indent=indent + 1)


def copy_collection(
    db: firestore.Client,
    src_path: str,
    dst_path: str,
    *,
    dry_run: bool,
    counters: dict,
) -> None:
    print(f"\n── {src_path}  →  {dst_path}")
    src_coll = db.collection(src_path)
    for doc_snap in src_coll.stream():
        src_ref = src_coll.document(doc_snap.id)
        dst_ref = _doc_ref_from_path(db, f"{dst_path}/{doc_snap.id}")
        copy_doc_recursive(src_ref, dst_ref, dry_run=dry_run, counters=counters)


def _doc_ref_from_path(db: firestore.Client, path: str) -> firestore.DocumentReference:
    parts = path.split("/")
    if len(parts) % 2 != 0:
        raise ValueError(f"Document path must have even segments: {path}")
    ref: firestore.CollectionReference | firestore.DocumentReference = db.collection(parts[0])
    for i, p in enumerate(parts[1:], start=1):
        if i % 2 == 1:
            ref = ref.document(p)
        else:
            ref = ref.collection(p)
    return ref  # type: ignore[return-value]


# ─── Storage migration ─────────────────────────────────────────────────
def migrate_storage(
    bucket,
    tenant_id: str,
    *,
    dry_run: bool,
    counters: dict,
) -> None:
    src_prefix = f"{tenancy.LEGACY_STORAGE_PREFIX}/"
    dst_prefix = f"{tenancy.storage_face_dataset_prefix(tenant_id)}/"
    print(f"\n── storage: {src_prefix}*  →  {dst_prefix}*")
    blobs = list(bucket.list_blobs(prefix=src_prefix))
    for blob in blobs:
        new_name = dst_prefix + blob.name[len(src_prefix):]
        counters["blobs"] += 1
        if dry_run:
            print(f"  [dry] copy gs://{bucket.name}/{blob.name}  →  gs://{bucket.name}/{new_name}")
        else:
            # Skip if already migrated
            new_blob = bucket.blob(new_name)
            if new_blob.exists():
                continue
            bucket.copy_blob(blob, bucket, new_name)


# ─── Main ──────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually perform copies (default is dry-run)")
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--skip-firestore", action="store_true")
    parser.add_argument(
        "--collection",
        action="append",
        default=None,
        help="Only migrate this legacy collection (repeatable). Defaults to all.",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    tenant_id = tenancy.get_tenant_id(args.tenant_id)
    db, bucket = init_firebase()
    counters = {"docs": 0, "blobs": 0}

    print(f"Migration plan — tenant '{tenant_id}'  (mode: {'DRY-RUN' if dry_run else 'EXECUTE'})")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    selected = args.collection or list(LEGACY_COLLECTIONS.keys())

    if not args.skip_firestore:
        for legacy_name in selected:
            if legacy_name not in LEGACY_COLLECTIONS:
                print(f"  [warn] unknown legacy collection: {legacy_name}")
                continue
            dst_collection_name = LEGACY_COLLECTIONS[legacy_name]
            copy_collection(
                db,
                src_path=legacy_name,
                dst_path=f"{tenancy.tenant_doc(tenant_id)}/{dst_collection_name}",
                dry_run=dry_run,
                counters=counters,
            )

    if not args.skip_storage:
        migrate_storage(bucket, tenant_id, dry_run=dry_run, counters=counters)

    print()
    print("─" * 60)
    print(f"Documents copied:   {counters['docs']}")
    print(f"Storage objects:    {counters['blobs']}")
    print(f"Mode:               {'DRY-RUN (no writes)' if dry_run else 'EXECUTE (committed)'}")
    if dry_run:
        print("\nRe-run with --execute to perform the migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
