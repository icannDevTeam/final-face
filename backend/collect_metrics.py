#!/usr/bin/env python3
"""
collect_metrics.py

Reads a student CSV (idStudent,idBinusian,fileName,filePath,hasFile), downloads available photos,
and (optionally) computes face-recognition metrics (face count, face encoding) using the
`face_recognition` package if installed. Metrics are written as JSON Lines to
`api_testing/metrics_face_recognition.jsonl` by default.

Usage:
  python collect_metrics.py --csv api_testing/student_report_1_1A_20251030_165056.csv

This script is resilient if `face_recognition` is not installed: it will still download
images and write an entry with detection status set to "skipped".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sqlite3
import hashlib

try:
    import requests
except Exception:
    print("Missing dependency 'requests'. Please install with: pip install requests")
    raise

try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except Exception:
    FACE_LIB_AVAILABLE = False


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create DB and return a connection. Table uses idStudent as primary key."""
    safe_mkdir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            idStudent TEXT PRIMARY KEY,
            idBinusian TEXT,
            fileName TEXT,
            source_url TEXT,
            downloaded INTEGER,
            local_path TEXT,
            detection_status TEXT,
            face_count INTEGER,
            encoding TEXT,
            encoding_size_bytes INTEGER,
            encoding_bits INTEGER,
            encoding_hash TEXT,
            timestamp_utc TEXT,
            last_updated TEXT
        )
        """
    )
    conn.commit()
    return conn


def save_metrics_to_db(conn: sqlite3.Connection, entry: Dict[str, Any], store_encoding: bool = True) -> None:
    cur = conn.cursor()

    encoding_json = None
    encoding_size = None
    encoding_bits = None
    encoding_hash = None

    enc = entry.get('encoding')
    if store_encoding and enc:
        try:
            encoding_json = json.dumps(enc, separators=(',', ':'))
            encoding_bytes = encoding_json.encode('utf-8')
            encoding_size = len(encoding_bytes)
            encoding_bits = encoding_size * 8
            encoding_hash = hashlib.sha256(encoding_bytes).hexdigest()
        except Exception:
            encoding_json = None

    cur.execute(
        "INSERT OR REPLACE INTO metrics (idStudent, idBinusian, fileName, source_url, downloaded, local_path, detection_status, face_count, encoding, encoding_size_bytes, encoding_bits, encoding_hash, timestamp_utc, last_updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            entry.get('idStudent'),
            entry.get('idBinusian'),
            entry.get('fileName'),
            entry.get('source_url'),
            1 if entry.get('downloaded') else 0,
            entry.get('local_path'),
            entry.get('detection_status'),
            entry.get('face_count'),
            encoding_json,
            encoding_size,
            encoding_bits,
            encoding_hash,
            entry.get('timestamp_utc'),
            datetime.utcnow().isoformat() + 'Z',
        ),
    )
    conn.commit()


def download_image(url: str, dest: Path, timeout: int = 15) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1024 * 8):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url} -> {e}")
        return False


def compute_face_metrics(image_path: Path) -> Dict[str, Any]:
    """Return metrics: face_count, encoding (list) or None, detection_status."""
    if not FACE_LIB_AVAILABLE:
        return {"detection_status": "skipped", "face_count": 0, "encoding": None}

    try:
        img = face_recognition.load_image_file(str(image_path))
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, known_face_locations=locations)
        if not encodings:
            return {"detection_status": "no_face_found", "face_count": 0, "encoding": None}

        # Use first face only (common approach for single-subject photos)
        enc0 = encodings[0]
        # Convert to list of floats for JSON serialization; round to 6 decimals to reduce size
        enc_list = [round(float(x), 6) for x in enc0.tolist()]
        return {"detection_status": "ok", "face_count": len(encodings), "encoding": enc_list}
    except Exception as e:
        return {"detection_status": f"error:{e}", "face_count": 0, "encoding": None}


def process_csv(csv_path: Path, out_dir: Path, metrics_path: Path, db_path: Optional[Path] = None, store_encoding: bool = True) -> None:
    safe_mkdir(out_dir)
    safe_mkdir(metrics_path.parent)

    conn = None
    if db_path:
        conn = init_db(db_path)

    written = 0
    with open(csv_path, newline='', encoding='utf-8') as fh_in, open(metrics_path, 'a', encoding='utf-8') as metrics_out:
        reader = csv.reader(fh_in)
        for row in reader:
            if not row:
                continue
            # Expect format: idStudent,idBinusian,fileName,filePath,hasFile
            # be defensive about row lengths
            idStudent = row[0].strip() if len(row) > 0 else ''
            idBinusian = row[1].strip() if len(row) > 1 else ''
            fileName = row[2].strip() if len(row) > 2 else ''
            fileUrl = row[3].strip() if len(row) > 3 else ''
            hasFile = row[4].strip() if len(row) > 4 else 'No'

            entry: Dict[str, Any] = {
                "idStudent": idStudent,
                "idBinusian": idBinusian,
                "fileName": fileName,
                "source_url": fileUrl,
                "hasFile": hasFile,
                "timestamp_utc": datetime.utcnow().isoformat() + 'Z'
            }

            if hasFile.lower() != 'yes' or not fileUrl:
                entry.update({"downloaded": False, "note": "no_file_reported"})
                metrics_out.write(json.dumps(entry) + "\n")
                if conn:
                    save_metrics_to_db(conn, entry, store_encoding=store_encoding)
                written += 1
                continue

            # create a safe local filename
            suffix = Path(fileName).suffix or '.jpg'
            local_name = f"{idStudent}{suffix}"
            local_path = out_dir / local_name

            downloaded = download_image(fileUrl, local_path)
            entry.update({"downloaded": downloaded, "local_path": str(local_path) if downloaded else None})

            if not downloaded:
                entry.update({"note": "download_failed"})
                metrics_out.write(json.dumps(entry) + "\n")
                if conn:
                    save_metrics_to_db(conn, entry, store_encoding=store_encoding)
                written += 1
                continue

            # compute face metrics
            metrics = compute_face_metrics(local_path)
            entry.update(metrics)

            metrics_out.write(json.dumps(entry) + "\n")
            if conn:
                save_metrics_to_db(conn, entry, store_encoding=store_encoding)
            written += 1

    if conn:
        conn.close()

    print(f"Wrote {written} metric records to {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download student photos and compute face-recognition metrics.")
    p.add_argument('--csv', type=Path, default=Path('api_testing/student_report_1_1A_20251030_165056.csv'), help='Path to student CSV')
    p.add_argument('--out-dir', type=Path, default=Path('student_photo_downloaded'), help='Directory to store downloaded images')
    p.add_argument('--metrics', type=Path, default=Path('api_testing/metrics_face_recognition.jsonl'), help='JSONL file for metrics')
    p.add_argument('--db', type=Path, default=Path('api_testing/metrics_face_recognition.db'), help='SQLite DB file to write metrics to (default: api_testing/metrics_face_recognition.db)')
    p.add_argument('--no-encoding', dest='store_encoding', action='store_false', help='Do not store full encoding in the DB (store only hash/size)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    print(f"CSV: {csv_path}\nOut dir: {args.out_dir}\nMetrics file: {args.metrics}\nDB: {args.db}\nStore encoding: {args.store_encoding}\nFace lib available: {FACE_LIB_AVAILABLE}")

    db_path = args.db if args.db else None
    process_csv(csv_path, args.out_dir, args.metrics, db_path=db_path, store_encoding=args.store_encoding)


if __name__ == '__main__':
    main()
