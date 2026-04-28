"""
devices_config.py — Shared loader for Hikvision device configs.

Resolves passwords in this priority order:
  1. devices.local.json (gitignored)        — preferred for production
  2. Per-device env var named in `password_env` field of devices.json
  3. Generic env var HIKVISION_PASS         — last-resort default
  4. Plaintext `password` field in devices.json (LEGACY — emits warning)

This keeps the committed devices.json safe to share while letting operators
inject the real password via env or local override.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEVICES_FILE = SCRIPT_DIR / "devices.json"
DEVICES_LOCAL_FILE = SCRIPT_DIR / "devices.local.json"


def _load_local_overrides() -> dict[str, dict]:
    """Read devices.local.json (gitignored) and return name → device dict."""
    if not DEVICES_LOCAL_FILE.exists():
        return {}
    try:
        data = json.loads(DEVICES_LOCAL_FILE.read_text())
        return {d["name"]: d for d in data if isinstance(d, dict) and "name" in d}
    except Exception as e:
        print(f"⚠️  Failed to read {DEVICES_LOCAL_FILE.name}: {e}", file=sys.stderr)
        return {}


def _resolve_password(device: dict, override: dict | None) -> str | None:
    """Resolve the password for a single device, in priority order."""
    # 1. Local override file
    if override and override.get("password"):
        return override["password"]

    # 2. Per-device env var (e.g. HIKVISION_PASS_BASEMENT1)
    env_name = device.get("password_env")
    if env_name:
        val = os.environ.get(env_name)
        if val:
            return val

    # 3. Generic HIKVISION_PASS
    val = os.environ.get("HIKVISION_PASS")
    if val:
        return val

    # 4. Legacy plaintext field — warn loudly
    legacy = device.get("password")
    if legacy:
        print(
            f"⚠️  SECURITY: device {device.get('name', '?')!r} is using a plaintext "
            "password from devices.json. Move it to devices.local.json or set "
            f"{device.get('password_env') or 'HIKVISION_PASS'}.",
            file=sys.stderr,
        )
        return legacy

    return None


def load_devices(filter_substr: str | None = None, only_enabled: bool = True) -> list[dict]:
    """
    Load device configs and resolve passwords from env/local override.

    Returns a list of dicts with at least: name, ip, password, enabled.
    Skips devices where the password cannot be resolved.
    """
    if not DEVICES_FILE.exists():
        print(f"✗ {DEVICES_FILE} not found", file=sys.stderr)
        sys.exit(1)

    base = json.loads(DEVICES_FILE.read_text())
    overrides = _load_local_overrides()

    resolved: list[dict] = []
    for d in base:
        if not isinstance(d, dict):
            continue
        if only_enabled and not d.get("enabled", True):
            continue
        ov = overrides.get(d.get("name", ""))
        password = _resolve_password(d, ov)
        if not password:
            print(
                f"✗ No password resolved for device {d.get('name', '?')!r}. "
                "Set the env var or add to devices.local.json. Skipping.",
                file=sys.stderr,
            )
            continue
        merged = {**d, **(ov or {}), "password": password}
        resolved.append(merged)

    if filter_substr:
        needle = filter_substr.lower()
        resolved = [d for d in resolved if needle in d["name"].lower() or needle in d["ip"]]

    return resolved
