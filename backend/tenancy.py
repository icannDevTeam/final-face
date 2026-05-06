"""
Multi-tenancy helpers — Phase A1 foundation.

All code that previously hit root-level Firestore collections / Storage prefixes
should route through these helpers so that a single env var (TENANT_ID) flips
the entire backend to the new tenant-scoped layout.

Schema (Firestore):
    tenants/{tenantId}
        settings/config            — tenant config doc (timezone, lateCutoff, …)
        users/{email}              — RBAC (role: owner|admin|viewer)
        students/{studentId}
        student_metadata/{employeeNo}
        face_descriptors/{id}
        attendance/{date}/records/{employeeNo}
        access_logs/{id}
        spoof_attempts/{date}/logs/{id}
        devices/{deviceId}
        consents/{studentId}
        policy_versions/{versionId}
        data_requests/{requestId}
        erasure_log/{eventId}
        biometric_access_log/{id}

Schema (Storage):
    tenants/{tenantId}/face_dataset/{homeroom}/{studentName}/{filename}

During the dual-read window (Phase A6), `legacy_paths_enabled()` returns True
and writers double-write to both locations.
"""
from __future__ import annotations

import os
from typing import Optional

# ─── Environment ──────────────────────────────────────────────────────
DEFAULT_TENANT_ID = "binus-simprug"

def get_tenant_id(explicit: Optional[str] = None) -> str:
    """Resolve the active tenant id. Prefers explicit arg, then env, then default."""
    if explicit:
        return explicit
    return os.environ.get("TENANT_ID", DEFAULT_TENANT_ID)


def tenant_aware_enabled() -> bool:
    """True once the system has cut over to tenant-scoped reads/writes."""
    return os.environ.get("TENANT_AWARE", "false").lower() in ("true", "1", "yes")


def legacy_paths_enabled() -> bool:
    """True during the dual-write/dual-read migration window."""
    return os.environ.get("LEGACY_PATHS", "true").lower() in ("true", "1", "yes")


# ─── Firestore path builders ──────────────────────────────────────────
def tenant_doc(tenant_id: Optional[str] = None) -> str:
    return f"tenants/{get_tenant_id(tenant_id)}"


def students_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/students"


def student_metadata_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/student_metadata"


def face_descriptors_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/face_descriptors"


def attendance_day_doc(date_str: str, tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/attendance/{date_str}"


def attendance_record_path(date_str: str, employee_no: str, tenant_id: Optional[str] = None) -> str:
    return f"{attendance_day_doc(date_str, tenant_id)}/records/{employee_no}"


def devices_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/devices"


def consents_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/consents"


def policy_versions_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/policy_versions"


def data_requests_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/data_requests"


def erasure_log_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/erasure_log"


def biometric_access_log_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/biometric_access_log"


def security_incidents_path(tenant_id: Optional[str] = None) -> str:
    """Append-only feed of pickup security incidents (unknown chaperone, suspended scans, etc.)."""
    return f"{tenant_doc(tenant_id)}/security_incidents"


# ─── PickupGuard (chaperone pick-up) ─────────────────────────────────
def chaperones_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/chaperones"


def pickup_events_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/pickup_events"


def pickup_overrides_path(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/pickup_overrides"


def pickup_settings_doc(tenant_id: Optional[str] = None) -> str:
    return f"{tenant_doc(tenant_id)}/settings/pickup"


def pickup_onboarding_path(tenant_id: Optional[str] = None) -> str:
    """Pending parent-submitted onboarding records, awaiting admin approval."""
    return f"{tenant_doc(tenant_id)}/pickup_onboarding"


def id_allocations_doc(name: str, tenant_id: Optional[str] = None) -> str:
    """Single-doc counter (e.g. 'chaperone-counter') incremented atomically."""
    return f"{tenant_doc(tenant_id)}/id_allocations/{name}"


def storage_pickup_capture_path(event_id: str, tenant_id: Optional[str] = None) -> str:
    """Per-scan face JPEG captured by Hikvision at the gate."""
    return f"tenants/{get_tenant_id(tenant_id)}/pickup_captures/{event_id}.jpg"


# ─── Phase 2: terminal registry + release groups + tablet pairing ───
def terminals_path(tenant_id: Optional[str] = None) -> str:
    """Hikvision face terminals managed by this tenant (admin-editable)."""
    return f"{tenant_doc(tenant_id)}/terminals"


def release_groups_path(tenant_id: Optional[str] = None) -> str:
    """Binds N terminalIds + 1 tabletDeviceId + gradeLabel for teacher iPad."""
    return f"{tenant_doc(tenant_id)}/release_groups"


def tablet_devices_path(tenant_id: Optional[str] = None) -> str:
    """Paired teacher iPads (mirror of tv_devices but bound to release groups)."""
    return f"{tenant_doc(tenant_id)}/tablet_devices"


def storage_chaperone_face_prefix(chaperone_id: str, tenant_id: Optional[str] = None) -> str:
    return f"tenants/{get_tenant_id(tenant_id)}/chaperone_faces/{chaperone_id}"


# Reserved employeeNo block for chaperones (Hikvision side).
# Students use BINUS-issued IDs (10-digit, never starts with 9).
CHAPERONE_EMPLOYEENO_PREFIX = "9"


def is_chaperone_employee_no(employee_no: str) -> bool:
    """Classify a Hikvision event by employeeNo prefix."""
    return bool(employee_no) and str(employee_no).startswith(CHAPERONE_EMPLOYEENO_PREFIX)


# ─── Storage path builders ────────────────────────────────────────────
def storage_face_dataset_prefix(tenant_id: Optional[str] = None) -> str:
    """Returns the storage path prefix for face dataset photos (no trailing slash)."""
    return f"tenants/{get_tenant_id(tenant_id)}/face_dataset"


def storage_student_folder(homeroom: str, student_name: str, tenant_id: Optional[str] = None) -> str:
    return f"{storage_face_dataset_prefix(tenant_id)}/{homeroom}/{student_name}"


# ─── Legacy path equivalents (for dual-read) ──────────────────────────
LEGACY_STUDENTS = "students"
LEGACY_STUDENT_METADATA = "student_metadata"
LEGACY_FACE_DESCRIPTORS = "face_descriptors"
LEGACY_ATTENDANCE = "attendance"
LEGACY_STORAGE_PREFIX = "face_dataset"


# ─── Tenant config defaults ───────────────────────────────────────────
DEFAULT_TENANT_CONFIG = {
    "name": "BINUS School Simprug",
    "slug": DEFAULT_TENANT_ID,
    "timezone": "Asia/Jakarta",
    "lateCutoffHHmm": "07:30",
    "geofence": {
        "type": "radius",
        "centerLat": -6.2349,
        "centerLng": 106.7956,
        "radiusMeters": 250,
    },
    "branding": {
        "schoolName": "BINUS School",
        "logoUrl": "/logo.jpg",
        "primaryColor": "#1e40af",
    },
    "downstreamSink": {
        "type": "binus",
        "config": {},
    },
    "dataRetention": {
        "attendanceDays": 365 * 7,        # 7 years (school records)
        "photoDays": 365,                 # 1 year after departure
        "spoofLogDays": 90,
        "accessLogDays": 365,
    },
    "currentPolicyVersionId": None,       # set by init_tenant when policy seeded
}
