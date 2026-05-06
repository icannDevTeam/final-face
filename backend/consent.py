"""
Consent gate — Phase B2.

Wraps every enrollment / face-descriptor write so it refuses unless the
student has an active, current-version consent on file. Backend (Admin SDK)
honors this gate at the application layer; Firestore rules ALSO refuse
client-side writes to consents (admin-managed only).

Schema:
    tenants/{tid}/consents/{studentId}
        {
          studentId, tenantId,
          guardianName, guardianEmail, guardianRelation,
          policyVersionId, consentedAt, ipAddress, userAgent,
          signatureMethod: 'click' | 'esign' | 'paper-uploaded',
          signatureRef,
          expiresAt,                # optional — for re-consent cycles
          withdrawnAt, withdrawalReason,
        }

Status helpers return one of:
    'active'         — consent valid & matches current policy version
    'stale'          — consent valid but policy version has been bumped
    'withdrawn'      — guardian revoked
    'expired'        — past expiresAt
    'missing'        — no consent record at all
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import firestore

import tenancy
import tenant_config


@dataclass
class ConsentStatus:
    state: str                                # 'active' | 'stale' | 'withdrawn' | 'expired' | 'missing'
    consent: Optional[dict]
    current_policy_version_id: Optional[str]

    @property
    def allows_enrollment(self) -> bool:
        return self.state == "active"


class ConsentRequiredError(RuntimeError):
    """Raised when a biometric write is attempted without active consent."""
    def __init__(self, student_id: str, status: ConsentStatus):
        self.student_id = student_id
        self.status = status
        super().__init__(
            f"Consent gate refused enrollment for student '{student_id}': state={status.state}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_consent_status(student_id: str, tenant_id: Optional[str] = None) -> ConsentStatus:
    """Return the consent state for a student in a tenant."""
    if not firebase_admin._apps:
        raise RuntimeError("firebase_admin not initialized; call this after your app initializes Firebase")

    db = firestore.client()
    tid = tenancy.get_tenant_id(tenant_id)
    cfg = tenant_config.load_tenant_config(tid)
    current_version = cfg.get("currentPolicyVersionId")

    snap = db.document(f"{tenancy.consents_path(tid)}/{student_id}").get()
    if not snap.exists:
        return ConsentStatus("missing", None, current_version)

    data = snap.to_dict() or {}
    if data.get("withdrawnAt"):
        return ConsentStatus("withdrawn", data, current_version)
    expires = data.get("expiresAt")
    if expires and expires < _now_iso():
        return ConsentStatus("expired", data, current_version)
    if current_version and data.get("policyVersionId") != current_version:
        return ConsentStatus("stale", data, current_version)
    return ConsentStatus("active", data, current_version)


def assert_can_enroll(student_id: str, tenant_id: Optional[str] = None) -> ConsentStatus:
    """
    Raise ConsentRequiredError if the student has no active consent.
    Honors the kill-switch env var CONSENT_GATE_ENABLED (default: 'true').
    Set to 'false' for the migration window so existing students keep working
    until their guardians have completed the consent flow.
    """
    import os
    enabled = os.environ.get("CONSENT_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
    status = get_consent_status(student_id, tenant_id)
    if not enabled:
        return status
    if not status.allows_enrollment:
        raise ConsentRequiredError(student_id, status)
    return status


def record_consent(
    *,
    student_id: str,
    guardian_name: str,
    guardian_email: str,
    policy_version_id: str,
    guardian_relation: str = "guardian",
    signature_method: str = "click",
    signature_ref: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    expires_at: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> dict:
    """Persist a consent record (Admin SDK only). Idempotent on (tenant, studentId)."""
    if not firebase_admin._apps:
        raise RuntimeError("firebase_admin not initialized")
    db = firestore.client()
    tid = tenancy.get_tenant_id(tenant_id)
    doc = {
        "studentId": student_id,
        "tenantId": tid,
        "guardianName": guardian_name,
        "guardianEmail": guardian_email.lower(),
        "guardianRelation": guardian_relation,
        "policyVersionId": policy_version_id,
        "consentedAt": _now_iso(),
        "signatureMethod": signature_method,
        "signatureRef": signature_ref,
        "ipAddress": ip_address,
        "userAgent": user_agent,
        "expiresAt": expires_at,
        "withdrawnAt": None,
        "withdrawalReason": None,
    }
    db.document(f"{tenancy.consents_path(tid)}/{student_id}").set(doc, merge=False)
    return doc


def withdraw_consent(
    *,
    student_id: str,
    reason: str = "guardian_request",
    tenant_id: Optional[str] = None,
) -> bool:
    """Mark consent as withdrawn. Triggers downstream deletion in B5."""
    if not firebase_admin._apps:
        raise RuntimeError("firebase_admin not initialized")
    db = firestore.client()
    tid = tenancy.get_tenant_id(tenant_id)
    ref = db.document(f"{tenancy.consents_path(tid)}/{student_id}")
    if not ref.get().exists:
        return False
    ref.update({
        "withdrawnAt": _now_iso(),
        "withdrawalReason": reason,
    })
    return True
