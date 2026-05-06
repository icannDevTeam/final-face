"""
PickupGuard — chaperone employeeNo allocator.

Hikvision uses a 1+ digit numeric employeeNo per enrolled face. Students
use BINUS-issued IDs (10-digit, never start with '9'). Chaperones get
their own block prefixed with `9` so the listener can tell who's who at
event time without an extra Firestore lookup.

Counter doc: tenants/{tid}/id_allocations/chaperone-counter
              { last: <int>, prefix: '9', updatedAt }

Allocation is atomic via Firestore transaction; safe under concurrent
admin approvals.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import firestore

import tenancy

# Start at 9_000_000_001 so we always have 10-digit IDs (matches BINUS shape)
_FIRST_CHAPERONE_NO = 9_000_000_000


def allocate_chaperone_employee_no(tenant_id: Optional[str] = None) -> str:
    """Reserve and return the next chaperone employeeNo as a string."""
    if not firebase_admin._apps:
        raise RuntimeError("firebase_admin not initialized")

    db = firestore.client()
    tid = tenancy.get_tenant_id(tenant_id)
    ref = db.document(tenancy.id_allocations_doc("chaperone-counter", tid))

    @firestore.transactional
    def _bump(tx):
        snap = ref.get(transaction=tx)
        cur = (snap.to_dict() or {}).get("last", _FIRST_CHAPERONE_NO) if snap.exists else _FIRST_CHAPERONE_NO
        nxt = max(cur, _FIRST_CHAPERONE_NO) + 1
        tx.set(ref, {
            "last": nxt,
            "prefix": tenancy.CHAPERONE_EMPLOYEENO_PREFIX,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "tenantId": tid,
        }, merge=True)
        return nxt

    nxt = _bump(db.transaction())
    return str(nxt)


if __name__ == "__main__":
    # Quick smoke test (allocates one — caller should not run in production)
    import sys
    from firebase_admin import credentials
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(
            "facial-attendance-binus-firebase-adminsdk.json"
        ))
    print(allocate_chaperone_employee_no(sys.argv[1] if len(sys.argv) > 1 else None))
