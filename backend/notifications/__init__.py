"""Push notifications — Settings Pass 2, Notifications & Alerts.

Split on the same line the rest of the backend already draws:

    notifications/rules.py     pure: documents in, "what should fire" out
    notifications/service.py   this half's I/O: reads settings + device tokens,
                               sends via FCM, records the dedupe marker

The split is what lets verify_notifications.py exercise every trigger rule with
no Firebase, no FCM and no network — the same reason ml/ and data/ are split.
"""
from . import rules, service  # noqa: F401
