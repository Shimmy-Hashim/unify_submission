"""Premium Due Reminders — the scheduled half of Notifications & Alerts.

    cd backend && venv\\Scripts\\activate && python -m notifications.run_premium_reminders
    ... --dry-run          scan and report, send nothing (no events recorded)
    ... --uid <uid>        one user, for testing
    ... --lead-days 7      override the reminder window

SCHEDULING — WHY A STANDALONE SCRIPT AND NOT A SCHEDULER IN THE API PROCESS
The three candidates, against this project's actual environment:

  Cloud Functions + Cloud Scheduler — NOT AVAILABLE. Both require the Blaze
    billing plan; this project is on Spark (the same constraint already
    recorded in Section 8, where Firebase Storage was declined for insurance
    PDFs for exactly this reason). Nothing was assumed here — see the report.

  APScheduler inside the FastAPI process — available, but a poor fit and it
    would cost a new dependency. The backend runs as `uvicorn main:app
    --reload` on a developer laptop: --reload forks a reloader parent and a
    worker child, so an in-process job schedule lives in the child and is
    destroyed and rebuilt on every source edit, firing unpredictably during
    development. More fundamentally, the process is not up 24/7, so a daily
    in-process job would simply not run on any day the server is off — the
    same limitation a cron job has, but with an extra dependency and a much
    less obvious failure mode.

  A standalone idempotent script (this file) — chosen. It matches the
    backend/scripts/ pattern already in the repo, it is directly runnable and
    testable, it needs no new dependency, and it can be driven by whatever
    scheduler the host actually has: Windows Task Scheduler here, cron on a
    Linux host, or a real Cloud Scheduler job later without rewriting the
    logic. Register it on Windows with:

      schtasks /create /tn "Unify premium reminders" /tr ^
        "C:\\path\\to\\unify\\backend\\venv\\Scripts\\python.exe -m notifications.run_premium_reminders" ^
        /sc daily /st 09:00

MISSED DAYS are handled in the rule, not here: the window is "renews within
lead_days", not "renews in exactly lead_days", so a day the machine was off
does not lose that policy's only reminder. The per-renewal dedupe marker in
service.py is what keeps that from re-sending on each subsequent day — the two
mechanisms are designed as a pair. See rules.find_due_premium_reminders.

WHAT THIS SCRIPT DOES NOT DO: it never writes to insurancePolicies. The dedupe
marker lives in notificationEvents precisely so the Section 8 nine-field policy
schema stays exactly as specified.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Runnable directly as well as via -m, matching backend/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firebase import get_db, init_firebase  # noqa: E402
from notifications import rules, service  # noqa: E402

logger = logging.getLogger(__name__)


def policies_for(db, uid: str) -> list[dict]:
    return [
        snap.to_dict()
        for snap in (db.collection("users").document(uid)
                     .collection("insurancePolicies").stream())
    ]


def run_for_user(db, uid: str, *, now: datetime, lead_days: int,
                 dry_run: bool) -> dict:
    """One user's due reminders. Returns a per-user summary."""
    due = rules.find_due_premium_reminders(
        policies_for(db, uid), now=now, lead_days=lead_days)
    if not due:
        return {"uid": uid, "due": 0, "sent": 0}
    if dry_run:
        # Nothing is sent AND no event is claimed, so a dry run leaves the next
        # real run free to send exactly what it reported.
        for note in due:
            print(f"    would send: {note.event_key} — {note.body}")
        return {"uid": uid, "due": len(due), "sent": 0, "dry_run": True}

    summary = service.deliver(db, uid, due)
    return {"uid": uid, "due": len(due), **summary}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--uid", help="run for one user instead of all")
    parser.add_argument("--lead-days", type=int, default=rules.PREMIUM_LEAD_DAYS)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be sent; send nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_firebase()
    db = get_db()
    now = datetime.now(timezone.utc)

    uids = [args.uid] if args.uid else [
        snap.id for snap in db.collection("users").select([]).stream()
    ]
    print(f"Premium reminder scan — {len(uids)} user(s), "
          f"{args.lead_days}-day window, "
          f"{'DRY RUN' if args.dry_run else 'live'}, {now.isoformat()}")

    totals = {"users": 0, "due": 0, "sent": 0, "skipped_disabled": 0,
              "skipped_duplicate": 0, "skipped_no_token": 0, "failed": 0}
    for uid in uids:
        try:
            summary = run_for_user(db, uid, now=now, lead_days=args.lead_days,
                                   dry_run=args.dry_run)
        except Exception:
            # One user's bad data must not stop the scan for everyone else.
            logger.exception("premium reminder scan failed for %s", uid)
            totals["failed"] += 1
            continue
        totals["users"] += 1
        for key in ("due", "sent", "skipped_disabled", "skipped_duplicate",
                    "skipped_no_token", "failed"):
            totals[key] += summary.get(key, 0)
        if summary.get("due"):
            print(f"  {uid}: {summary}")

    print(f"Done: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
