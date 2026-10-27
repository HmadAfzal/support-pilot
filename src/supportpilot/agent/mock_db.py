from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path("data/mock_db/support.db")
SEED = 42
N_RANDOM_USERS = 192


REFERENCE_DATE = date(2026, 10, 1)

PLAN_PRICE = {"free": 0, "premium": 2900, "ultimate": 9900}

SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE subscriptions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    plan TEXT NOT NULL CHECK (plan IN ('free', 'premium', 'ultimate')),
    status TEXT NOT NULL CHECK (status IN ('active', 'canceled', 'past_due')),
    seats INTEGER NOT NULL,
    started_at TEXT NOT NULL
);
CREATE TABLE invoices (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'failed', 'refunded')),
    issued_at TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    subject TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    created_at TEXT NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY,
    invoice_id INTEGER NOT NULL REFERENCES invoices(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    amount_cents INTEGER NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending_approval', 'approved', 'rejected', 'processed')),
    requested_at TEXT NOT NULL
);
"""

FIRST_NAMES = [
    "Aisha", "Omar", "Sara", "Bilal", "Hina", "Usman", "Zainab", "Ali", "Maria", "Daniel",
    "Priya", "Lucas", "Emma", "Noah", "Fatima", "Hassan", "Olivia", "Ethan", "Mei", "Kenji",
]  # fmt: skip
LAST_NAMES = [
    "Khan", "Ahmed", "Malik", "Sheikh", "Butt", "Smith", "Brown", "Patel", "Garcia", "Muller",
    "Rossi", "Tanaka", "Lee", "Silva", "Novak", "Hussain", "Iqbal", "Raza", "Ali", "Wilson",
]  # fmt: skip
TICKET_SUBJECTS = [
    "Pipeline stuck in pending state",
    "Cannot add a teammate to the project",
    "Runner is offline after upgrade",
    "Question about invoice amount",
    "Merge request cannot be merged",
    "Need help with permissions for a group",
]

CURATED = [
    # 1: paid invoice 5 days ago -> refund is inside the window
    {"name": "Test User One", "plan": "premium", "seats": 3,
     "invoices": [(5, "paid"), (35, "paid")]},
    # 2: only paid invoice is 60 days old -> refund is outside the window
    {"name": "Test User Two", "plan": "premium", "seats": 2, "invoices": [(60, "paid")]},
    # 3: invoice already refunded
    {"name": "Test User Three", "plan": "ultimate", "seats": 1,
     "invoices": [(20, "refunded")],
     "refunds": [(0, "processed", "Duplicate charge", 18)]},
    # 4: free plan, no invoices
    {"name": "Test User Four", "plan": "free", "seats": 1, "invoices": []},
    # 5: latest payment failed, subscription past due
    {"name": "Test User Five", "plan": "premium", "status": "past_due", "seats": 5,
     "invoices": [(3, "failed"), (33, "paid")]},
    # 6: refund already requested and waiting for approval
    {"name": "Test User Six", "plan": "premium", "seats": 1,
     "invoices": [(10, "paid")],
     "refunds": [(0, "pending_approval", "Charged after cancelling", 2)]},
    # 7: two open tickets
    {"name": "Test User Seven", "plan": "premium", "seats": 4,
     "invoices": [(12, "paid")],
     "tickets": [("Pipeline stuck in pending state", "open", 6),
                 ("Runner is offline after upgrade", "open", 2)]},
    # 8: canceled subscription, older paid invoices
    {"name": "Test User Eight", "plan": "ultimate", "status": "canceled", "seats": 2,
     "invoices": [(45, "paid"), (75, "paid")]},
]  # fmt: skip


def iso(days_ago: int) -> str:
    return (REFERENCE_DATE - timedelta(days=days_ago)).isoformat()


def add_user(
    con: sqlite3.Connection,
    *,
    name: str,
    email: str,
    plan: str,
    status: str = "active",
    seats: int = 1,
    joined_days_ago: int = 200,
    invoices: list[tuple[int, str]] | None = None,
    tickets: list[tuple[str, str, int]] | None = None,
    refunds: list[tuple[int, str, str, int]] | None = None,
) -> int:
    cur = con.cursor()
    cur.execute(
        "INSERT INTO users (name, email, created_at) VALUES (?, ?, ?)",
        (name, email, iso(joined_days_ago)),
    )
    user_id = cur.lastrowid
    assert user_id is not None
    cur.execute(
        "INSERT INTO subscriptions (user_id, plan, status, seats, started_at) VALUES (?,?,?,?,?)",
        (user_id, plan, status, seats, iso(joined_days_ago)),
    )
    amount = PLAN_PRICE[plan] * seats
    invoice_ids: list[int] = []
    for days_ago, inv_status in invoices or []:
        cur.execute(
            "INSERT INTO invoices (user_id, amount_cents, status, issued_at, description)"
            " VALUES (?,?,?,?,?)",
            (user_id, amount, inv_status, iso(days_ago), f"{plan.title()} plan, {seats} seat(s)"),
        )
        assert cur.lastrowid is not None
        invoice_ids.append(cur.lastrowid)
    for subject, t_status, days_ago in tickets or []:
        cur.execute(
            "INSERT INTO tickets (user_id, subject, status, created_at) VALUES (?,?,?,?)",
            (user_id, subject, t_status, iso(days_ago)),
        )
    for inv_index, r_status, reason, days_ago in refunds or []:
        cur.execute(
            "INSERT INTO refunds (invoice_id, user_id, amount_cents, reason, status, requested_at)"
            " VALUES (?,?,?,?,?,?)",
            (invoice_ids[inv_index], user_id, amount, reason, r_status, iso(days_ago)),
        )
    return user_id


def random_user(rng: random.Random, n: int) -> dict:
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    plan = rng.choices(["free", "premium", "ultimate"], weights=[40, 40, 20])[0]
    status, seats, invoices, refunds = "active", 1, [], []
    if plan != "free":
        status = rng.choices(["active", "canceled", "past_due"], weights=[80, 10, 10])[0]
        seats = rng.randint(1, 25)
        offset = rng.randint(0, 29)
        statuses = ["paid", "paid", "paid"]
        if status == "past_due":
            statuses[0] = "failed"
        elif rng.random() < 0.05:
            statuses[1] = "refunded"
            refunds = [(1, "processed", "Billing error", offset + 25)]
        invoices = [(offset + 30 * k, s) for k, s in enumerate(statuses)]
    tickets = [
        (rng.choice(TICKET_SUBJECTS), rng.choice(["open", "closed"]), rng.randint(1, 90))
        for _ in range(rng.choice([0, 0, 1, 2]))
    ]
    return {
        "name": f"{first} {last}",
        "email": f"{first}.{last}.{n}@example.com".lower(),
        "plan": plan,
        "status": status,
        "seats": seats,
        "joined_days_ago": rng.randint(60, 700),
        "invoices": invoices,
        "tickets": tickets,
        "refunds": refunds,
    }


def build(db_path: Path | str = DB_PATH, seed: int = SEED) -> dict[str, int]:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    rng = random.Random(seed)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    for i, spec in enumerate(CURATED, start=1):
        add_user(con, email=f"test.user.{i}@example.com", **spec)
    for n in range(len(CURATED) + 1, len(CURATED) + N_RANDOM_USERS + 1):
        add_user(con, **random_user(rng, n))
    con.commit()
    counts = {
        table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("users", "subscriptions", "invoices", "tickets", "refunds")
    }
    con.close()
    return counts


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


if __name__ == "__main__":
    counts = build()
    print(f"Wrote {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:<14}{n:>5}")
