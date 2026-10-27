from datetime import date

from supportpilot.agent.mock_db import (
    CURATED,
    N_RANDOM_USERS,
    REFERENCE_DATE,
    build,
    connect,
)


def test_build_creates_expected_counts(tmp_path):
    counts = build(tmp_path / "t.db")
    assert counts["users"] == len(CURATED) + N_RANDOM_USERS
    assert counts["subscriptions"] == counts["users"]
    assert counts["invoices"] > 0


def test_build_is_deterministic(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    build(a)
    build(b)
    dump_a = list(connect(a).iterdump())
    dump_b = list(connect(b).iterdump())
    assert dump_a == dump_b


def test_foreign_keys_are_valid(tmp_path):
    path = tmp_path / "t.db"
    build(path)
    assert connect(path).execute("PRAGMA foreign_key_check").fetchall() == []


def days_ago(iso_date: str) -> int:
    return (REFERENCE_DATE - date.fromisoformat(iso_date)).days


def test_curated_scenarios(tmp_path):
    path = tmp_path / "t.db"
    build(path)
    con = connect(path)

    def invoices(uid):
        return con.execute(
            "SELECT * FROM invoices WHERE user_id = ? ORDER BY issued_at DESC", (uid,)
        ).fetchall()

    def refunds(uid):
        return con.execute("SELECT * FROM refunds WHERE user_id = ?", (uid,)).fetchall()

    assert invoices(1)[0]["status"] == "paid" and days_ago(invoices(1)[0]["issued_at"]) <= 30
    assert [days_ago(i["issued_at"]) for i in invoices(2)] == [60]
    assert invoices(3)[0]["status"] == "refunded"
    assert refunds(3)[0]["status"] == "processed"
    assert invoices(4) == []
    assert invoices(5)[0]["status"] == "failed"
    sub5 = con.execute("SELECT status FROM subscriptions WHERE user_id = 5").fetchone()
    assert sub5["status"] == "past_due"
    assert refunds(6)[0]["status"] == "pending_approval"
    open_tickets = con.execute(
        "SELECT COUNT(*) FROM tickets WHERE user_id = 7 AND status = 'open'"
    ).fetchone()[0]
    assert open_tickets == 2
    sub8 = con.execute("SELECT status FROM subscriptions WHERE user_id = 8").fetchone()
    assert sub8["status"] == "canceled"
