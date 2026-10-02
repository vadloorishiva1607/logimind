"""
LogiMind seed script.

What this is for: your schema is empty on its own, and demo questions
like "why did D102's settlement decrease?" only work if D102 actually
has a real settlement drop in the data. This script does two things:

  1. Generates realistic BACKGROUND data (10 drivers, several weeks of
     fuel/settlement/performance/escrow history each) using Faker, so
     the dataset doesn't look sparse or obviously fake.
  2. INJECTS the specific scripted anomalies your demo relies on:
       - D102: settlement drop this week (fuel spike + escrow deduction)
       - D104: declining safety score over 3 weeks
       - D105: sharp escrow balance drop
       - S10023: a delayed shipment tied to D102

Run it with:  python scripts/seed.py --reset
(--reset truncates existing data first, so it's safe to re-run
after schema changes during development)
"""

import argparse
import os
import random
from datetime import date, timedelta

import psycopg2
from faker import Faker

fake = Faker()
random.seed(42)  # reproducible demo data across re-runs

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://logimind:logimind_dev_pw@localhost:5432/logimind",
)

DRIVER_IDS = [f"D{100 + i}" for i in range(1, 11)]  # D101 .. D110
WEEKS_OF_HISTORY = 6
TODAY = date.today()


def week_ending(weeks_ago: int) -> date:
    """Most recent Sunday, minus N weeks."""
    last_sunday = TODAY - timedelta(days=TODAY.weekday() + 1)
    return last_sunday - timedelta(weeks=weeks_ago)


def reset_tables(cur):
    cur.execute(
        """
        TRUNCATE fuel_transactions, settlements, escrow_transactions,
                 performance, shipments, audit_log
        RESTART IDENTITY CASCADE;
        TRUNCATE drivers CASCADE;
        """
    )


def seed_drivers(cur):
    roles = ["standard", "standard", "standard", "onboarding", "standard"]
    for driver_id in DRIVER_IDS:
        cur.execute(
            """
            INSERT INTO drivers (driver_id, full_name, status, onboarding_date, role)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                driver_id,
                fake.name(),
                "onboarding" if driver_id == "D103" else "active",
                fake.date_between(start_date="-1y", end_date="-2m"),
                random.choice(roles),
            ),
        )


def seed_fuel_settlement_performance(cur):
    for driver_id in DRIVER_IDS:
        for w in range(WEEKS_OF_HISTORY, 0, -1):
            wk_end = week_ending(w - 1)

            # --- baseline realistic values ---
            gallons = round(random.uniform(150, 220), 2)
            cost_per_gallon = round(random.uniform(3.6, 4.1), 2)
            mpg = round(random.uniform(6.2, 7.4), 2)
            fuel_cost = round(gallons * cost_per_gallon, 2)

            gross_pay = round(random.uniform(1800, 2400), 2)
            fuel_deduction = fuel_cost
            escrow_deduction = round(random.uniform(50, 100), 2)
            other_deduction = round(random.uniform(20, 60), 2)

            on_time_pct = round(random.uniform(88, 98), 2)
            safety_score = round(random.uniform(85, 97), 2)
            late_deliveries = random.randint(0, 2)
            incidents = 0

            # --- scripted anomalies (this week only, w == 1) ---
            if driver_id == "D102" and w == 1:
                # Settlement-drop demo: fuel spike + bigger escrow deduction
                mpg = round(mpg * 0.8, 2)          # efficiency declined
                fuel_cost = round(fuel_cost * 1.35, 2)
                fuel_deduction = fuel_cost
                escrow_deduction = round(escrow_deduction + 150, 2)

            if driver_id == "D104" and w <= 3:
                # 3-week declining safety score trend
                decline_factor = (4 - w) * 6  # bigger drop each week closer to now
                safety_score = round(max(55, safety_score - decline_factor), 2)
                incidents = 1 if w == 1 else 0

            net_pay = round(
                gross_pay - fuel_deduction - escrow_deduction - other_deduction, 2
            )

            cur.execute(
                """
                INSERT INTO fuel_transactions
                    (driver_id, transaction_date, gallons, cost_usd, mpg, location)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (driver_id, wk_end, gallons, fuel_cost, mpg, fake.city()),
            )

            cur.execute(
                """
                INSERT INTO settlements
                    (driver_id, week_ending, gross_pay, fuel_deduction,
                     escrow_deduction, other_deduction, net_pay)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    driver_id, wk_end, gross_pay, fuel_deduction,
                    escrow_deduction, other_deduction, net_pay,
                ),
            )

            cur.execute(
                """
                INSERT INTO performance
                    (driver_id, period_start, period_end, on_time_pct,
                     safety_score, late_deliveries, incidents)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    driver_id, wk_end - timedelta(days=6), wk_end,
                    on_time_pct, safety_score, late_deliveries, incidents,
                ),
            )


def seed_escrow(cur):
    for driver_id in DRIVER_IDS:
        balance = round(random.uniform(1500, 3000), 2)
        for w in range(WEEKS_OF_HISTORY, 0, -1):
            wk_end = week_ending(w - 1)
            amount = round(random.uniform(50, 100), 2)
            balance = round(balance + amount, 2)
            reason = "Weekly escrow contribution"

            # Scripted anomaly: sharp escrow balance drop for D105, most recent week
            if driver_id == "D105" and w == 1:
                amount = round(-(balance * 0.4), 2)
                balance = round(balance + amount, 2)
                reason = "Adjustment — see settlement deduction detail"

            cur.execute(
                """
                INSERT INTO escrow_transactions
                    (driver_id, transaction_date, type, amount_usd,
                     balance_after, reason)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    driver_id, wk_end,
                    "withdrawal" if amount < 0 else "deposit",
                    amount, balance, reason,
                ),
            )


def seed_shipments(cur):
    # A handful of normal shipments
    for i in range(1, 21):
        shipment_id = f"S{10000 + i}"
        driver_id = random.choice(DRIVER_IDS)
        scheduled = fake.date_time_between(start_date="-14d", end_date="now")
        cur.execute(
            """
            INSERT INTO shipments
                (shipment_id, driver_id, origin, destination,
                 scheduled_eta, actual_eta, status, delay_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                shipment_id, driver_id, fake.city(), fake.city(),
                scheduled, scheduled, "delivered", None,
            ),
        )

    # Scripted anomaly: S10023, delayed, tied to D102 for the delay-investigation demo
    scheduled = fake.date_time_between(start_date="-2d", end_date="-1d")
    actual = scheduled + timedelta(hours=4)
    cur.execute(
        """
        INSERT INTO shipments
            (shipment_id, driver_id, origin, destination,
             scheduled_eta, actual_eta, status, delay_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (shipment_id) DO UPDATE SET
            actual_eta = EXCLUDED.actual_eta,
            status = EXCLUDED.status,
            delay_reason = EXCLUDED.delay_reason
        """,
        (
            "S10023", "D102", "Dallas, TX", "Memphis, TN",
            scheduled, actual, "delayed", "Traffic + driver rest-break delay",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="Truncate existing data before seeding (safe to re-run)",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            if args.reset:
                reset_tables(cur)
            seed_drivers(cur)
            seed_fuel_settlement_performance(cur)
            seed_escrow(cur)
            seed_shipments(cur)
        conn.commit()
        print("Seed complete: 10 drivers, "
              f"{WEEKS_OF_HISTORY} weeks of history each, "
              "plus scripted anomalies for D102, D104, D105, S10023.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
