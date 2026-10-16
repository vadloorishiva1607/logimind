-- LogiMind schema. Run once against your Postgres (local or RDS).
-- Note: RAG policy storage moved to ChromaDB (embedded mode) -- see
-- backend/scripts/ingest_policies.py. This schema now covers ONLY the
-- structured operational data; no pgvector extension needed here.

CREATE TABLE IF NOT EXISTS drivers (
    driver_id       VARCHAR(10) PRIMARY KEY,
    full_name       VARCHAR(100) NOT NULL,
    status          VARCHAR(20) NOT NULL CHECK (status IN ('active', 'onboarding', 'suspended')),
    onboarding_date DATE,
    role            VARCHAR(30),
    created_at      TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fuel_transactions (
    id              SERIAL PRIMARY KEY,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    transaction_date DATE NOT NULL,
    gallons         NUMERIC(6,2),
    cost_usd        NUMERIC(8,2),
    mpg             NUMERIC(5,2),
    location        VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_fuel_driver ON fuel_transactions(driver_id);

CREATE TABLE IF NOT EXISTS settlements (
    id              SERIAL PRIMARY KEY,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    week_ending     DATE NOT NULL,
    gross_pay       NUMERIC(9,2),
    fuel_deduction  NUMERIC(9,2),
    escrow_deduction NUMERIC(9,2),
    other_deduction NUMERIC(9,2),
    net_pay         NUMERIC(9,2)
);
CREATE INDEX IF NOT EXISTS idx_settlements_driver ON settlements(driver_id);

CREATE TABLE IF NOT EXISTS escrow_transactions (
    id              SERIAL PRIMARY KEY,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    transaction_date DATE NOT NULL,
    type            VARCHAR(20) CHECK (type IN ('deposit', 'withdrawal', 'adjustment')),
    amount_usd      NUMERIC(9,2),
    balance_after   NUMERIC(9,2),
    reason          VARCHAR(200)
);
CREATE INDEX IF NOT EXISTS idx_escrow_driver ON escrow_transactions(driver_id);

CREATE TABLE IF NOT EXISTS performance (
    id              SERIAL PRIMARY KEY,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    period_start    DATE,
    period_end      DATE,
    on_time_pct     NUMERIC(5,2),
    safety_score    NUMERIC(5,2),
    late_deliveries INT,
    incidents       INT
);
CREATE INDEX IF NOT EXISTS idx_performance_driver ON performance(driver_id);

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id     VARCHAR(10) PRIMARY KEY,
    driver_id       VARCHAR(10) REFERENCES drivers(driver_id),
    origin          VARCHAR(100),
    destination     VARCHAR(100),
    scheduled_eta   TIMESTAMP,
    actual_eta      TIMESTAMP,
    status          VARCHAR(20) CHECK (status IN ('in_transit', 'delayed', 'delivered')),
    delay_reason    VARCHAR(200)
);
CREATE INDEX IF NOT EXISTS idx_shipments_driver ON shipments(driver_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id              SERIAL PRIMARY KEY,
    request_id      UUID DEFAULT gen_random_uuid(),
    user_role       VARCHAR(30),
    agent_used      VARCHAR(50),
    tool_used       VARCHAR(100),
    decision        VARCHAR(20) CHECK (decision IN ('allowed', 'denied', 'refused')),
    created_at      TIMESTAMP DEFAULT now()
);
