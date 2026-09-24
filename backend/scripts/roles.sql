-- Database role separation.
-- Run this once, as the RDS master user, after schema.sql.
--
-- Note: RAG policy storage (previously policy_chunks in Postgres) now
-- lives in ChromaDB, not this database -- see the write-access note at
-- the bottom of this file for how that's controlled instead.
--
-- Two roles remain relevant here:
--   logimind_app    -- used by the FastAPI backend at runtime. Read-only
--                       on operational data, INSERT-only on audit_log
--                       (agents log decisions but never modify or erase
--                       past log entries).
--   logimind_ingest -- reserved for any future admin-only write task
--                       against this database. Currently unused for RAG
--                       (see note below) but kept for that separation-
--                       of-privilege pattern if a future write-capable
--                       task needs one.
--
-- This is defense-in-depth: even if application code has a bug, the
-- runtime database connection physically lacks write permission on the
-- operational tables.

CREATE ROLE logimind_app LOGIN PASSWORD 'CHANGE_ME_APP_PASSWORD';
CREATE ROLE logimind_ingest LOGIN PASSWORD 'CHANGE_ME_INGEST_PASSWORD';

-- logimind_app: read-only on operational data
GRANT SELECT ON drivers, fuel_transactions, settlements, escrow_transactions,
    performance, shipments TO logimind_app;

-- logimind_app may only INSERT into audit_log (never SELECT/UPDATE/DELETE
-- its own past entries -- an agent shouldn't be able to erase what it logged)
GRANT INSERT ON audit_log TO logimind_app;
GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO logimind_app;

GRANT INSERT ON audit_log TO logimind_ingest;
GRANT USAGE, SELECT ON SEQUENCE audit_log_id_seq TO logimind_ingest;

-- WRITE ACCESS CONTROL FOR RAG DOCUMENTS (ChromaDB, not Postgres):
-- Chroma's embedded mode has no user/role system of its own -- write
-- access is controlled by which process/OS user can write to
-- CHROMA_PERSIST_DIR on disk, not by a SQL GRANT. In practice:
--   - The running backend (logimind_app's process) should mount or run
--     against that directory in a way that a chat request's code path
--     never calls ingest_policies.py or any Chroma write method.
--   - Only an admin manually running `python scripts/ingest_policies.py`
--     writes to it -- same "never triggered by chat" principle as the
--     pgvector version, enforced by code path rather than DB permission.
--   - For a stronger boundary if you outgrow embedded mode later, a
--     client-server Chroma deployment DOES support per-request auth
--     tokens, which would let you enforce this the same way as
--     logimind_ingest above.
