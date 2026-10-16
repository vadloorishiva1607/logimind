# LogiMind — Backend Skeleton

## Structure
```
logimind/
  docker-compose.yml      # Postgres+pgvector, and the FastAPI backend
  .env.example            # copy to .env and fill in real values
  backend/
    Dockerfile
    requirements.txt
    app/
      main.py             # FastAPI entrypoint + health checks
      db.py                # SQLAlchemy engine/session setup
      llm.py                # single point of control for the LLM provider
```

## Day 1 quickstart

1. Copy the env file and fill in your OpenAI key and LangSmith key:
   ```
   cp .env.example .env
   ```
2. Build and start everything:
   ```
   docker compose up --build
   ```
3. Confirm the API is up:
   ```
   curl http://localhost:8000/health
   ```
4. Confirm the API can reach Postgres:
   ```
   curl http://localhost:8000/health/db
   ```
   Expect: `{"status":"ok","db_check":1}`

If step 4 fails, check `docker compose logs db` — the healthcheck usually means the backend started before Postgres was ready; docker-compose is already configured to wait for it (`depends_on: condition: service_healthy`), so this should be rare.

## Next steps (Day 2 onward)
- Add the schema (drivers, fuel_transactions, settlements, escrow_transactions, performance, shipments, policy_chunks, audit_log) as a SQLAlchemy models file or a raw migration script.
- Write and run the seed script.
- Build the RAG ingestion script.
- Build the LangGraph Supervisor + first agent.

## Switching LLM provider later
Everything routes through `app/llm.py` (`get_llm()` / `get_embeddings()`). To switch to Anthropic later, this is the only file that changes.
