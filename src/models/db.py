"""
Database connection setup for LogiMind.
Reads DATABASE_URL from the environment (set in docker-compose.yml locally,
and via AWS environment variables / secrets manager in production).
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://logimind:logimind_dev_pw@localhost:5432/logimind",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
