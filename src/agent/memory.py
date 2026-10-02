import os

from langgraph.checkpoint.postgres import PostgresSaver

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://logimind_app:CHANGE_ME_APP_PASSWORD@localhost:5432/logimind",
)

_checkpointer_cm = None
_checkpointer = None


def get_checkpointer():
    global _checkpointer_cm, _checkpointer
    if _checkpointer is None:
        _checkpointer_cm = PostgresSaver.from_conn_string(DATABASE_URL)
        _checkpointer = _checkpointer_cm.__enter__()
        _checkpointer.setup()
    return _checkpointer


def close_checkpointer():
    global _checkpointer_cm, _checkpointer
    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)
        _checkpointer_cm = None
        _checkpointer = None