from src.agent.agent import build_graph
from src.agent.memory import get_checkpointer


def ask(db, question: str, thread_id: str = "default", role: str = "dispatcher"):
    checkpointer = get_checkpointer()
    graph = build_graph(db, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"question": question, "role": role}, config=config)
    return {
        "answer": result.get("answer"),
        "agents_used": result.get("agents_used", []),
    }