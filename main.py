from dotenv import load_dotenv

# Must run before anything below is imported -- uvicorn imports
# "src.api.routes:app" (and transitively src.agent.memory, which reads
# DATABASE_URL at module import time, not lazily) after this process
# starts, so .env has to be loaded into os.environ first. This was never
# actually happening before: nothing in this codebase called
# load_dotenv() anywhere, so .env has never been read by the app in any
# session -- it only ever "worked" when DATABASE_URL happened to be
# manually exported in that shell. Found 2026-10-04 while live-testing
# the MCP/RAG routing changes.
load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.routes:app", host="0.0.0.0", port=8000, reload=True)