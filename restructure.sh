#!/bin/bash
set -e

# Run from the repo root (E:/AI/IK/Capstone Project/logimind_skeleton/logimind)

# 1. New skeleton
mkdir -p src/agent src/tools src/models src/prompts src/utils src/api tests data/knowledge_base scripts logs

# 2. Move files (git mv preserves history)
git mv backend/app/graph.py src/agent/_graph_old.py   # temporary, split below
git mv backend/app/db.py src/models/db.py
git mv backend/app/llm.py src/models/llm_client.py
git mv backend/app/agents/settlement.py src/tools/settlement.py
git mv backend/app/agents/rag.py src/tools/rag.py
git mv backend/app/main.py src/api/routes.py

git mv backend/data/policies data/knowledge_base
git mv backend/scripts/ingest_policies.py backend/scripts/roles.sql backend/scripts/schema.sql backend/scripts/seed.py scripts/
git mv backend/tests/test_checkpointer.py tests/test_checkpointer.py
git mv backend/requirements.txt requirements.txt
git mv backend/Dockerfile Dockerfile

# 3. Clean up now-empty backend/ tree
git rm backend/app/__init__.py backend/app/agents/__init__.py
rmdir backend/app/agents backend/app backend/data backend/scripts backend/tests backend 2>/dev/null || true

# 4. Package markers
touch src/__init__.py src/agent/__init__.py src/tools/__init__.py \
      src/models/__init__.py src/prompts/__init__.py src/utils/__init__.py \
      src/api/__init__.py tests/__init__.py

# 5. Placeholders (empty, for optional future cleanup)
touch src/prompts/system_prompts.py src/prompts/agent_prompts.py
touch src/utils/helpers.py src/utils/logger.py src/utils/config.py
touch logs/.gitkeep

# 6. Remove the temp copy of the old graph.py -- you'll replace its
# contents with the 4 split files Claude provided (state.py, memory.py,
# agent.py, executor.py) before this line will make sense to delete.
# Do NOT run this until those 4 files exist:
# git rm src/agent/_graph_old.py

# 7. Mechanical import-path fixes across everything that moved
grep -rl 'from app\.' src tests 2>/dev/null | while read -r f; do
  sed -i \
    -e 's/from app\.agents\.settlement import/from src.tools.settlement import/g' \
    -e 's/from app\.agents\.rag import/from src.tools.rag import/g' \
    -e 's/from app\.llm import/from src.models.llm_client import/g' \
    -e 's/from app\.db import/from src.models.db import/g' \
    "$f"
done

echo "Mechanical moves done."
echo "NOW: create the 4 split files under src/agent/ from Claude's message,"
echo "then run: git rm src/agent/_graph_old.py"
echo "Then manually check tests/test_checkpointer.py's import line (see Claude's note)."