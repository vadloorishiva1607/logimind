"""
Single point of control for the LLM and embeddings provider.
Swapping providers later (OpenAI -> Anthropic or vice versa) means
changing this file only -- nothing else in the codebase should
import openai/anthropic directly.
"""
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


def get_embeddings():
    # text-embedding-3-small -> 1536 dimensions, matches the
    # VECTOR(1536) column in the policy_chunks table.
    return OpenAIEmbeddings(model="text-embedding-3-small")
