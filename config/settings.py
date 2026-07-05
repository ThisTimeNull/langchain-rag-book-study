"""
공통 설정 모듈
================
질문별 코드(`code/QNNNN.py`)에서 공통으로 쓰는 LLM·임베딩·Neo4j 연결을 한곳에서 만듭니다.
.env 에서 키를 읽으며, 각 코드 파일은 필요한 헬퍼만 import 하면 됩니다.

예)
    from config.settings import get_openai_llm, get_deepseek_llm, get_embeddings
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val or val.startswith("sk-..."):
        raise RuntimeError(
            f"환경변수 {key} 가 설정되지 않았습니다. "
            f"`cp env.sample .env` 후 .env 에 실제 값을 채워주세요."
        )
    return val


# --- OpenAI ---------------------------------------------------------------
def get_openai_llm(temperature: float = 0.0, model: str | None = None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=_require("OPENAI_API_KEY"),
    )


def get_embeddings(model: str | None = None):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=_require("OPENAI_API_KEY"),
    )


def get_local_embeddings(model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """API 키 없이 실습할 때 쓰는 무료 로컬 임베딩."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model)


# --- DeepSeek -------------------------------------------------------------
def get_deepseek_llm(temperature: float = 0.0, model: str | None = None):
    """DeepSeek 채팅 모델. ChatDeepSeek 우선, 없으면 OpenAI 호환 엔드포인트로 폴백."""
    api_key = _require("DEEPSEEK_API_KEY")
    model = model or os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    try:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(model=model, temperature=temperature, api_key=api_key)
    except ImportError:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, temperature=temperature, api_key=api_key, base_url=base_url
        )


# --- Neo4j (GraphRAG) -----------------------------------------------------
def get_neo4j_graph():
    from langchain_neo4j import Neo4jGraph

    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=_require("NEO4J_PASSWORD"),
    )
