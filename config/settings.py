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


# --- Azure AI Foundry -----------------------------------------------------
def get_azure_llm(temperature: float | None = None, deployment: str | None = None):
    """Azure AI Foundry에 배포한 모델을 부른다.

    주의: gpt-5.x 계열은 '추론(reasoning) 모델'이라 temperature를 지원하지 않는다.
    (비추론 -chat 변형은 전부 Deprecated 상태라 신규 배포가 불가능하다)
    그래서 temperature는 명시적으로 넘길 때만 전달한다.
    """
    from langchain_openai import AzureChatOpenAI

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    return AzureChatOpenAI(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        api_key=_require("AZURE_OPENAI_API_KEY"),
        azure_deployment=deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-4-mini"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        **kwargs,
    )


def get_azure_embeddings(deployment: str | None = None):
    """Azure AI Foundry에 배포한 임베딩 모델 (text-embedding-3-small, 1536차원).

    로컬 MiniLM(384차원)보다 한국어 품질이 좋다. 단, 인덱스를 만든 모델과
    검색할 때 모델이 같아야 하므로 모델을 바꾸면 인덱스를 새로 만들어야 한다.
    """
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        api_key=_require("AZURE_OPENAI_API_KEY"),
        azure_deployment=deployment or os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )


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
