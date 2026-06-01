"""文本向量化服务 - 负责本地密集向量；BM25 稀疏向量由 Milvus 2.6 内置 Function 生成。"""
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingService:
    """文本向量化服务 - 仅负责密集向量。

    稀疏 BM25 不再由应用层维护 vocab / df / N / avg_doc_len，而是交给 Milvus
    collection schema 上的 BM25 Function 根据 text 字段自动生成。
    """

    def __init__(self):
        self._embedder = _create_dense_embedder()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e


# 全进程唯一实例：写入与检索共用同一个本地 dense embedding 模型实例
embedding_service = EmbeddingService()
