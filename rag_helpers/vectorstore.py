# standard library
import json
import os
from typing import Iterable, Self, Any
# third-party library
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import snowflake.connector as connector

SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")


class SnowflakeCortexVectorStore(VectorStore):
    def __init__(
        self,
        topic: str,
        embeddings: Embeddings,
        dimensions: int,
    ):
        connector.paramstyle = "format"
        self.connection = connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            client_session_keep_alive=True,
        )
        self.topic = topic
        self.embeddings = embeddings
        self.dimensions = dimensions
        self.create_table_if_not_exists()

    @classmethod
    def from_texts(
        cls: type[Self],
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Iterable[dict] | None = None,
        topic: str | None = None,
        **kwargs: Any,
    ) -> Self:
        if topic is None:
            raise ValueError("Must provide 'topic' named parameter.")
        if metadatas is None:
            metadatas = ({} for _ in texts)
        vector_store = cls(topic=topic, embeddings=embeddings, **kwargs)
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return vector_store

    def create_table_if_not_exists(self) -> None:
        self.connection.cursor().execute(
            """
            CREATE TABLE IF NOT EXISTS IDENTIFIER(%(topic)s)
            (
                ID INTEGER AUTOINCREMENT,
                UUID STRING DEFAULT UUID_STRING(),
                TEXT VARCHAR,
                METADATA VARCHAR,
                EMBEDDINGS VECTOR(FLOAT, %(dim)s)
            );
            """,
            params={"topic": self.topic, "dim": self.dimensions},
        )

    def add_texts(
        self,
        texts: list[str],
        metadatas: Iterable[dict],
    ) -> list[str]:
        cursor = self.connection.cursor()
        embeddings = self.embeddings.embed_documents(texts)
        for t, m, e in zip(texts, metadatas, embeddings):
            cursor.execute(
                """
                MERGE INTO IDENTIFIER(%(topic)s) orig USING (
                    SELECT
                        UUID_STRING() AS UUID,
                        TO_VARCHAR(%(text)s) AS TEXT,
                        PARSE_JSON(%(metadata)s) AS METADATA,
                        %(embeddings)s::VECTOR(FLOAT, %(dim)s) AS EMBEDDINGS
                    ) new
                ON new.UUID = orig.UUID
                WHEN NOT MATCHED THEN
                    INSERT INTO (UUID, TEXT, METADATA, EMBEDDINGS)
                    VALUES (new.UUID, new.TEXT, new.METADATA, new.EMBEDDINGS);
                """,
                params={
                    "topic": self.topic,
                    "text": t.replace("'", "\\'"),
                    "metadata": json.dumps(m),
                    "embeddings": e,
                    "dim": self.dimensions,
                }
            )
        cursor.connection.commit()

    def _similarity_search(self, embedding: list[float], k: int = 3) -> list[Document]:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT
                TEXT,
                METADATA,
                VECTOR_COSINE_DISTANCE(EMBEDDINGS, %(query)s::VECTOR(FLOAT, %(dim)s)) AS SCORE
            FROM IDENTIFIER(%(topic)s)
            ORDER BY SCORE DESC
            LIMIT %(k)s;
            """,
            params={
                "query": embedding,
                "dim": self.dimensions,
                "k": k,
                "topic": self.topic,
            }
        )
        documents = []
        for t, m, s in cursor:
            metadata = json.loads(m) or {}
            metadata["score"] = s
            doc = Document(t, metadata=metadata)
            documents.append(doc)
        return documents

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        embedding = self.embeddings.embed_query(query)
        docs_and_scores = self._similarity_search(embedding, k)
        return docs_and_scores
