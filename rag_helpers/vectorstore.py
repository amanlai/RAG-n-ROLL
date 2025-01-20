# standard library
import json
from typing import Iterable, Self, Any
# third-party library
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from snowflake.connector import SnowflakeConnection


class SnowflakeCortexVectorStore(VectorStore):
    def __init__(
        self,
        connection: SnowflakeConnection,
        topic: str,
        embedding: Embeddings,
        dimensions: int,
    ) -> None:
        self.connection = connection
        self.topic = topic
        self.embedding = embedding
        self.dimensions = dimensions
        self.create_db_schema_wh_if_not_exists()
        self.create_table_if_not_exists()

    @classmethod
    def from_texts(
        cls: type[Self],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Iterable[dict] | None = None,
        topic: str | None = None,
        connection: SnowflakeConnection | None = None,
        **kwargs: Any,
    ) -> Self:
        if topic is None:
            raise ValueError("Must provide 'topic' named parameter.")
        if connection is None:
            raise ValueError("Must provide 'connection' named parameter.")
        if metadatas is None:
            metadatas = ({} for _ in texts)
        vector_store = cls(
            topic=topic, embedding=embedding, connection=connection, **kwargs
        )
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return vector_store

    def create_db_schema_wh_if_not_exists(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS IDENTIFIER(%(database)s);",
            params={"database": f"{self.topic}_database"}
        )
        cursor.execute(
            "USE DATABASE IDENTIFIER(%(database)s);",
            params={"database": f"{self.topic}_database"}
        )
        cursor.execute(
            "CREATE SCHEMA IF NOT EXISTS IDENTIFIER(%(schema)s);",
            params={"schema": f"{self.topic}_schema"}
        )
        cursor.execute(
            "USE SCHEMA IDENTIFIER(%(schema)s);",
            params={"schema": f"{self.topic}_schema"}
        )
        cursor.execute(
            """
            CREATE OR REPLACE WAREHOUSE IDENTIFIER(%(warehouse)s) WITH
                WAREHOUSE_SIZE='X-SMALL'
                AUTO_SUSPEND = 120
                AUTO_RESUME = TRUE
                INITIALLY_SUSPENDED=TRUE;
            """,
            params={"warehouse": f"{self.topic}_warehouse"}
        )
        cursor.execute(
            "USE WAREHOUSE IDENTIFIER(%(warehouse)s);",
            params={"warehouse": f"{self.topic}_warehouse"}
        )

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
        embeddings = self.embedding.embed_documents(texts)
        for t, m, e in zip(texts, metadatas, embeddings):
            # https://github.com/b-art-b/langchain-snowpoc/blob/main/langchain_snowpoc/vectorstores.py#L97-L109
            cursor.execute(
                f"""
                MERGE INTO IDENTIFIER(%(topic)s) orig USING (
                    SELECT
                        UUID_STRING() AS UUID,
                        TO_VARCHAR(%(text)s) AS TEXT,
                        PARSE_JSON(%(metadata)s) AS METADATA,
                        {e}::VECTOR(FLOAT, %(dim)s) AS EMBEDDINGS
                    ) new
                ON new.UUID = orig.UUID
                WHEN NOT MATCHED THEN
                    INSERT (UUID, TEXT, METADATA, EMBEDDINGS)
                    VALUES (new.UUID, new.TEXT, new.METADATA, new.EMBEDDINGS);
                """,
                params={
                    "topic": self.topic,
                    "text": t.replace("'", "\\'"),
                    "metadata": json.dumps(m),
                    "dim": self.dimensions,
                }
            )
        cursor.connection.commit()

    def _similarity_search(self, embedding: list[float], k: int) -> list[Document]:
        cursor = self.connection.cursor()
        cursor.execute(
            f"""
            SELECT
                TEXT,
                METADATA,
                VECTOR_COSINE_SIMILARITY(EMBEDDINGS, {embedding}::VECTOR(FLOAT, %(dim)s)) AS SCORE
            FROM IDENTIFIER(%(topic)s)
            ORDER BY SCORE DESC
            LIMIT %(k)s;
            """,
            params={
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

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        embedding = self.embedding.embed_query(query)
        docs_and_scores = self._similarity_search(embedding, k)
        return docs_and_scores
