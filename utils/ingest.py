# third-party library
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from snowflake.snowpark import Session
# local
from rag_helpers.embeddings import SnowflakeCortexEmbeddings
from rag_helpers.vectorstore import SnowflakeCortexVectorStore

# if os.getenv("LOCAL", "False") == "False":
#     import sys
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


class IngestData:
    def __init__(
        self,
        session: Session,
        topic: str,
        model: str = "e5-base-v2",
        chunk_size: int = 256,
        chunk_overlap: int = 10,
    ) -> None:
        self._topic = topic
        self._model = model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.validate_and_init()
        self._session = session
        self._embeddings = SnowflakeCortexEmbeddings(
            session=self._session,
            model=self._model,
            dimensions=self._dimensions
        )

    def validate_and_init(self) -> None:
        if self._model in {
            "snowflake-arctic-embed-m-v1.5",
            "snowflake-arctic-embed-m",
            "e5-base-v2"
        }:
            self._dimensions = 768
        elif self._model in {
            "snowflake-arctic-embed-l-v2.0",
            "nv-embed-qa-4",
        }:
            self._dimensions = 1024
        else:
            raise ValueError(f"The model: {self._model} is not supported.")
        if not isinstance(self._chunk_size, int):
            raise ValueError("chunk_size must be an integer.")
        if not isinstance(self._chunk_overlap, int):
            raise ValueError("chunk_overlap must be an integer.")
        if not isinstance(self._topic, str):
            raise ValueError("topic must be a string.")

    def load_document(self, filename: str) -> list[Document]:
        loader = Docx2txtLoader(filename)
        documents = loader.load()
        return documents

    def chunk_data(self, filename: str) -> list[Document]:
        documents = self.load_document(filename)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, 
            chunk_overlap=self._chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def build_embeddings(self, filename: str) -> VectorStore:
        chunks = self.chunk_data(filename)
        vector_store = SnowflakeCortexVectorStore.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            topic=self._topic,
            dimensions=self._dimensions,
            connection=self._session.connection
        )
        return vector_store

    def get_vector_store(self) -> VectorStore:
        vector_store = SnowflakeCortexVectorStore(
            connection=self._session.connection,
            topic=self._topic,
            embeddings=self._embeddings,
            dimensions=self._dimensions
        )
        return vector_store
