# standard library
from typing import Iterable
# third-party library
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict
from snowflake.cortex import embed_text_768, embed_text_1024
from snowflake.snowpark import Session


class SnowflakeCortexEmbeddings(BaseModel, Embeddings):

    session: Session
    model: str = "e5-base-v2"
    dimensions: int = 768
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _embed(self, text: str) -> list[float]:
        text = text.replace("'", "\\'")
        if self.model in {
            "snowflake-arctic-embed-m-v1.5",
            "snowflake-arctic-embed-m",
            "e5-base-v2"
        }:
            response = embed_text_768(
                model=self.model, text=text, session=self.session
            )
        elif self.model in {
            "snowflake-arctic-embed-l-v2.0",
            "nv-embed-qa-4",
        }:
            response = embed_text_1024(
                model=self.model, text=text, session=self.session
            )
        else:
            raise ValueError(f"The model: {self.model} is not supported.")
        return response

    def embed_documents(self, texts: list[str]) -> Iterable[list[float]]:
        return map(self._embed, texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
