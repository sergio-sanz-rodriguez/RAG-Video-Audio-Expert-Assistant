import os
import requests
from typing import List
from langchain.embeddings.base import Embeddings


class ClaudeEmbeddings(Embeddings):
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.url = "https://api.anthropic.com/v1/embeddings"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"model": self.model, "input": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

