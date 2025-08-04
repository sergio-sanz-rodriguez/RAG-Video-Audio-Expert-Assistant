# Generic
import re
import os
import glob
import numpy as np
import logging
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from typing import Optional, List
from dotenv import load_dotenv

# LangChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

# From langchain.py
from modules.langchain import CUSTOM_LOG_LEVELS, setup_logger

# Sklearn for clustering
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


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

class VectorEmbedding:

    """
    A class for processing documents into vector embeddings and visualizing them in 2D space.

    This class supports:
    - Loading markdown and PDF documents from directories.
    - Splitting them into text chunks.
    - Computing vector embeddings using a selected model.
    - Storing them in a Chroma vector store.
    - Loading an existing vectorstore.
    - Visualizing results using dimensionality reduction and clustering.
    """

    def __init__(
        self,
        device: str="cpu",
        log_level: str = "info",
        logger: Optional[logging.Logger] = None
    ):
        
        """
        Initialize the VectorEmbedding instance with document path, DB name, and embedding model.

        Args:
            embedding_model (str): Embedding model to use. Options:
                - "huggingface_384"
                - "huggingface_768"
                - "huggingface_1024" (default)
                - "openai_default", "openai_small", "openai_large"
                - "claude"
        """

        self.device = device
        self.folders = None
        self.vector_db_name = None
        self.documents = None
        self.chunks = None
        self.chunk_size = None
        self.chunk_overlap = None
        self.clean_text_flag = None
        self.vectorstore = None
        self.collection = None
        self.embedding_model = None #embedding_model
        self.embeddings = None

        # Initialize logger
        levels = CUSTOM_LOG_LEVELS.keys()
        self.log_level = log_level if log_level in levels else 'none'
        self.log_level_num = CUSTOM_LOG_LEVELS[self.log_level]
        if logger:
            # Use the provided logger, but override level if different
            self.logger = logger
            level_map = lambda custom_level: max(10, min(int(20 / custom_level), 50))
            self.logger.setLevel(level_map(self.log_level_num))
        else:
            # Create logger here if not passed
            self.logger = setup_logger(name=__name__, level=self.log_level.upper())

        self.prefix = "DB"
        self.logger.info(f"{self.prefix}: Set {self.device} device.")

    @staticmethod
    def clean_text(text):
        """
        Clean text by removing common boilerplate elements from research papers and slides.
        
        Args:
            text (str): Input raw text.
            
        Returns:
            str: Cleaned text.
        """

        # Remove page numbers like "Page 1", "page 2 of 10", "p. 3"
        text = re.sub(r'\b(Page|page|p\.)\s+\d+(\s+of\s+\d+)?\b', '', text)

        # Remove common confidentiality disclaimers
        text = re.sub(r'(?i)(confidential|proprietary|do not distribute|internal use only|copyright).*?\n', '', text)

        # Remove headers/footers with author names, journal titles, slide titles
        text = re.sub(r'(?i)(IEEE|ACM|Springer|Elsevier|Slide\s*\d+|presentation title|author:.*|university of .*)', '', text)

        # Remove figure/table captions
        text = re.sub(r'(?i)(figure|fig\.|table)\s*\d+[:.\-]', '', text)

        # Remove references and citations like [1], [12], (Smith et al., 2020)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(([^()]+ et al\.,? \d{4})\)', '', text)

        # Remove superindices and footnote markers (*, †, ‡, §, #, etc.)
        #text = re.sub(r'(?<=\w)[\*\†\‡\§#]+', '', text)

        # Remove standalone bullet points or numbering (e.g. "•", "-", "1.", "a)")
        text = re.sub(r'^\s*[\u2022\-–•]|\b\d+\.\s+|\b[a-z]\)\s+', '', text, flags=re.MULTILINE)

        text=re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*', "", text)

        # Remove citations like [1], (Smith et al., 2020)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(([^()]+ et al\.,? \d{4})\)', '', text)

        # Remove multiple punctuation artifacts (e.g., " ,", ",,,")
        text = re.sub(r'\s+([,.;:])', r'\1', text)
        text = re.sub(r',,+', ',', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def clean_text_(text):

        """
        Clean text by removing common boilerplate elements like page numbers and disclaimers.

        Args:
            text (str): Input raw text.

        Returns:
            str: Cleaned text.
        """

        text = re.sub(r'\bPage\s+\d+\b', '', text)
        text = re.sub(r'(?i)confidential.*?\n', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def load_embeddings(self):

        """
        Load the embedding function based on the chosen model name.

        Returns:
            An instance of a LangChain-compatible embedding function.
        """

        if "huggingface" in self.embedding_model:
            model_map = {
                "huggingface_384": "sentence-transformers/all-MiniLM-L6-v2",
                "huggingface_768": "bert-base-nli-mean-tokens",
                "huggingface_1024": "BAAI/bge-large-en"
            }
            model_name = model_map.get(self.embedding_model)
            if not model_name:
                self.logger.error(f"{self.prefix}: Unknown HuggingFace model {self.embedding_model}")
                raise ValueError(f"{self.prefix}: Unsupported HuggingFace embedding model {self.embedding_model}")
            self.logger.info(f"{self.prefix}: Using Hugging Face model {model_name}.")
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": self.device})

        elif "openai" in self.embedding_model:
            load_dotenv(override=True)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(f"{self.prefix}: Missing OPENAI_API_KEY in environment.")
            os.environ['OPENAI_API_KEY'] = api_key
            model_map = {
                "openai_default": "text-embedding-ada-002",
                "openai_small": "text-embedding-3-small",
                "openai_large": "text-embedding-3-large"
            }
            model = model_map.get(self.embedding_model)
            if not model:
                self.logger.error(f"{self.prefix}: Unknown OpenAI model {self.embedding_model}")
                raise ValueError(f"{self.prefix}: Unsupported OpenAI embedding model {self.embedding_model}")
            self.logger.info(f"{self.prefix}: Using OpenAI model {model}.")
            return OpenAIEmbeddings(model=model)

        elif self.embedding_model == "claude":
            if ClaudeEmbeddings is None:
                raise ImportError(f"{self.prefix}: ClaudeEmbeddings class is not available.")
            self.logger.info(f"{self.prefix}: Using Claude embeddings.")
            return ClaudeEmbeddings(model="claude-3-sonnet-20240229")

        else:
            self.logger.error(f"{self.prefix}: Unknown embedding model {self.embedding_model}.")
            raise ValueError(f"{self.prefix}: Unsupported embedding model {self.embedding_model}")
        

    def split_into_chunks(
        self,
        document_path: str = '',
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
        clean_text_flag: bool = False,
        ):

        """
        Load markdown and PDF documents from subfolders, optionally clean them,
        and split the content into chunks.

        Args:
            document_path (str): Path to the root folder containing subfolders with documents.            
            chunk_size (int): Number of characters per text chunk.
            chunk_overlap (int): Overlap between chunks.
            clean_text_flag (bool): Whether to clean text before chunking.

        Returns:
            tuple:
                - chunks (List[Document]): The text chunks to embed.
                - documents (List[Document]): Original loaded documents.
        """

        if not os.path.isdir(document_path):
            self.logger.error(f"{self.prefix}: Folder does not exist: {document_path}")
            raise FileNotFoundError(f"{self.prefix}: Folder does not exist: {document_path}")

        self.logger.info("Splitting documents into chunks.")

        self.folders = glob.glob(f"{document_path}/*")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.clean_text_flag = clean_text_flag

        documents = []
        doc_counter = 0
        for folder in self.folders:

            doc_type = os.path.basename(folder)
            md_loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
            docx_loader = DirectoryLoader(folder, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)            
            loaders = [md_loader, pdf_loader, docx_loader]
            #for loader in loaders:
            #    for doc in loader.load():
            #        if self.clean_text_flag:
            #            doc.page_content = self.clean_text(doc.page_content)
            #        doc.metadata["doc_type"] = doc_type
            #        documents.append(doc)

            for loader in loaders:
                try:
                    docs = loader.load()
                except Exception as e:
                    self.logger.warning(f"{self.prefix}: Failed to load documents from {loader}: {e}.")
                    continue

                for doc in docs:
                    doc_counter += 1
                    if self.clean_text_flag:
                        doc.page_content = self.clean_text(doc.page_content)
                    doc.metadata["doc_type"] = doc_type
                    documents.append(doc)

                    if doc_counter % 1000 == 0:
                        self.logger.info(f"{self.prefix}: Processed {doc_counter} documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        self.chunks, self.documents = chunks, documents

        self.logger.info(f"{self.prefix}: Total number of chunks: {len(chunks)}.")
        self.logger.info(f"{self.prefix}: Document types found: {set(doc.metadata['doc_type'] for doc in documents)}.")

        #return chunks, documents
        return self

    def create_vectorstore(
        self,
        vector_db_name: str = 'vector_db',
        embedding_model: str = "huggingface_1024",    
    ):

        """
        Compute and store embeddings for document chunks in a persistent Chroma vector database.

        Args:            
            vector_db_name (str): Name/path of the persistent vector database directory.
            embedding_model (str): Embedding model to use. Options:
                - "huggingface_384"
                - "huggingface_768"
                - "huggingface_1024" (default)
                - "openai_default",
                - "openai_small",
                - "openai_large"
                - "claude"

        Returns:
            Chroma: The created vector database instance.
        """

        if self.chunks == None:
            self.logger.error(f"{self.prefix}: Chunks not found. Please call split_into_chunks() first.")
            raise RuntimeError(f"{self.prefix}: Chunks not found. Please call split_into_chunks() first.")

        self.vector_db_name = vector_db_name
        self.embedding_model = embedding_model
        self.embeddings = self.load_embeddings()

        # Delete existing collection if it exists
        if os.path.exists(self.vector_db_name):
            Chroma(persist_directory=self.vector_db_name, embedding_function=self.embeddings).delete_collection()

        self.logger.info(f"{self.prefix}: Creating vector database.")

        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_db_name
        )

        self.collection = self.vectorstore._collection
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]

        self.logger.info(
            f"{self.prefix}: Vector database created: {self.collection.count():,} vectors of {len(sample_embedding):,} dimensions."
)

        return self.vectorstore

    def load_vectorstore(
        self,        
        vector_db_name: str = 'vector_db',
        embedding_model: str = "huggingface_1024",
        ):

        """
        Load an existing vectorstore from disk using the configured embedding function.

        Args:            
            vector_db_name (str): Name/path of the persistent vector database directory.
            embedding_model (str): Embedding model to use. Options:
                - "huggingface_384"
                - "huggingface_768"
                - "huggingface_1024" (default)
                - "openai_default",
                - "openai_small",
                - "openai_large"
                - "claude"
        """

        self.vector_db_name = vector_db_name
        self.embedding_model = embedding_model
        self.embeddings = self.load_embeddings()

        self.logger.info(f"{self.prefix}: Loading vector database.")

        self.vectorstore = Chroma(
            persist_directory=self.vector_db_name,
            embedding_function=self.embeddings
        )

        self.collection = self.vectorstore._collection
        docs = self.vectorstore._collection.count()
        count = self.collection.count()
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)

        self.logger.info(f"{self.prefix}: Vector database loaded: {docs:,} documents, {count:,} vectors, {dimensions:,} dims/vector.")

        return self.vectorstore

    def visualize_2d_cluster(
        self,
        n_clusters=5,
        pt_size=5,        
        cmap='viridis',
        seed=42):

        """
        Perform k-means clustering and t-SNE dimensionality reduction to visualize document embeddings in 2D.

        Args:
            n_clusters (int): Number of clusters for KMeans.
            seed (int): Random seed for reproducibility.
            cmap (str): Color map used for plotting.
        """

        result = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']
        doc_types = [metadata['doc_type'] for metadata in metadatas]

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            #algorithm='elkan',
            max_iter=1000
        )
        labels = kmeans.fit_predict(vectors)

        #clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
        #labels = clusterer.fit_predict(vectors)
        
        tsne = TSNE(n_components=2, random_state=seed)
        reduced = tsne.fit_transform(vectors)

        hover_text = [f"Type: {t}<br>Text: {d[:10]}..." for t, d in zip(doc_types, documents)]

        marker = dict(
            size=pt_size,
            color=labels,
            colorscale=cmap,
            line=dict(width=0.5, color='gray'),
            opacity=0.8
        )

        fig = go.Figure(data=[go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=marker,
            text=hover_text,
            hoverinfo='text'
        )])

        fig.update_layout(
            title="Document Clusters",
            xaxis_title="Vector Dim 1",
            yaxis_title="Vector Dim 2",
            width=900,
            height=800
        )

        fig.show()

    def visualize_2d(
            self,
            figsize=(14, 7),            
            cmap='plasma',
            background='white',
            pt_size=5,
            marker='o',
            legend_fontsize='large',
            legend_loc='upper right',
            seed=42):
        """
        Perform t-SNE dimensionality reduction and visualize document embeddings in 2D.

        Args:
            seed (int): Random seed for reproducibility.
            cmap (str): Colormap name (e.g., 'plasma', 'tab10').
            background (str): 'black' or 'white' background mode.
        """

        if not self.vectorstore:  #or self.collection
            raise RuntimeError(f"{self.prefix}: Vector database is not initialized. Please run 'create_vectorstore()' or 'load_vectorstore()' first.")

        result = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']
        doc_types = [metadata['doc_type'] for metadata in metadatas]
        unique_types = sorted(set(doc_types))

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=seed)
        reduced = tsne.fit_transform(vectors)

        # Color setup
        colormap = cm.get_cmap(cmap, len(unique_types))
        type_to_color = {
            t: to_hex(colormap(i)) for i, t in enumerate(unique_types)
        }

        # Background color config
        if background == 'black':
            bg_color = 'black'
            fg_color = 'white'
            tick_color = 'gray'
            grid_color = 'gray'
        else:
            bg_color = 'white'
            fg_color = 'black'
            tick_color = 'black'
            grid_color = 'lightgray'

        # Plot setup
        #fig = plt.figure(figsize=(7, 7))
        #fig.patch.set_facecolor(bg_color)
        #ax = plt.gca()
        #ax.set_facecolor(bg_color)

        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax.set_facecolor(bg_color)

        for t in unique_types:
            idx = [i for i, label in enumerate(doc_types) if label == t]
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                s=pt_size,
                alpha=1,
                label=t,
                color=type_to_color[t],
                marker=marker
            )

        ax.grid(True, color=grid_color, linestyle='-', alpha=0.5)

        legend = fig.legend(
            title="Category",
            loc=legend_loc,
            facecolor=bg_color,
            edgecolor=fg_color,
            labelcolor=fg_color,
            fontsize=legend_fontsize,
            #layout='constrained',
            #bbox_to_anchor=bbox_to_anchor,
            #borderaxespad=0.
        )
        legend.get_title().set_color(fg_color)
        legend.get_title().set_fontsize('large')

        
        ax.set_title("Document Clusters", color=fg_color, size='x-large', pad=20)
        ax.set_xlabel("Vector Dim 1", color=fg_color)
        ax.set_ylabel("Vector Dim 2", color=fg_color)
        ax.tick_params(axis='x', colors=tick_color)
        ax.tick_params(axis='y', colors=tick_color)
        
        #plt.tight_layout()
        plt.show()