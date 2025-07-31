 # Generic
import re
import os
import glob
import numpy as np
import logging
import random
import torch
import subprocess
import requests
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
from typing import Optional
from modules.embeddings import ClaudeEmbeddings
from modules.common import Colors
from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, Style, init

# LangChain
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Sklearn for clustering
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

#Talker and Whisper Agents
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

set_seed(SEED)

CUSTOM_LOG_LEVELS = {
    'none': 0,
    'error': 0.5,
    'info': 1,
    'debug': 2
}

class OllamaService:

    """
    Service class to interact with and manage the Ollama server.

    Attributes:
        url (str): The URL of the Ollama server.
        logger (Logger): Logger instance used for logging.
    """

    def __init__(self, url="http://localhost:11434", logger=None):
        self.url = url
        self.logger = logger or logging.getLogger(__name__)

        """
        Initializes the OllamaService.

        Args:
            url (str): The URL of the Ollama server.
            logger (Logger, optional): Custom logger. If not provided, uses a default logger.
        """

    # Ollama activation functions
    def is_running(self):

        """
        Checks whether the Ollama server is running.

        Returns:
            bool: True if the server is up and running, False otherwise.
        """

        try:
            response = requests.get(self.url)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def activate(self, max_retries=10, delay=1):

        """
        Attempts to start the Ollama server if it is not already running.

        Args:
            max_retries (int): Number of retries to check for server startup.
            delay (int): Delay between retries in seconds.
        """

        if self.is_running():
            self.logger.info("Ollama is already running.")
            return
        
        self.logger.info("Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for _ in range(max_retries):
            if self.is_running():
                self.logger.info("Ollama is now running.")
                return
            time.sleep(delay)
        
        self.logger.error("Ollama failed to start.")

        
class HTMLFormatter(logging.Formatter):

    """
    A custom logging formatter that outputs colored HTML-styled log messages.

    Attributes:
        COLOR_MAP (dict): Maps logging levels to color strings.
    """

    COLOR_MAP = {
        logging.DEBUG: "blue",
        logging.INFO: "forestgreen",
        logging.WARNING: "darkorange",
        logging.ERROR: "red",
        logging.CRITICAL: "darkred",
    }
    
    def format(self, record):

        """
        Formats a log record using HTML styles for log levels.

        Args:
            record (LogRecord): The log record to format.

        Returns:
            str: HTML-formatted log message.
        """

        color = self.COLOR_MAP.get(record.levelno, "black")
        levelname = f'<span style="color:{color}; font-weight:bold;">[{record.levelname}]</span>'
        message = super().format(record)
        # Remove the old levelname to avoid duplication
        message = message.replace(record.levelname, "").strip()
        return f"{levelname} {message}"

class QueueHandler(logging.Handler):

    """
    Custom logging handler that places formatted log records into a queue.

    Attributes:
        log_queue (queue.Queue): The queue to send formatted log messages to.
    """

    def __init__(self, log_queue):

        """
        Initializes the handler with a target queue.

        Args:
            log_queue (queue.Queue): Queue to store formatted log messages.
        """

        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):

        """
        Sends a formatted log record to the queue.

        Args:
            record (LogRecord): Log record to emit.
        """

        self.log_queue.put(self.format(record))

def setup_logger(
    log_queue = None,
    name: Optional[str] = None,
    level: str = "info"
    ) -> logging.Logger:

    """
    Configures and returns a customized logger with HTML formatting and optional queue logging.

    Args:
        log_queue (queue.Queue, optional): Queue for GUI log output.
        name (str, optional): Name of the logger. Uses root logger if None.
        level (str): Logging level as a string (e.g., "info", "debug").

    Returns:
        logging.Logger: Configured logger instance.
    """
        
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Basic root config (no handlers set here because we add them manually)
    logging.basicConfig(level=numeric_level, force=True)

    # Suppress overly verbose logs
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('gradio').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.INFO)
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)


    # Custom formatter and handler
    #formatter = CustomFormatter("%(levelname)s %(message)s")
    #handler = logging.StreamHandler()
    #handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    #logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging

    # Colored formatter
    colored_formatter = HTMLFormatter("%(levelname)s %(message)s")

    # Console handler with colors
    console_handler = logging.StreamHandler()    
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    # Queue handler for GUI logs (without colors)
    if log_queue:
        queue_handler = QueueHandler(log_queue)
        #queue_formatter = logging.Formatter("%(levelname)s %(message)s")
        queue_handler.setFormatter(colored_formatter)
        logger.addHandler(queue_handler)

    logging.captureWarnings(True)

    return logger

# Initialize a shared logger once at module level
logger = setup_logger(__name__, level="INFO")


class OpenAILangChain:
    """
    A wrapper class to build a conversational retrieval-based question-answering system
    using OpenAI's language models and LangChain utilities.

    This class sets up a language model, memory, retriever, and a conversational chain
    that can be used to answer questions based on vectorstore-backed retrieval.
    """

    def __init__(
        self,
        model_name: str = 'gpt-4o-mini',
        mode: str = 'conservative',
        top_docs: int = 25,
        vectorstore=None,
        callbacks=None,
    ):
        """
        Initialize the OpenAILangChain system with model and retrieval settings.

        Args:
            model_name (str): Name of the OpenAI model to use.
            temperature (float): Sampling temperature for the language model.
            top_docs (int): Number of documents to retrieve for each query.
            vectorstore: A LangChain-compatible vectorstore instance.
            callbacks: Optional callbacks for chain execution tracing or logging.
        """
        self.model_name = model_name
        self.top_docs = top_docs
        self.vectorstore = vectorstore
        self.callbacks = callbacks

        # Load the model, memory, retriever, and build the conversational chain.
        self.llm = self.load_llm()
        self.memory = self.load_memory()
        self.retriever = self.load_retriever()
        self.retrieval_chain = self.build_chain()

        mode_presets = {
            "conservative": {"temperature": 0.0, "seed": SEED},
            "creative":     {"temperature": 0.7, "seed": None} 
        }

        if mode in mode_presets:
            self.temperature = mode_presets[mode]["temperature"]
            self.seed = mode_presets[mode]["seed"]
        else:
            # fallback to provided values or defaults
            self.temperature = 0.7
            self.seed = None

    def load_llm(self):
        """
        Load the language model from OpenAI via LangChain.

        Returns:
            ChatOpenAI: Configured OpenAI chat model instance.
        """
        return ChatOpenAI(model_name=self.model_name, temperature=self.temperature, seed=self.SEED)
        

    def load_memory(self):
        """
        Initialize conversational memory to store chat history.

        Returns:
            ConversationBufferMemory: Memory buffer for the chat history.
        """
        return ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    def load_retriever(self):
        """
        Convert the vectorstore into a retriever with the top-k search.

        Returns:
            BaseRetriever: Configured retriever instance.
        """
        return self.vectorstore.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                "k": self.top_docs,
                "score_threshold": 0.75})

    def build_chain(self):
        """
        Construct the conversational retrieval chain using the LLM, retriever, and memory.

        Returns:
            ConversationalRetrievalChain: The main chain to answer user questions.
        """
        if self.callbacks is not None:
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                callbacks=self.callbacks
            )
        else:
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory
            )

    def answer_question(self, question, history):
        """
        Get an answer to a question based on current input and chat history.

        Args:
            question (str): The user's question.
            history (list): List of previous messages in the conversation.

        Returns:
            str: The model-generated answer.
        """
        result = self.retrieval_chain.invoke({
            "input": question,
            "chat_history": history
        })
        return result["answer"]

class CustomLangChain:
    """
    A custom LangChain-based conversational retrieval system that uses a local model
    (e.g., Qwen3 via Ollama) for question answering over a vectorstore-based knowledge base.

    This class enables history-aware retrieval and custom prompt handling for precise responses.
    """

    def __init__(self,
                 openai_client,
                 model_name: str="qwen3:8b",
                 tone: str="formal",
                 mode: str="conservative", # or "creative"
                 top_docs: int=25,
                 vectorstore=None,
                 extra_commands=[],                 
                 log_level: str='info',
                 logger: Optional[logging.Logger] = None
                 ):
        """
        Initialize the CustomLangChain instance.

        Args:
            model_name (str): Name of the local model (e.g., 'qwen3:8b') served by Ollama.
            mode (float): Anwser mode: conservative, or creative. Dependes on sampling temperature for generation.
            top_docs (int): Number of top documents to retrieve.
            vectorstore: A LangChain-compatible vector store.
            extra_commands (list): Extra commands to guide the assistant's behavior in the prompt.
            openai_client (object): OpenAI client object.
            log_level (str): Specifies the logging level
        """
        # Initialize parameters from GUI
        self.model_name = model_name
        self.tone = tone
        self.top_docs = top_docs
        self.vectorstore = vectorstore
        self.extra_commands = extra_commands

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

        # Create the LLMs: the question rewriter and the answerer
        mode_presets = {
            "conservative": {"temperature": 0.0, "top_k": 10, "top_p": 0.3, "do_sample": False, "seed": SEED},
            "creative":     {"temperature": 0.8, "top_k": 70, "top_p": 0.9, "do_sample": True, "seed": None}, 
            "default":      {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "do_sample": True, "seed": None} 
        }
        
        self.openai = openai_client
        self.llm_rewriter = self.load_llm(mode=mode_presets['conservative'])
        self.llm_answerer = self.load_llm(mode=mode_presets[mode])

        # Initialize framework components
        self.logger.info(f"Initializing LangChaing: {model_name}, top {top_docs} docs.")
        self.retriever_prompt = self.get_retriever_prompt()
        self.qa_prompt = self.get_qa_prompt()
        self.retriever = self.load_retriever()
        self.extra_fields = ["source", "doc_type"]
        self.retrieval_chain = self.build_chain()

        # Start Ollama if needed
        self.ollama_service = OllamaService(logger=self.logger)
        self.ollama_service.activate()

        self.logger.info("LangChain sucessfully initialized.")  
 

    def load_llm(self, mode):
        """
        Load the local chat model from Ollama.

        Returns:
            ChatOllama: Configured language model.
        """

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug(f"Loading Ollama model: {self.model_name}.")

        # Load LLM
        if self.model_name.startswith('gpt-'):              
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=mode["temperature"],
                seed=mode["seed"])
        else:
            return ChatOllama(
                model=self.model_name,
                temperature=mode['temperature'],
                top_k=mode['top_k'],
                top_p=mode['top_p'],
                do_sample=mode['do_sample']
                )

    def load_llm_hf(self, mode):
        
        """
        Load a Hugging Face model (e.g., Qwen3-8B) for use in the chatbot using LangChain's HuggingFacePipeline wrapper.

        This method:
        1. Loads the tokenizer and model from Hugging Face Hub.
        2. Creates a text-generation pipeline.
        3. Wraps the pipeline into a LangChain-compatible LLM interface.

        Returns:
            HuggingFacePipeline: A LangChain-compatible wrapper around the Hugging Face generation pipeline.
        """

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug(f"Loading Hugging Face model: {self.model_name}.")

         # Load the tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True) # e.g. "Qwen/Qwen3-8B"

        # Load the model from Hugging Face
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True)

        # Create a Hugging Face text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=mode['temperature'],
            top_k=mode['top_k'],
            top_p=mode['top_p'],
            do_sample=mode['do_sample']
            )

        # Load LLM
        return HuggingFacePipeline(pipeline=pipe)
    
    def load_llm_hf_api(self, mode):

        """
        Load a Hugging Face hosted model via API, not locally.
        
        Returns:
            HuggingFaceEndpoint: LangChain-compatible LLM using remote inference.
        """

        #os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.environ['HF_TOKEN']
        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug(f"Connecting to Hugging Face Hub model: {self.model_name}.")

        # Load LLM
        return HuggingFaceHub(
            repo_id=self.model_name,
            model_kwargs={
                "temperature": mode['temperature'],
                "max_new_tokens": 512,
                "top_k": mode['top_k'],
                "top_p": mode['top_p'],
            }
        )

    def load_memory(self):

        """
        Initialize the conversation memory to retain chat history.

        Returns:
            ConversationBufferMemory: A simple memory buffer.
        """

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug("Loading conversation memory.")

        # Load memory
        return ConversationBufferMemory(
            #memory_key="chat_history",
            #input_key="question",  # match the key used in your chain inputs
            return_messages=True
        )

    def load_retriever(self):

        """
        Convert the vectorstore to a retriever interface.

        Returns:
            BaseRetriever: The retriever object.
        """

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug("Creating retriever from vectorstore.")

        # Load retriever
        #return self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        return self.vectorstore.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                "k": self.top_docs,
                "score_threshold": 0.75})

    def get_retriever_prompt(self):

        """
        Define the prompt used to rephrase questions for retrieval, making them clearer.

        Returns:
            ChatPromptTemplate: A prompt guiding the assistant to rephrase input.
        """

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug("Creating retriever prompt.")

        # Get retriever prompt
        return ChatPromptTemplate.from_messages([
            #("system", "You are a helpful assistant that rephrases questions for retrieval."),
            ("system", "You are a helpful assistant that reformulates user questions to improve retrieval from a document database. Focus on clarity, keywords, and semantic similarity. Do not answer the question, only rephrase it."),
            ("human", "{chat_history}\nQuestion: {input}")
        ])

    def get_qa_prompt(self):

        """
        Define the QA prompt used for generating answers strictly based on context.

        Returns:
            ChatPromptTemplate: Prompt used for answering the final question.
        """
        
        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug("Creating QA prompt.")

        # Get QA prompt
        base_system_prompt = (
            "You are an assistant answering questions based ONLY on the provided context. Follow these rules:\n"
           f"1. IMPORTANT: You must answer using a {self.tone.upper()} tone. Use expressions, style, and word choice — even emojis — to make the tone unmistakable. This is a hard constraint.\n"
           f"2. IMPORTANT: For unrelated inputs to the context (e.g., compliments, feedback, general thoughts, greetings such as 'hi', acknowledgments, etc.), please respond BRIEFLY using the same {self.tone.upper()} tone, WITHOUT referencing the context or guessing their intent."
            "3. IMPORTANT: Avoid showing any internal reasoning or step-by-step thought process unless asked. Only return clear and final answers, formated cleanly.\n"
            "4. Do NOT include tags like <think>.\n"
            "5. If the answer is not in the context, answer from your own knowledge (training data) and let the user know saying 'Responding from internal knowledge.'.\n"
            "6. If you are unsure about the answer, say 'I did not understand well. Could you please repeat the question or provide more context?'\n"
            "7. IMPORTANT: Prioritize extracting document names from the provided context when asked about sources of specific information, unless the user explicitly instructs otherwise.\n"
            "8. If no relevant documents are found, acknowledge it and suggest potential sources instead.\n"            
            "9. Give only the final answer as a direct response.\n"           
           "10. If you are asked to answer using your own knowledge from data used for your training, DO it, and IGNORE the provided document sources (field 'context') in the prompt."
           #"11. AVOID giving introductory phrases like 'Based on the provided context'.\n"
        )
    
        # Append rule only if extra_commands is not empty or just whitespace
        if self.extra_commands and any(cmd.strip() for cmd in self.extra_commands):
        #if self.extra_commands and self.extra_commands.strip():
            base_system_prompt += f"\n11. {' '.join(self.extra_commands).strip()}"            

        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
            self.logger.debug(f"Final QA system prompt:\n{base_system_prompt}.")

        return ChatPromptTemplate.from_messages([
            ("system", base_system_prompt),
            ("human", 
            "Chat History (if relevant):\n{chat_history}\n\n"
            "Context from documents:\n{context}\n\n"
            "Question:\n{input}\n\n"
            )
        ])

    @staticmethod
    def format_with_metadata(
        inputs: dict,
        fields: list = None) -> dict:

        """
        Format retrieved documents by prefixing their content with selected metadata fields.

        Args:
            inputs (dict): Must include 'input' (str) and 'context' (List[Document]).
            fields (list): List of metadata fields to include. Default to ["source", "doc_type"].

        Returns:
            dict: With 'input' and formatted 'context'.
        """
        if fields is None:
            fields = ["source", "doc_type"]

        question = inputs["input"]
        docs = inputs["context"]

        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata_lines = [f"[{field.upper()}: {doc.metadata.get(field, 'unknown')}]" for field in fields]
            formatted_text = "\n".join(metadata_lines) + f"\n{doc.page_content}"
            formatted_docs.append(Document(page_content=formatted_text, metadata=doc.metadata))

        return {
            **inputs,  # keep chat_history, input, etc. "input": question,
            "context": formatted_docs}

    def build_chain(self):
        """
        Construct the complete retrieval-augmented QA chain with history-aware retrieval.
        
        1. User asks a question.
        2. create_history_aware_retriever:
            Uses chat history + current question → rewrites the question.
            Retrieves relevant documents from a vector store.
        3. create_stuff_documents_chain:
            Takes the retrieved docs + rewritten question.
            Sends them to the LLM to generate the answer.

        Returns:
            Runnable: A LangChain-compatible retrieval QA pipeline.
        """

        # Creates a retriever aware of previous chat history.
        # 'history_aware' is an object that stores: 
        # 1. The LLM used to rephrase queries.
        # 2. The vector search engine
        # 3. A prompt template for rewriting the query
        retrieved_documents = create_history_aware_retriever(
            llm=self.llm_rewriter,              # LLM to rephrase the user's question based on chat history
            retriever=self.retriever,           # A document retriever from Chroma. It is used to fetch relevant documents.
            prompt=self.retriever_prompt        # A prompt template that tells the LLM how to rewrite the user's question
        )

        # QA chain using retrieved documents.
        # 'document_chain' is an object that:
        # 1. Takes in retrieved documents + rephrased question
        # 2. Formats them into a prompt using self.qa_prompt
        # 3. Sends that prompt to the LLM
        # 4. Returns the LLM's answer.
        document_chain = create_stuff_documents_chain(
            llm=self.llm_answerer,              # LLM to generate the final answer
            prompt=self.qa_prompt               # A prompt template that combines the retrieved documents and question into a full prompt for the LLM
        )
        
        # Evaluate log level creating a function and connecting it into the pipeline
        def debug_log_input(x):
            if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
                self.logger.debug("Inputs to final QA chain:")
                self.logger.debug(f"{x}")
                self.logger.debug(f"Keys received: {list(x.keys())}")
            return x

        pipeline = (
             # Takes the input query, the generated retrieved documents + history-based rephrased question, and chat history
             # This output is then injected into the ChatPromptTemplate in get_qa_prompt()
            RunnableMap({
                "input": lambda x: x["input"],                          # user question
                "context": retrieved_documents,                         # retrieved documents
                "chat_history": lambda x: x.get("chat_history", "")     # chat turns
                })

            # Adds extra metadata such as source in the case that the user asks for sources
            | RunnableLambda(lambda x: self.format_with_metadata(x, fields=self.extra_fields))      

            # Debug mode
            | RunnableLambda(debug_log_input)

            # Generates the final answer from the retrieved documents + history-based rephrased question
            | document_chain                                                                        
        )

        return pipeline

    def answer_question(self, question, history):
        """
        Answer a question based on current input and full chat history.

        Args:
            question (str): The user's question.
            history (list): List of previous conversation messages.

        Returns:
            str: Cleaned model-generated answer.
        """
        
        # Evaluate log level
        if self.log_level_num >= CUSTOM_LOG_LEVELS['info']:
            self.logger.info("Answering question...")

        try:
            inputs = {"input": question, "chat_history": history}
            result = self.retrieval_chain.invoke(inputs)

            if self.log_level_num >= CUSTOM_LOG_LEVELS['debug']:
                rewritten_query = self.history_aware.question_generator.invoke({
                    "question": question,
                    "chat_history": history
                })
                retrieved_docs = self.retriever.invoke(rewritten_query)
                for i, doc in enumerate(retrieved_docs):
                    self.logger.debug(f"Chunk {i+1} | Source: {doc.metadata.get('source', 'N/A')}.")
                    self.logger.debug(doc.page_content[:50])
            
            if self.log_level_num >= CUSTOM_LOG_LEVELS['info']:
                self.logger.info("Question answered!")
            
            return self.clean_answer(result)

        except Exception as e:
            self.logger.error(f"Error answering question: {e}")        

    @staticmethod
    def clean_answer(answer):
        """
        Clean the model's output by removing any <think> tags (if generated).

        Args:
            answer (str): Raw generated output.

        Returns:
            str: Cleaned answer string.
        """

        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    
    @staticmethod
    def stream_text(text, chunk_size=10):
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]


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
        self.embeddings = None #self.load_embeddings()

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

        self.logger.info(f"Set {self.device} device.")

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
                self.logger.error(f"Unknown HuggingFace model: {self.embedding_model}")
                raise ValueError(f"Unsupported HuggingFace embedding model: {self.embedding_model}")
            self.logger.info(f"Using Hugging Face model: {model_name}.")
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": self.device})

        elif "openai" in self.embedding_model:
            load_dotenv(override=True)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
            os.environ['OPENAI_API_KEY'] = api_key
            model_map = {
                "openai_default": "text-embedding-ada-002",
                "openai_small": "text-embedding-3-small",
                "openai_large": "text-embedding-3-large"
            }
            model = model_map.get(self.embedding_model)
            if not model:
                self.logger.error(f"Unknown OpenAI model: {self.embedding_model}")
                raise ValueError(f"Unsupported OpenAI embedding model: {self.embedding_model}")
            self.logger.info(f"Using OpenAI model: {model}.")
            return OpenAIEmbeddings(model=model)

        elif self.embedding_model == "claude":
            if ClaudeEmbeddings is None:
                raise ImportError("ClaudeEmbeddings class is not available.")
            self.logger.info("Using Claude embeddings.")
            return ClaudeEmbeddings(model="claude-3-sonnet-20240229")

        else:
            self.logger.error(f"Unknown embedding model: {self.embedding_model}.")
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
        

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
            self.logger.error(f"Folder does not exist: {document_path}")
            raise FileNotFoundError(f"Folder does not exist: {document_path}")

        self.logger.info("Splitting documents into chunks...")

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
                    self.logger.warning(f"Failed to load documents from {loader}: {e}.")
                    continue

                for doc in docs:
                    doc_counter += 1
                    if self.clean_text_flag:
                        doc.page_content = self.clean_text(doc.page_content)
                    doc.metadata["doc_type"] = doc_type
                    documents.append(doc)

                    if doc_counter % 1000 == 0:
                        self.logger.info(f"Processed {doc_counter} documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        self.chunks, self.documents = chunks, documents

        self.logger.info(f"Total number of chunks: {len(chunks)}.")
        self.logger.info(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}.")

        #return chunks, documents
        return self

    def create_vectorstore(
        self,
        vector_db_name: str = 'vector_db',
        embedding_model: str = "huggingface_1024",    
    ):

        """
        Compute and store embeddings for document chunks in a persistent Chroma vectorstore.

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
            Chroma: The created vectorstore instance.
        """

        if self.chunks == None:
            self.logger.error("Chunks not found. Please call split_into_chunks() first.")
            raise RuntimeError("Chunks not found. Please call split_into_chunks() first.")

        self.vector_db_name = vector_db_name
        self.embedding_model = embedding_model
        self.embeddings = self.load_embeddings()

        # Delete existing collection if it exists
        if os.path.exists(self.vector_db_name):
            Chroma(persist_directory=self.vector_db_name, embedding_function=self.embeddings).delete_collection()

        self.logger.info("Creating vectorstore...")

        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_db_name
        )

        self.collection = self.vectorstore._collection
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]

        self.logger.info(f"Vectorstore created with {self.collection.count():,} vectors of {len(sample_embedding):,} dimensions.")

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

        self.logger.info("Loading vectorstore...")

        self.vectorstore = Chroma(
            persist_directory=self.vector_db_name,
            embedding_function=self.embeddings
        )

        self.collection = self.vectorstore._collection
        count = self.collection.count()
        sample_embedding = self.collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)

        self.logger.info(f"Vectorstore loaded with {self.vectorstore._collection.count()} documents and {count:,} vectors with {dimensions:,} dimensions.")

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
            raise RuntimeError("Vectorstore is not initialized. Please run 'create_vectorstore()' or 'load_vectorstore()' first.")

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


class Talker:
    """
    Agent class responsible for converting text messages into speech
    and playing the audio output using a specified voice model.
    """
    def __init__(
        self,
        openai_client,
        voice="onyx",        
        log_level="info"):
        """
        Initialize the Talker agent with a default or specified voice.

        Args:
            voice (str): The voice model to use for TTS (default is "onyx").
            log_level (str): Specifies the logging level
        """
    
        # Initialize OpenAI
        self.openai = openai_client

        # Set voice model
        self.voice = voice

        # Initialize logger
        levels = CUSTOM_LOG_LEVELS.keys()
        self.log_level = log_level if log_level in levels else 'none'
        self.log_level_num = CUSTOM_LOG_LEVELS[self.log_level]
        self.logger = logger
        self.logger.info(f"Talker initialized with voice: {self.voice}.")

    def speak(self, message):
        """
        Generate speech audio from the given text message and play it.

        Args:
            message (str): The text message to convert to speech.
        """
        if not isinstance(message, str) or message.strip() == "":
            self.logger.error("Invalid message passed to Talker.speak — must be a non-empty string.")
            return

        try:
            self.logger.debug(f"Generating speech for message: {message}")
        
            # Call OpenAI API to generate TTS audio
            response = self.openai.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=message
            )

            # Convert response content to audio segment
            audio_stream = BytesIO(response.content)
            audio = AudioSegment.from_file(audio_stream, format="mp3")

            # Play the generated audio
            play(audio)
            self.logger.info("Audio playback completed successfully.")
        
        except Exception as e:
            self.logger.error(f"Error in Talker.speak: {e}")


class Whisper:
    """
    Agent class responsible for transcribing audio files into text
    using the Whisper speech-to-text model.
    """
    def __init__(
        self,
        openai_client,
        model="whisper-1",        
        log_level: str="info"):
        """
        Initialize the Whisper agent with the specified transcription model.

        Args:
            model (str): The Whisper model to use for transcription (default is "whisper-1").
            log_level (str): Specifies the logging level
        """
        # Initialize OpenAI
        self.openai = openai_client

        # Set model
        self.model = model

        # Initialize logger
        levels = CUSTOM_LOG_LEVELS.keys()
        self.log_level = log_level if log_level in levels else 'none'
        self.log_level_num = CUSTOM_LOG_LEVELS[self.log_level]
        self.logger = logger
        self.logger.info(f"Whisper initialized with model: {self.model}.")


    def transcribe(self, audio_file_path):
        """
        Transcribe the audio file located at audio_file_path into text.

        Args:
            audio_file_path (str): Path to the audio file to transcribe.

        Returns:
            str: The transcribed text, or an error message if transcription fails.
        """
        try:
            # Validate input path
            if audio_file_path is None or audio_file_path == "":
                return ""
            if not os.path.exists(audio_file_path):
                error_msg = f"Error: Audio file does not exist at path {audio_file_path}"
                self.logger.error(error_msg)
                return error_msg
            if os.path.getsize(audio_file_path) == 0:
                error_msg = "Error: Audio file is empty"
                self.logger.error(error_msg)
                return error_msg

            # Open audio file and send to OpenAI Whisper model
            with open(audio_file_path, "rb") as audio_file:
                response = self.openai.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file
                )
                # Debug: print the raw transcription response text
                self.logger.debug(f"OpenAI API response: {response.text}")
                return response.text

        except Exception as e:
            # Return error message if something goes wrong
            error_msg = f"An error occurred: {e}"
            self.logger.error(error_msg)
            return error_msg
