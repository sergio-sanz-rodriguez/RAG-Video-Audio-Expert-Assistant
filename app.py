# Imports

import os
import re
import sys
import time
import queue
import psutil
import GPUtil
import winreg
import torch
import warnings
import threading
import subprocess
import gradio as gr
import matplotlib.pyplot as plt
from modules.tts_agent import Talker
from modules.stt_agent import Whisper
from modules.embeddings import VectorEmbedding 
from modules.langchain import OllamaService, CustomLangChain, setup_logger
from dotenv import load_dotenv
from openai import OpenAI
from ansi2html import Ansi2HTMLConverter
from collections import deque

# Globals

TONE_OPTIONS = [
    ("😀 Formal", "formal"),
    ("😎 Casual", "casual"),
    ("😊 Friendly", "friendly"),
    ("🥰 Sweet", "sweet & lovely"),
    ("😏 Sarcastic", "sarcastic"),
    ("😜 Snarky", "snarky"),
    ("😤 Impatient", "impatient"),
    ("😒 Condescending", "condescending"),
    ("😡 Disrespectful", "disrespectful"),
]

VOICE_OPTIONS = [
    ("👨‍🦰 Onyx (Male - Deep, calm)", "onyx"),
    ("👨‍🦱 Echo (Male - Crisp, energetic)", "echo"),
    ("👨‍🦲 Alloy (Male - Friendly, upbeat)", "alloy"),
    ("👩‍🦰 Fable (Female - Warm, storytelling)", "fable"),
    ("👩‍🦳 Nova (Female - Clear, bright)", "nova"),
    ("👩‍🦱 Shimmer (Female - Smooth, soft)", "shimmer"),
]

TEMPERATURE_OPTIONS = [
    ("🧠 Precise, deterministic answer", "conservative"),
    ("🎨 Creative, imaginative answer", "creative")
]

def get_abs_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#DB_NAME = 'vector_db_1024'
DB_NAME = get_abs_path('vector_db_1024')
EMBEDDING_MODEL = 'huggingface_1024'
FOLDER_PATH = './'

# Supress known warnings
warnings.filterwarnings(
    "ignore",
    message=r".*encoder_attention_mask.*is deprecated and will be removed in version 4\.55\.0.*",
    category=FutureWarning,
    module=r"torch\.nn\.modules\.module"
)

# App class
class App:

    def __init__(
            self,
            window_size=30, # in seconds
            max_len_history=60, # in seconds
            ):

        """
        Initialize the app environment, including API keys, device setup,
        logging, AI agents, and data structures for performance monitoring.

        Args:
            window_size (int): Number of seconds to keep performance stats.
            max_len_history (int): Max length for CPU/GPU history buffers.
            database_path (str): Path to the vector database folder.
        """

        # Set environment variables for API keys, with default placeholders
        load_dotenv(override=True)
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
        os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.environ['HF_TOKEN']

        #openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Select device: 'cuda' if GPU is available, else 'cpu'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
       
        # Clear GPU cache to free memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Initialize plotting arguments
        self.window_size = window_size
        self.max_len_history = max_len_history

        # Setup logger with a queue for asynchronous logging
        self.log_queue = queue.Queue()
        self.ui_logger = setup_logger(log_queue=self.log_queue, name="ui_logger", level="info")
        self.console_logger = setup_logger(log_queue=None, name="console_logger", level="info")

        # Initialize CPU/GPU usage history deques with maximum length
        self.cpu_history = deque(maxlen=self.max_len_history)
        self.gpu_history = deque(maxlen=self.max_len_history)                

        # Initialize CPU/GPU usage buffers with zeros for smoothing plots
        self.cpu_history = deque([0]*self.window_size, maxlen=self.window_size)
        self.gpu_history = deque([0]*self.window_size, maxlen=self.window_size)

        # Initialize time history deque with negative to zero values for timeline plotting
        self.time_history = deque(range(-self.window_size + 1, 1), maxlen=self.window_size)     

        # Detect available LLMs
        self.model_options = self.detect_llms()

        # Initalization of voice agents and vector database (DB)
        self.initialize_agents_db()

    def initialize_agents_db(self):      
            
        self.console_logger.info("🚀 Initialization of the app started.")

        # Load or create vectorstore for document embeddings
        self.vectorstore = self.load_vectorstore()

        # Initialize OpenAI client and agents for speech tasks
        self.openai_client = OpenAI()
        self.whisper_agent = Whisper(self.openai_client, logger=self.console_logger)        
        self.talker_agent = Talker(self.openai_client, logger=self.console_logger)        
        self.conv = Ansi2HTMLConverter()

        self.console_logger.info("✅ Initialization of the app complete.")

        # Replace the logger instance so that subsequent messages appear in the GUI.
        self.whisper_agent.set_logger(self.ui_logger)
        self.talker_agent.set_logger(self.ui_logger)
          
    def load_vectorstore(self):

        """
        Load existing vectorstore if available; otherwise create it from documents.
        """

        # Create embedding instance for vector operations
        vector_embedding = VectorEmbedding(    
            device=self.device,
            log_level='info',
            logger=self.console_logger
        )

        #if os.path.isdir(DB_NAME):
        if os.path.isdir(DB_NAME) and os.listdir(DB_NAME):
            # Exeture this code if the vectorstore has already been created
            vectorstore = vector_embedding.load_vectorstore(
                vector_db_name = DB_NAME,
                embedding_model = EMBEDDING_MODEL,
                )
        else:
            # Execute this code to generate the vectorstore, the database with the vectorized documents
            vector_embedding.split_into_chunks(
                document_path = FOLDER_PATH,
                chunk_size=1000,
                chunk_overlap=300,
                clean_text_flag=True,
            )
            
            vectorstore = vector_embedding.create_vectorstore(
                vector_db_name = DB_NAME,
                embedding_model = EMBEDDING_MODEL,
            )
        
        return vectorstore
       
    def handle_audio(self, audio_path):
        
        """
        Transcribe audio to text using Whisper agent.
        """
        
        return self.whisper_agent.transcribe(audio_path)

    def handle_speech(self, text, voice):        
        
        """
        Speak text aloud using Talker agent with specified voice.
        """
        
        self.talker_agent.voice = voice  # change voice dynamically
        self.talker_agent.speak(text)
    
    def initialize_chain(self, model, tone, mode, lang_chain_state):

        """
        Initialize or reuse a CustomLangChain instance based on model, tone, and mode.

        Args:
            model (str): Language model name
            tone (str): Tone style
            mode (str): Mode of operation
            lang_chain_state (dict): Current chain state to check against

        Returns:
            dict: Updated lang_chain_state including chain object
        """

        # Build LangChain
        if (
            lang_chain_state is None
            or lang_chain_state["model"] != model
            or lang_chain_state["tone"] != tone
            or lang_chain_state["mode"] != mode
        ):
            chain = CustomLangChain(
                openai_client=self.openai_client,
                model_name=model,
                tone=tone,
                mode=mode,
                top_docs=15,
                vectorstore=self.vectorstore,
                log_level='info',
                logger=self.ui_logger,
            )
            return {"chain": chain, "model": model, "tone": tone, "mode": mode}
        
        return lang_chain_state
    
    def question_fn(self, question, history):

        """
        Append user's question to chat history.
        """

        # Initialize history if None
        history = history or []

        # Append new messages
        history.append({"role": "user", "content": question})

        return history, history, ""
    
    def answer_fn(self, history, model, tone, mode, lang_chain_state):

        """
        Generate answer based on latest user question, history and language chain state.

        Returns updated history, the answer, and updated chain state.
        """

        # History is empty or malformed
        if not history or not isinstance(history, list) or "content" not in history[-1]: 
            return history or [], "Sorry, I couldn't find your question.", lang_chain_state
        question = history[-1]["content"]
        lang_chain_state = self.initialize_chain(model, tone, mode, lang_chain_state)    
        lang_chain = lang_chain_state["chain"]
        answer = lang_chain.answer_question(question, history)
        
        # Append new messages
        history.append({"role": "assistant", "content": answer})

        return history, answer, lang_chain_state
    
    @staticmethod
    def clear_chat():

        """
        Reset the chat history and LangChain state.
        """

        return [], [], "", None # Also reset the LangChain object

    @staticmethod
    def reset_mic_input():

        """
        Reset microphone input UI component.
        """

        mic = gr.Audio(value=None, label="🎙️ Speak", type="filepath", interactive=True, sources=['microphone']) 
        return mic

    @staticmethod
    def toggle_audio(audio_flag):

        """
        Toggle audio playback state and update button label.

        Args:
            audio_flag (bool): Current audio state

        Returns:
            tuple: Updated mic input state, button label, and new audio state
        """

        new_state = not audio_flag
        button_label = "🔈 Enable Voice Assistant" if not new_state else "🔇 Disable Voice Assistant"
        return (
            gr.update(interactive=new_state),  # mic_input
            button_label,
            new_state
        )
    
    def detect_llms(self, include_openai=True):

        """
        Detect local Ollama models and optionally include OpenAI models.
        """
    
        models = []

        # Detect local Ollama models
        ollama_service = OllamaService(logger=self.console_logger)
        ollama_service.activate()

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if result.returncode != 0:
            self.console_logger.error("Error detecting Ollama models.")
        else:
            # Split stdout into lines and parse
            lines = result.stdout.strip().splitlines()
            # Each line is formatted like: "llama2:7b  3.8GB"
            rows = [re.split(r'\s{2,}', line.strip()) for line in lines]
            # Skip header if it exists (assuming first row is a header)
            ollama_models = [row[0] for row in rows if row][1:] if len(rows) > 1 else []
            models.extend([(model, model) for model in ollama_models])

        # Add OpenAI models manually
        if include_openai:
            openai_models = [            
                ("gpt-4o", "gpt-4o"),
                ("gpt-4o-mini", "gpt-4o-mini"),            
            ]
            models.extend(openai_models)

        return models
    
    def stream_logs(self):

        """
        Continuously yield formatted logs from internal queue.
        """

        buffer = ""
        while True:
            try:
                while True:
                    log = self.log_queue.get_nowait()
                    #log = conv.convert(log, full=False)
                    buffer += log + "<br>" #"\n"
            except queue.Empty:
                pass
            #yield f"<pre>{buffer}</pre>"        
            yield f"<pre style='font-family: monospace;'>{buffer}</pre>"
            time.sleep(1)

    @staticmethod
    def is_dark_mode_enabled():

        """
        Detect if Windows OS dark mode is enabled using registry query.
        """

        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0  # 0 = dark mode
        except:
            return False  # default fallback
    
    def get_stats(self):

        """
        Get CPU and GPU usage stats, update history buffers, and generate matplotlib plot.

        Returns:
            matplotlib.figure.Figure: Figure object with usage plots
        """

        fontsize = 14
        cpu = psutil.cpu_percent()
        gpus = GPUtil.getGPUs()
        gpu = gpus[0].load * 100 if gpus else 0

        # FIFO update
        self.cpu_history.append(cpu)
        self.gpu_history.append(gpu)
        self.time_history.append(self.time_history[-1] + 1)

        # Keep same window size for all
        for hist in (self.cpu_history, self.gpu_history, self.time_history):
            if len(hist) > self.window_size:
                del hist[:len(hist) - self.window_size]

        # Set colors based on dark mode setting
        is_dark = self.is_dark_mode_enabled()
        bg_color = "#0b0f19" if is_dark else "#FFFFFF"
        fg_color = "#f0f0f0" if is_dark else "#000000"

        # Create plot figure and configure colors
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor(bg_color)
        ax.grid()
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=fg_color)
        ax.yaxis.label.set_color(fg_color)
        ax.xaxis.label.set_color(fg_color)

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)

        # Plot CPU and GPU usage lines
        ax.plot(self.time_history, self.cpu_history, label="CPU Util.", color="deepskyblue", linewidth=2)
        ax.plot(self.time_history, self.gpu_history, label="GPU Util.", color="magenta", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_ylabel("Percentage [%]", fontsize=fontsize)
        ax.set_xlabel("Time", fontsize=fontsize)

        # Legend styling
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            fontsize=fontsize,
            frameon=False
            )
        frame = legend.get_frame()
        frame.set_facecolor(bg_color)
        frame.set_edgecolor(fg_color)
        for text in legend.get_texts():
            text.set_color(fg_color)

        return fig

    def live_plot(self):

        """
        Generator yielding live updated stats plots every second.
        """

        while True:
            fig = self.get_stats()
            yield fig
            plt.close(fig)
            time.sleep(1)

    
    def run(self):

        """
        Build and launch the Gradio user interface for the assistant.
        
        This method sets up all UI components such as chatbot, input boxes,
        buttons, dropdown selectors for model, tone, voice, and controls 
        microphone input and audio output toggling. It also connects UI events 
        to their respective callback functions.
        """
        
        # Create a Gradio Blocks context with a custom theme and CSS overrides
        with gr.Blocks(
            theme=gr.themes.Origin(),
            css="""
            #log_output {
            height: 360px;
            overflow-y: auto;
            border: 1px solid black;
            font-family: monospace;
            white-space: pre-wrap;
            }

            /* Hide the theme toggle button */
            .gr-theme-toggle,
            .gr-theme-toggle * {
                all: unset !important;
                display: none !important;
                width: 0 !important;
                height: 0 !important;
                overflow: hidden !important;
                pointer-events: none !important;
                visibility: hidden !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            .gr-box > .gr-prose > label:has-text("Plot") {
                display: none !important;
            }
            """) as ui:     

            # Title Markdown at the top of the UI
            gr.Markdown("# 🚀 Video/Audio Tech Expert Assistant")
            
            with gr.Row():

                # Chat display box for conversation history
                chatbot = gr.Chatbot(label=" ", height=500, type="messages")

            with gr.Row():

                # Textbox for user input (typed messages)
                user_input = gr.Textbox(label="📝 Write your message", placeholder="Type your message here and press Enter.")
            
            with gr.Row():

                # Buttons for clearing chat and toggling audio
                clear_btn = gr.Button("🔄 Clear Chat")
                disable_audio_btn = gr.Button("🔈 Enable Voice Assistant")

            with gr.Row():
                
                # Dropdown selectors for model, tone, response mode, and voice
                model_selector = gr.Dropdown(
                    choices=self.model_options,
                    value='qwen3:8b' if 'qwen3:8b' in [name for name, _ in self.model_options] else self.model_options[0][0],
                    label="🤖 Select Language model"
                )

                # Tone selector
                tone_selector = gr.Dropdown(
                    choices=TONE_OPTIONS,  
                    value="formal",
                    label="🎭Select Tone"
                )

                temperature_selector = gr.Dropdown(
                    choices=TEMPERATURE_OPTIONS,
                    value="conservative",
                    label="🗣️ Select Response Mode"
                )

                voice_selector = gr.Dropdown(
                    choices=VOICE_OPTIONS,
                    value="onyx",
                    label="🎤 Assistant Voice"
                )
                
                # Microphone input for speech-to-text
                mic_input = gr.Audio(label="🎙️ Speak", type="filepath", interactive=True, sources=['microphone'])
            
            with gr.Row():

                # HTML element for logging output (console style)       
                log_output = gr.HTML(elem_id="log_output")            

                # Start streaming logs into the log_output element
                ui.load(fn=self.stream_logs, inputs=[], outputs=log_output)

                # Column containing live CPU and GPU utilization plot
                with gr.Column():            
                    plot_output = gr.Plot(label=" ")

                # Start live updating the plot_output element
                ui.load(fn=self.live_plot, inputs=[], outputs=plot_output)
            
            # State holders for conversation, language model chain, answers, audio toggle state
            lang_chain_state = gr.State({
                "chain": None,
                "model": None,
                "tone": None,
                "mode": None,
            })
            history_state = gr.State([])
            answer_state = gr.State([]) #gr.Textbox(visible=False)
            audio_enabled = gr.State(False)
            #theme_toggle = gr.Checkbox(label="Dark Theme", value=True)

            # When user submits text input:
            # 1) Append question to history
            # 2) Generate answer from model
            # 3) Optionally speak answer aloud if audio enabled
            # 4) Reset mic input to ready state
            user_input.submit(
                fn=self.question_fn,
                inputs=[user_input, history_state],
                outputs=[chatbot, history_state, user_input]
            ).then(
                fn=self.answer_fn,
                inputs=[history_state, model_selector, tone_selector, temperature_selector, lang_chain_state],
                outputs=[chatbot, answer_state, lang_chain_state]
            ).then(
                fn=lambda answer, voice, enabled: self.handle_speech(answer, voice) if enabled else None,
                inputs=[answer_state, voice_selector, audio_enabled],
            ).then(
                fn=self.reset_mic_input,
                inputs=None,
                outputs=mic_input
            )

            # When microphone input changes (user speaks), transcribe audio to text
            mic_input.change(
                fn=self.handle_audio,
                inputs=mic_input,
                outputs=user_input
            )
            
            # Clear chat history and states when clear button is clicked
            clear_btn.click(
                fn=self.clear_chat,
                inputs=None,
                outputs=[chatbot, history_state, user_input, lang_chain_state]
            )

            # Toggle audio on/off when disable_audio button clicked
            disable_audio_btn.click(
                fn=self.toggle_audio,
                inputs=[audio_enabled],
                outputs=[mic_input, disable_audio_btn, audio_enabled]
            )

        # Start initialization of the app on a background thread
        #threading.Thread(target=self.initialize, daemon=True).start()

        # Launch the UI in a new browser tab, non-inline, with custom height
        ui.launch(inbrowser=True, inline=False, height=1000)

if __name__=="__main__":
    App().run()

