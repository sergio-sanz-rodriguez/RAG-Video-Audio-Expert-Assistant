# Generic
import os
import logging
import numpy as np
from typing import Optional

# From langchain.py
from modules.langchain import CUSTOM_LOG_LEVELS #, logger

class Whisper:
    """
    Agent class responsible for transcribing audio files into text
    using the Whisper speech-to-text model.
    """
    def __init__(
        self,
        openai_client,
        model="whisper-1",        
        logger: Optional[logging.Logger] = None,
        #log_level: str="info"
        ):
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
        #self.log_level = log_level if log_level in levels else 'none'
        #self.log_level_num = CUSTOM_LOG_LEVELS[self.log_level]
        self.logger = logger
        self.prefix = "STT"
        self.logger.info(f"{self.prefix}: Speech-to-Text initialized with model {self.model}.")

    
    def set_logger(self, logger):

        """
        Sets the logger instance to be used by this object.

        Args:
            logger (logging.Logger): The logger to assign for logging messages.
        """
        
        self.logger = logger
        
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
                error_msg = f"{self.prefix}: Error: Audio file does not exist at path {audio_file_path}"
                self.logger.error(error_msg)
                return error_msg
            if os.path.getsize(audio_file_path) == 0:
                error_msg = f"{self.prefix}: Error: Audio file is empty"
                self.logger.error(error_msg)
                return error_msg

            # Open audio file and send to OpenAI Whisper model
            with open(audio_file_path, "rb") as audio_file:
                response = self.openai.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file
                )
                # Debug: print the raw transcription response text
                self.logger.debug(f"{self.prefix}: OpenAI API response: {response.text}")
                return response.text

        except Exception as e:
            # Return error message if something goes wrong
            error_msg = f"{self.prefix}: An error occurred: {e}"
            self.logger.error(error_msg)
            return error_msg
