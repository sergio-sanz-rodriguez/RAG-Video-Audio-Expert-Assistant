# Generic
import os
import uuid
import time
import tempfile
import simpleaudio as sa
import logging
import numpy as np
from typing import Optional

# From langchain.py
from modules.langchain import CUSTOM_LOG_LEVELS #, logger

# Talker and Whisper Agents
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

class Talker:
    """
    Agent class responsible for converting text messages into speech
    and playing the audio output using a specified voice model.
    """
    def __init__(
        self,
        openai_client,
        model="tts-1",
        voice="onyx",
        logger: Optional[logging.Logger] = None,     
        #log_level="info"
        ):
        """
        Initialize the Talker agent with a default or specified voice.

        Args:
            voice (str): The voice model to use for TTS (default is "onyx").
            log_level (str): Specifies the logging level
        """
    
        # Initialize OpenAI
        self.openai = openai_client

        # Set model and voice
        self.model = model
        self.voice = voice

        # Initialize logger
        levels = CUSTOM_LOG_LEVELS.keys()
        #self.log_level = log_level if log_level in levels else 'none'
        #self.log_level_num = CUSTOM_LOG_LEVELS[self.log_level]
        self.logger = logger

        self.prefix = "TTS"
        self.logger.info(f"{self.prefix}: Text-to-Speech initialized with model {self.model} and voice {self.voice}.")
    
    def set_logger(self, logger):

        """
        Sets the logger instance to be used by this object.

        Args:
            logger (logging.Logger): The logger to assign for logging messages.
        """
        
        self.logger = logger
        
    def speak(self, message):
        """
        Generate speech audio from the given text message and play it.

        Args:
            message (str): The text message to convert to speech.
        """
        if not isinstance(message, str) or message.strip() == "":
            self.logger.error(f"{self.prefix}: Invalid message passed to Talker.speak — must be a non-empty string.")
            return

        try:
            self.logger.debug(f"{self.prefix}: Generating speech for message: {message}")

            # Call OpenAI TTS and get response as bytes
            response = self.openai.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=message
            )

            # Load directly from memory (MP3)
            audio_bytes = BytesIO(response.content)
            audio = AudioSegment.from_file(audio_bytes, format="mp3")

            # Convert to raw PCM for simpleaudio
            sa.play_buffer(
                audio.raw_data,
                audio.channels,
                audio.sample_width,
                audio.frame_rate)

            self.logger.info(f"{self.prefix}: Audio playback completed successfully.")

        except Exception as e:
            self.logger.error(f"{self.prefix}: Error in Talker.speak: {e}")