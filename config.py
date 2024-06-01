# vad_config.py
SAMPLE_RATE = 16000
MIN_SILENCE_DURATION_MS = 500

# Wake Word Engine
MODEL_NAME = "wake_up_word_model.pt"
SENSITIVITY = 10
FILTER_PATH = "Modules/wake_word_engine/neuralnet/filter_coefficients.npy"
RECORD_SECONDS = 2  

# Whisper
WHISPER_MODEL = 'small'
LANGUAGE = 'es'

# LLM
MODEL = "phi3:instruct"

# TTS
TTS_MODEL =  'tts_models/multilingual/multi-dataset/xtts_v2'
