from whispercpp import Whisper
import time
from config import WHISPER_MODEL, LANGUAGE
#Seleccionar modelo
w = Whisper(WHISPER_MODEL)

def transcribe(file_path, lang=LANGUAGE):
    start_time = time.perf_counter()
    result = w.transcribe(file_path, lang)
    print(result)
    text = w.extract_text(result)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Transcription took {elapsed_time} seconds")
    return ' '.join(text)