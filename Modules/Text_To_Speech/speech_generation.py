from TTS.api import TTS
from playsound import playsound
import os
#from config import TTS_MODEL
# Model
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')

def speak(text_generated):
    tts.tts_to_file(text=text_generated,
                file_path=os.getcwd() + "\Temp\speech.wav",
                speaker="Alison Dietlinde",
                language="es",
                split_sentences=False,
                speed = 1.5,
                emotion= "happiness"
                )
    playsound(os.getcwd() + "\Temp\speech.wav")
    os.remove(os.getcwd() + "\Temp\speak.wav")
    os.remove(os.getcwd() + "\Temp\instructions.wav")