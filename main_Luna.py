# Importar librerías
import pyaudio
import threading
import time
import wave
import torchaudio
import torch
import numpy as np
from threading import Event
import soundfile as sf
from scipy.signal import lfilter
import os
from playsound import playsound

# Importar funciones necesarias
from Modules.VAD.voice_activity_detection import record_audio
from Modules.wake_word_engine.neuralnet.dataset import get_featurizer
from Modules.Transcription.transcription_whisper import transcribe
from Modules.LLM.logic_engine import LunaChat
from Modules.Text_To_Speech.speech_generation import speak

# Importar configuraciones necesarias
from config import *

class Listener:
    def __init__(self, sample_rate=16000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)
        self.paused = Event()
        self.paused.clear()

    def __end__(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception as e:
            print(f"Error closing stream: {e}")

    def listen(self, queue):
        while True:
            if not self.paused.is_set():
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening...\n")

class WakeWordEngine:
    def __init__(self):
        self.listener = Listener(sample_rate=16000, record_seconds=RECORD_SECONDS)
        self.model = torch.jit.load(os.path.join(os.getcwd(),'resources', MODEL_NAME))
        self.model.eval().to('cpu')
        self.featurizer = get_featurizer(sample_rate=16000)
        self.audio_q = list()
        self.win = np.load(os.path.join(os.getcwd(),'Modules', 'Band-pass-filter', 'CoefficientsBPF.npy'))
      

    def save(self, waveforms, fname="wakeword_temp"):
        try:
            wf = wave.open(fname, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b"".join(waveforms))
            wf.close()
            audio, sr = sf.read(fname)
            return fname, audio
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None, None

    def filter_audio(self, audio, sample_rate=16000, fname="wakeword_temp"):
        try:
            AudioFiltrado = lfilter(self.win, 1, audio)
            sf.write(fname, AudioFiltrado, sample_rate, format='wav')
            return fname
        except Exception as e:
            print(f"Error filtering audio: {e}")
            return None

    def predict(self, audio):
        try:
            fname, audio = self.save(audio)
            if fname and audio is not None:
                fname = self.filter_audio(audio=audio, fname=fname)
                waveform, _ = torchaudio.load(fname, normalize=False)
                mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)
                with torch.no_grad():
                    out = self.model(mfcc)
                    pred = torch.round(torch.sigmoid(out))
                    return pred.item()
            return 0
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()

    def pause_listening(self):
        print('Antes de pausar:', self.listener.paused.is_set())
        self.listener.paused.set()
        print("Escucha pausada", self.listener.paused.is_set())

    def resume_listening(self):
        print("Antes de reanudar", self.listener.paused.is_set())
        self.detect_in_row = 0

        if os.path.exists("wakeword_temp.wav"):
            os.remove("wakeword_temp.wav")

        self.audio_q.clear()
        
        # Introducir una breve espera para permitir que cualquier ruido transitorio se disipe
        time.sleep(0.1)

        self.listener.paused.clear()
        print("Escucha reanudada", self.listener.paused.is_set())
    
    def close_listener(self):
        self.listener.__end__()
        print("Listener cerrado.")

class DemoAction:
    def __init__(self, sensitivity=10, wake_word_engine=None):
        self.detect_in_row = 0
        self.sensitivity = sensitivity
        self.wake_word_engine = wake_word_engine

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            print(self.detect_in_row)
            if self.detect_in_row >= self.sensitivity:
                self.wake_word_engine.pause_listening()
                luna_listen_instructions(self.wake_word_engine)
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0
            print('.')

'''
Función que se ejecuta cuando el motor de palabras clave detecta la palabra clave "Luna"
Se encarga de realizar las siguientes acciones:
1. Detener el motor de palabras clave
2. Reproducir un sonido de atención
3. Grabar las instrucciones del usuario
4. Reproducir un sonido de atención para indicar que la grabación ha finalizado
5. Llamar al servicio de transcripción para obtener el texto de las instrucciones
6. Llamar al servicio de procesamiento de lenguaje natural para obtener la respuesta a las instrucciones
7. Reproducir la respuesta obtenida
8. Reiniciar el motor de palabras clave
'''
def luna_listen_instructions(wake_word_engine):
    wake_word_engine.close_listener()
    try:
        try:
            playsound(os.path.join(os.getcwd(),'resources', 'attention_enabled.wav'))
        except Exception as e:
            print(f"Error reproduciendo sonido: {e}")
        print("Luna activado")
        audio_data = record_audio(sample_rate=SAMPLE_RATE)
        sf.write(os.getcwd() + "\Temp\instructions.wav", audio_data, SAMPLE_RATE)
        try:
            playsound(os.path.join(os.getcwd(),'resources', 'attention_disabled.wav'))
        except Exception as e:
            print(f"Error reproduciendo sonido: {e}")

        try:
            instructions = transcribe(os.getcwd() + "\Temp\instructions.wav")
            print(f"\nInstrucciones: {instructions}")
        except Exception as e:
            print(f"Error al transcribir las instrucciones: {e}")

        try:
            response = LunaChat(pregunta=instructions)
            print(f"Respuesta: {response}")
        except Exception as e:
            print(f"Error al generar la respuesta: {e}")

        try:
            speak(response)
        except Exception as e:
            print(f"Error al sintetizar la voz: {e}")
    except Exception as e:
        print(f"Error al procesar la solicitud: {e}")
    # Reinicia la espera activa del motor de palabras clave
    time.sleep(0.2)
    main()


    

def main():
    wakeword_engine = WakeWordEngine()
    action = DemoAction(SENSITIVITY, wake_word_engine=wakeword_engine)
    
    try:
        wakeword_engine.run(action)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Deteniendo el programa...")
        if os.path.exists("wakeword_temp.wav"):
            os.remove("wakeword_temp.wav")
        wakeword_engine.listener.__end__()

if __name__ == "__main__":
    main()