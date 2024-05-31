import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter
import os

def calculate_initial_threshold(audio):
    yn = audio**2
    yne = np.log(yn + 1e-10)
    
    yneAux = np.copy(yne)
    yneAux[np.isneginf(yne)] = np.nan
    minimo = np.nanmin(yneAux)
    yne[np.isneginf(yne)] = minimo
    
    aux1 = np.min(yne) + 17
    YnenNoise = yne[np.where(yne <= aux1)[0]]
    
    hyne, _ = np.histogram(YnenNoise, bins=10)
    Ind1, Ind2, Ind3 = np.argsort(hyne)[-3:]
    
    hyne[Ind1] = 0
    hyne[Ind2] = 0
    hyne[Ind3] = 0
    
    Q = ((-((10 - Ind1) + 0.5) + aux1) + (-((10 - Ind2) + 0.5) + aux1) + (-((10 - Ind3) + 0.5) + aux1)) / 3
    audio_max = np.max(yne)
    k3 = (Q - (audio_max*(0.375/2)))
    return k3

def VAD_realtime(indata, silence_threshold):
    yn = indata**2
    yne = np.log(yn + 1e-10)
    mean_yne = np.mean(yne)
    return mean_yne < silence_threshold, mean_yne

def filtrarAudio(audio, sample_rate):
    # Construir la ruta relativa al archivo CoefficientsBPF.npy Y cargarlo
    win = np.load(os.path.join(os.getcwd(),'Modules', 'Band-pass-filter', 'CoefficientsBPF.npy'))
    AudioFiltrado = lfilter(win, 1, audio)
    return AudioFiltrado

def record_audio(sample_rate=16000, min_silence_duration_ms=500):
    print("Escuchando instrucciones...")
    audio = []
    silence_detected = False
    silence_duration = 0
    silence_threshold = None

    def callback(indata, frames, time, status):
        nonlocal silence_detected, silence_duration, audio, silence_threshold
        audio.extend(indata[:, 0])
        current_audio = np.array(audio[-sample_rate:])

        if silence_threshold is None and len(audio) >= sample_rate * 3:
            initial_audio = np.array(audio[:sample_rate * 3])
            initial_audio = filtrarAudio(initial_audio, sample_rate)
            silence_threshold = -16
            #print(f"Umbral de silencio: {silence_threshold}")

        if len(audio) > sample_rate * 3:
            current_audio = filtrarAudio(current_audio, sample_rate)
            is_silence, mean_yne = VAD_realtime(current_audio, silence_threshold)

            if is_silence:
                silence_duration += len(indata) / sample_rate * 1000  # Convert to ms
                if silence_duration > min_silence_duration_ms:
                    silence_detected = True
                    sd.stop()
                    #print("Silencio detectado, finalizando grabación...")
            else:
                silence_duration = 0

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        while not silence_detected:
            sd.sleep(100)  # Check for silence every 100ms

    audio = filtrarAudio(np.array(audio), sample_rate)

    return audio

'''if __name__ == "__main__":
    sample_rate = 16000
    audio_data = record_audio(sample_rate=sample_rate)
    sf.write(os.getcwd() + "\Temp\instructions.wav", audio_data, sample_rate)
    print("Grabación guardada como: 'instructions.wav'")
'''