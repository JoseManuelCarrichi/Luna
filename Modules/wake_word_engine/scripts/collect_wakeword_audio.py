"""
    Script to collect data for wake word training..

    To record environment sound run set seconds to None. This will
    record indefinitely until ctrl + c

    To record for a set amount of time set seconds to whatever you want

    To record interactively (usually for recording your own wake words N times).
    use --interactive mode
    ---------------------------------------------------------------------------------
    
    Script para recopilar datos para el entrenamiento de la palabra de activación.
    
    Para grabar el sonido ambiental, ajuste los segundos a "None", esto
    grabará indefinidamente hasta presionar ctrl + c
    
    Para grabar durante un periodo de tiempo determiando, establezca los 
    segundos en el valor que desee.
    
    Para grabar interactivamente (Normalmente para grabar tus propias palabras
    de activación N veces) use --interactive mode
"""

import pyaudio
import wave
import argparse
import time
import os

class Listener:

    def __init__(self, args):
        self.chunk = 1024 # Tamaño de cada fragmento de audio a leer
        self.FORMAT = pyaudio.paInt16 # Formato de audio (16 bits)
        self.channels = 1  # Número de canales de audio (mono)
        self.sample_rate = args.sample_rate # Frecuencia de muestreo del audio
        self.record_seconds = args.seconds # Duración de la grabación en segundos
        
        # Inicialización de la instancia PyAudio para la captura de audio
        self.p = pyaudio.PyAudio()
        
        # Apertura de un stream de audio para la captura
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk)

    def save_audio(self, file_name, frames):
        print('saving file to {}'.format(file_name))
        self.stream.stop_stream() # Detiene la captura de audio
        self.stream.close() # Cierra el stream de audio

        self.p.terminate() # Termina la instancia PyAudio

        # Guarda los datos de audio en un archivo WAV
        # abre el archivo en modo 'escritura de bytes' 
        wf = wave.open(file_name, "wb")
        # Configura los canales
        wf.setnchannels(self.channels)
        # configura el formato de muestreo
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        # configura la frecuencia de muestreo
        wf.setframerate(self.sample_rate)
        # Escribe los frames como bytes
        wf.writeframes(b"".join(frames))
        # Cierra el archivo
        wf.close()

def interactive(args):
    index = 0
    try:
        while True:
            listener = Listener(args) # Crea una instancia de la clase Listener
            frames = [] # Inicializa una lista para almacenar los fragmentos de audio
            
            # Mensaje para indicar al usuario que comience a grabar
            input('Presiona Enter para continuar. La grabación será de {} segundos. Presiona Ctrl + C para salir'.format(args.seconds))
            print('Iniciando grabación....')
            # Espera 0.2 segundos para evitar que el micrófono capte el sonido del clic al presionar Enter
            time.sleep(0.2)  
            
            # Bucle para capturar los fragmentos de audio durante el tiempo especificado
            for i in range(int((listener.sample_rate/listener.chunk) * listener.record_seconds)):
                data = listener.stream.read(listener.chunk, exception_on_overflow = False)
                frames.append(data) # Agrega el fragmento de audio a la lista frames
                
            # Guarda el archivo de audio con un nombre único
            save_path = os.path.join(args.interactive_save_path, "{}.wav".format(index))
            listener.save_audio(save_path, frames)
            
            index += 1 # Incrementa el contador de archivos grabados
    except KeyboardInterrupt:
        print('Interrupción por teclado')
    except Exception as e:
        print(str(e))

def generate_file_name(file_path):
    index = 0
    while True:
        file_name = f"{index}.wav"
        if not os.path.exists(os.path.join(file_path, file_name)):
            return file_name
        index += 1

def main(args):
    listener = Listener(args) # Crea una instancia de la clase Listener
    frames = [] # Inicializa una lista para almacenar los fragmentos de audio
    print('Grabando...')
    try:
        while True:
            if listener.record_seconds == None or listener.record_seconds == 0:  # Grabar hasta una interrupción de teclado
                print('Grabando indefinidamente... presiona ctrl + c para cancelar', end="\r")
                data = listener.stream.read(listener.chunk)
                frames.append(data)
            else:
                for i in range(int((listener.sample_rate/listener.chunk) * listener.record_seconds)):
                    data = listener.stream.read(listener.chunk)
                    frames.append(data)
                raise Exception('Grabación terminada')


    except KeyboardInterrupt:
        print('Interrupción por teclado')
    except Exception as e:
        print(str(e))

    print('Grabación finalizada...')
     # Guarda el archivo de audio con un nombre único
    save_path = os.path.join(args.save_path, "{}".format(generate_file_name(args.save_path)))
    listener.save_audio(save_path, frames)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grabaciones de audio para entrenamiento de palabras de activación")
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='the sample_rate to record at')
    parser.add_argument('--seconds', type=int, default=None,
                        help='if set to None, then will record forever until keyboard interrupt')
    parser.add_argument('--save_path', type=str, default=None, required=False,
                        help='full path to save file. i.e. /to/path/sound.wav')
    parser.add_argument('--interactive_save_path', type=str, default=None, required=False,
                        help='directory to save all the interactive 2 second samples. i.e. /to/path/')
    parser.add_argument('--interactive', default=False, action='store_true', required=False,
                        help='sets to interactive mode')
    print(f"parser: {parser.parse_args()}")

    args = parser.parse_args()
    print(f"Args: {args}")
    if args.interactive:
        if args.interactive_save_path is None:
            raise Exception('need to set --interactive_save_path')
        interactive(args)
    else:
        if args.save_path is None:
            raise Exception('need to set --save_path')
        main(args)