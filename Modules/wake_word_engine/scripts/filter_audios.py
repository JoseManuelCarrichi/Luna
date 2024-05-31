import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
import argparse
import os

def generarCoeficientes(sample_rate, order, low_frequency, high_frequency, filter_path):
    # Filtro Pasa banda
    Fss = sample_rate / 2  # Frecuencia de muestreo y Normalización
    N = order  # Filtro de orden 100
    Flow = low_frequency  # Frecuencia de corte baja
    Fhigh = high_frequency  # Frecuencia de corte alta
    Low = Flow / Fss
    High = Fhigh / Fss
    win = np.hamming(N)  # Tipo de ventana
    Coef = firwin(N, [Low, High], pass_zero=False) * win  # Filtro pasa banda
    
    # Guardar los coeficientes en un archivo numpy
    np.save(os.path.join(filter_path, 'CoeficientesFPB.npy'), Coef)  
    
def filtrarAudio(audioPath, filenamePath, filterPath):
    # Cargar coeficientes
    win = np.load(os.path.join(filterPath, 'CoeficientesFPB.npy'))
    #win = np.load('CoeficientesFPB.npy')
    # Cargar audio
    Audio,sample_rate = sf.read(audioPath)
    # Aplicar filtro    
    AudioFiltrado = lfilter(win, 1, Audio)
    sf.write(filenamePath,AudioFiltrado,sample_rate,format='wav')
    

def main(args):
    # Leer los audios de la carpeta
    index = 0
    total = len(os.listdir(args.mainPath))
    for filename in os.listdir(args.mainPath):
        audioPath = os.path.join(args.mainPath, filename)
        Audio, sample_rate = sf.read(audioPath)
        longitud = len(Audio)
        if(longitud == 31744):
            filenamePath = os.path.join(args.savePath, f"{args.label}_{index}.wav" )
            # Filtrar audio
            filtrarAudio(audioPath, filenamePath, args.filterPath)
            print(f"Audio: {index} de {total}")
        else:
            print(f"Audio descartado {filename}")
        index += 1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filtrar audio")
    parser.add_argument('--mainPath', type=str, help="Path del los archivos de audio")
    parser.add_argument('--savePath', type=str, help="Path donde se guardará el audio filtrado")
    parser.add_argument('--filterPath', type=str, help="Path del archivo con los coeficientes del filtro")
    parser.add_argument('--sample_rate', type=int, default=None, help="Frecuencia de muestreo del audio")
    parser.add_argument('--order', type=int, default=100, help="Orden del filtro")
    parser.add_argument('--low_frequency', type=int, default=300, help="Frecuencia de corte baja")
    parser.add_argument('--high_frequency', type=int, default=3800, help="Frecuencia de corte alta")
    parser.add_argument('--label', type=int, default=0, help="Etiqueta del audio")

    args = parser.parse_args()
    # Si no se le pasa la frecuencia de muestreo, se filtra el audio
    if args.sample_rate is None:
        main(args)
    else:
        # Se generan los coeficientes del filtro
        generarCoeficientes(args.sample_rate, args.order, args.low_frequency, args.high_frequency,args.filterPath)
