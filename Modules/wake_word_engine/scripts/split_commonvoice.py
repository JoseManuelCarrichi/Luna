import os
import pandas as pd
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks
'''
El dataset Common Voice es un dataset de Mozilla que contiene datos de voz en varios idiomas.
La version en español cv-corpus-17.0-delta-2024-03-15 contiene un archivo con información sobre los archivos de audio

Internamente contiene las siguientes columnas:
- clip: nombre del archivo de audio
- duration: duración del archivo de audio
'''
def main(args):
     # Leer el archivo CSV que contiene información sobre los archivos de audio separados por una tabulación
    df = pd.read_csv(args.file_name, sep='\t')
    # Imprimir las primeras filas del DataFrame y el tamaño total de los datos
    print(df.head())
    print('total data size:', len(df))

    # Función para dividir y guardar un archivo de audio en fragmentos
    def chunk_and_save(file):

        path = os.path.join(args.data_path, file)
        audio = AudioSegment.from_file(path)
        length = args.seconds * 1000 # Calcular la longitud de cada fragmento en milisegundos
        # Dividir el archivo de audio en fragmentos
        chunks = make_chunks(audio, length)
        names = []
        for i, chunk in enumerate(chunks):
            _name = file.split(".")[0] + ".wav"
            name = "{}_{}".format(i, _name)
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")
        return 
    # Aplicar la función a cada fila (clip) del DataFrame
    df.path.apply(lambda x: chunk_and_save(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to split common voice data into chunks")
    parser.add_argument('--seconds', type=int, default=None,
                        help='if set to None, then will record forever until keyboard interrupt')
    # --data_path: ruta completa a los datos de audio
    parser.add_argument('--data_path', type=str, default=None, required=True,
                        help='full path to data. i.e. /to/path/clips/')
    # --file_name: nombre del archivo csv que contiene la información sobre los archivos de audio 
    parser.add_argument('--file_name', type=str, default=None, required=True,
                        help='common voice file')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='full path to to save data. i.e. /to/path/saved_clips/')

    args = parser.parse_args()

    main(args)
