import os
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks
import soundfile as sf
def main(args):

    def chunk_and_save(file):
         # Cargar el archivo de audio
        audio = AudioSegment.from_file(file)
        # Calcular la longitud de cada fragmento en milisegundos
        length = args.seconds * 1000
        # Dividir el archivo de audio en fragmentos
        chunks = make_chunks(audio, length)
        names = []
        # Iterar sobre los fragmentos y guardarlos como archivos WAV
        for i, chunk in enumerate(chunks):
            # Obtener el nombre del archivo sin la ruta
            _name = file.split("/")[-1] 
            # Crear un nombre único para el fragmento
            name = "{}_{}".format(i, _name)
            # Ruta completa para guardar el fragmento
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")
            # Normalizar la longitud de los segmentos
            audio, sr = sf.read(wav_path)
            audio = audio[:31744]
            sf.write(wav_path,audio,sr)

        return names

    chunk_and_save(args.audio_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to split audio files into chunks")
    parser.add_argument('--seconds', type=int, default=None,
                        help='if set to None, then will record forever until keyboard interrupt')
    parser.add_argument('--audio_file_name', type=str, default=None, required=True,
                        help='name of audio file')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='full path to to save data. i.e. /to/path/saved_clips/')

    args = parser.parse_args()

    main(args)
