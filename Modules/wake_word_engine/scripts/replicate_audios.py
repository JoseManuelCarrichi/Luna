import os
import argparse
import shutil

def main(args):
    # Listar archivos en el directorio de palabras clave de activación
    ones = os.listdir(args.wakewords_dir)
    # Crear un directorio de destino en el directorio de palabras clave de activación
    dest_dir = os.mkdir(args.wakewords_dir+'subfolder')
    # Listar archivos en el directorio actual
    os.listdir()
     # Iterar sobre cada archivo en el directorio de palabras clave de activación
    for file in ones:
        # Verificar si el archivo es un archivo de audio
        if file.endswith(".wav") or file.endswith(".mp3"):
            # Para cada archivo, hacer n copias especificadas por el usuario
            for i in range(args.copy_number):
                # Copiar el archivo al directorio de destino
                dest_dir = args.copy_destination
                srcFile = os.path.join(args.wakewords_dir, file)
                shutil.copy(srcFile, dest_dir)
                # Renombrar el archivo en el directorio de destino
                dst_file = os.path.join(dest_dir, file)
                new_dst_file = os.path.join(dest_dir, str(i) + "_" + file)
                os.rename(dst_file, new_dst_file)


if __name__ == "__main__":
    # Crear un analizador de argumentos
    parser = argparse.ArgumentParser(description="""
    Script de utilidad para replicar los clips wakeword por n número de veces.
    """
    )
    parser.add_argument('--wakewords_dir', type=str, default=None, required=True,
                        help='directory of clips with wakewords')

    parser.add_argument('--copy_destination', type=str, default=None, required=True,
                        help='directory of the destinations of the wakewords copies')

    parser.add_argument('--copy_number', type=int, default=100, required=False,
                        help='the number of copies you want')

    args = parser.parse_args()

    main(args)