"""
    Utility script to create training json file for wakeword.

    There should be two directories. one that has all of the 0 labels
    and one with all the 1 labels
    -------------------------------------------------------------------------
    Script de utilidad para crear el archivo json de entrenamiento para wakeword.
    
    Debe haber dos directorios. uno que tiene todas las etiquetas 0
    y otro con todas las etiquetas 1
"""
import os
import argparse
import json
import random


def main(args):
    # Obtener la lista de archivos en los directorios de etiquetas cero y uno
    zeros = os.listdir(args.zero_label_dir)
    ones = os.listdir(args.one_label_dir)
    percent = args.percent
    data = []
    # Agregar las rutas de los archivos de etiqueta cero al conjunto de datos con etiqueta 0
    for z in zeros:
        data.append({
            "key": os.path.join(args.zero_label_dir, z),
            "label": 0
        })
    # Agregar las rutas de los archivos de etiqueta uno al conjunto de datos con etiqueta 1
    for o in ones:
        data.append({
            "key": os.path.join(args.one_label_dir, o),
            "label": 1
        })
    # Barajar los datos
    random.shuffle(data)
    percent_data = int(len(data)*percent/100) 
    print("Total de datos: ",len(data))
    
    f = open(args.save_json_path +"/"+ "train.json", "w")
    
    with open(args.save_json_path +"/"+ 'train.json','w') as f:
        d = len(data)
        i=0
        while(i<len(data)-percent_data):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
        print("Datos de entrenamiento: ",i)
    
    f = open(args.save_json_path +"/"+ "test.json", "w")

    with open(args.save_json_path +"/"+ 'test.json','w') as f:
        d = len(data)
        i=len(data)-percent_data
        while(i<d):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
        print("Datos de test: ",i-(len(data)-percent_data))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script para crear archivos json de entrenamiento para wakeword")
    parser.add_argument('--zero_label_dir', type=str, default=None, required=True,
                        help='directory of clips with zero labels')
    parser.add_argument('--one_label_dir', type=str, default=None, required=True,
                        help='directory of clips with one labels')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to save json file')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    args = parser.parse_args()

    main(args)
