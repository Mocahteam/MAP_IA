import os
import argparse

# Ce script permet de transformer le jeu de test dans le format de donnée attendu pour l'approche algorithmique
# --rep indique pour chaque scénario (boucle simple, boucle avec une option, deux boucles en séquence...) combien de paires ont été générés (voir paramètre --rep_test du script buildDataSet.py).

parser = argparse.ArgumentParser(description="Définir les paramètres du jeu de donné.")
parser.add_argument("--rep", type=int, default=10, help="Nombre de répétitions de chaque scénario (défaut: 10)")
parser.add_argument("--src", type=str, default='long-comp_fortesting.txt', help="fichier source contenant les paires du jeu de test (défaut: long-comp_fortesting.txt)")
parser.add_argument("--outputDir", type=str, default='example', help="nom du dossier contenant la sortie (défaut: example)")

args = parser.parse_args()

repetition = args.rep
src = args.src
output = args.outputDir

os.makedirs(output, exist_ok=True)
os.makedirs(output+'/solutions', exist_ok=True)

with open(src, 'r') as file:
    lines = file.readlines()

for start_index in range(0, len(lines), repetition):
    
    case_lines = lines[start_index:start_index+repetition]
    case_num = int(start_index/repetition)+1
    for i, line in enumerate(case_lines):
        example, solution = line.strip().split('\t')
        
        example_filename = f'{output}/{case_num}_{i+1}_.log'
        solution_filename = f'{output}/solutions/{case_num}_{i+1}_.log'

        with open(example_filename, 'w') as example_file:
            example_file.write(example)
        
        with open(solution_filename, 'w') as solution_file:
            solution_file.write(solution)