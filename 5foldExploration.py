
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
import random
import math
import time
from random import choice
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if str(device) == "cuda":
    print("GPU detected.")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
else:
    print("No GPU detected. Using CPU.")
    
def normalizeSequence(s):
    s = s.replace(" ", ""); # suppression de tous les espaces
    return " ".join(s) # ajout d'un seul espace entre chaque caractère

print("Reading lines...")

# Read the file and split into lines
lines = open('long-comp_forlearning.txt', encoding='utf-8').\
    read().strip().split('\n')

# Split every line into pairs and normalize
pairs = [[normalizeSequence(s) for s in l.split('\t')] for l in lines]

# Exemple de ce que l'on veut obtenir dans SRC et TRG :
# SRC = ["AAAAAAAA", "ECECECECEC", ...]
# TRG = ["[A]", "[EC]", ...]
SRC = []
TRG = []
MAX_LENGTH = -1
print("Read %s sentence pairs" % len(pairs))
for pair in pairs:
    SRC.append(pair[0])
    TRG.append(pair[1])
    if len(pair[0]) > MAX_LENGTH:
        MAX_LENGTH = len(pair[0])
    if len(pair[1]) > MAX_LENGTH:
        MAX_LENGTH = len(pair[1])
print("One Example: "+str(choice(pairs)))

# Tokenisation
SRC = [s.split() for s in SRC]
TRG = [t.split() for t in TRG]
# Ajout tokens spéciaux
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
TRG = [[SOS_TOKEN] + t + [EOS_TOKEN] for t in TRG]

# Construction vocabulaire
def build_vocab(sentences):
    vocab = {PAD_TOKEN:0, SOS_TOKEN:1, EOS_TOKEN:2}
    idx = 3
    for s in sentences:
        for tok in s:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab

SRC_VOCAB = build_vocab(SRC)
TRG_VOCAB = build_vocab(TRG)

print("Counted tokens:")
print("SRC: ", len(SRC_VOCAB))
print("TRG: ", len(TRG_VOCAB))
print("Max token count: "+str(int((MAX_LENGTH+1)/2))) # on ne compte pas les espaces

# Construction des dictionnaires inversés
SRC_itos = {i:s for s,i in SRC_VOCAB.items()}
TRG_itos = {i:s for s,i in TRG_VOCAB.items()}

# Récupération de l'index du PAD
PAD_IDX = SRC_VOCAB[PAD_TOKEN]

# Taille du vocabulaire pour les séquences en entrée (nombre total de token uniques dans le corpus).
INPUT_DIM = len(SRC_VOCAB)
# Taille du vocabulaire pour les séquences en sortie (nombre total de token uniques dans le corpus).
OUTPUT_DIM = len(TRG_VOCAB)

# Convertit les tokens de "sentence" en tenseur PyTorch (tableau PyTorch)
def encode(sentence, vocab):
    return torch.tensor([vocab[tok] for tok in sentence])

# Encode les dataset src et trg
class TranslationDataset(Dataset):
    def __init__(self, src, trg, src_vocab, trg_vocab):
        self.src = [encode(s, src_vocab) for s in src]
        self.trg = [encode(t, trg_vocab) for t in trg]
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

# Uniformise la longueur des séquences dans un batch pour qu'ils aient tous la même taille
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, trg_batch

class Encoder(nn.Module):
    # input_dim: Taille du vocabulaire (nombre total de tokens uniques dans le corpus).
    # emb_dim: Dimension des vecteurs d'embedding (taille du vecteur dense pour chaque token).
    # hid_dim : Dimension de l'état caché (hidden size) du LSTM.
    # n_layers: Nombre de couches LSTM empilées.
    # dropout: Taux de dropout entre les couches LSTM
    # bidirectional: Définit si on veut un LSTM unidirectionnel ou bidirectionnel
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()
        # Création de l'embedding: transformation des indices de tokens (entiers) en vecteurs denses (représentations vectorielles)
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        # Création du réseau LSTM: traite les séquences de vecteurs d'embedding pour capturer des dépendances entre tokens
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        # Définition du dropout: désactivation aléatoirement d'une fraction des neurones pendant l'entraînement pour éviter le surapprentissage
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        # Couche de projection pour réduire la dimension quand bidirectionnel
        if bidirectional:
            self.hidden_projection = nn.Linear(hid_dim * 2, hid_dim)
            self.cell_projection = nn.Linear(hid_dim * 2, hid_dim)
    # src: Tenseur d'indices de tokens (entiers) représentant la séquence source.
    def forward(self, src):
        # Passe le tenseur source à travers la couche d'embedding  pour obtenir des vecteurs denses.
        embedding = self.embedding(src)
        # Applique un dropout aux vecteurs denses pour régulariser le modèle et éviter le surapprentissage.
        embedded = self.dropout(embedding)
        # Passe les vecteurs denses dropped à travers le LSTM pour capturer les dépendances dans la séquence.
        _, (hidden, cell) = self.rnn(embedded)
        # on a récupèré Les états finaux du LSTM (après avoir traité toute la séquence) et on les retourne
        # hidden : État caché final
        # cell : État de la cellule finale (mémoire à long terme du LSTM)
        if self.bidirectional :
            # Pour un LSTM bidirectionnel, on a (n_layers * 2, batch_size, hid_dim)
            # On doit combiner les directions et reformater pour le décodeur
            
            # Séparer les directions forward et backward
            # hidden: (n_layers * 2, batch_size, hid_dim) -> (n_layers, 2, batch_size, hid_dim)
            hidden = hidden.view(self.n_layers, 2, hidden.size(1), self.hid_dim)
            cell = cell.view(self.n_layers, 2, cell.size(1), self.hid_dim)
            
            # Concaténer les directions pour chaque couche
            # (n_layers, batch_size, hid_dim * 2)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            
            # Projeter vers la dimension attendue par le décodeur
            # (n_layers, batch_size, hid_dim)
            hidden = self.hidden_projection(hidden)
            cell = self.cell_projection(cell)
        
        return hidden, cell

class Decoder(nn.Module):
    # output_dim: Taille du vocabulaire (nombre total de tokens uniques dans le corpus).
    # emb_dim: Dimension des vecteurs d'embedding (taille du vecteur dense pour chaque token).
    # hid_dim : Dimension de l'état caché (hidden size) du LSTM.
    # n_layers: Nombre de couches LSTM empilées.
    # dropout: Taux de dropout entre les couches LSTM
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # transformation des indices de tokens (entiers) en vecteurs denses (représentations vectorielles)
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        # création d'un réseau LSTM, qui traite les séquences de vecteurs d'embedding pour capturer des dépendances entre tokens
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        # désactive aléatoirement une fraction des neurones pendant l'entraînement pour éviter le surapprentissage
        self.dropout = nn.Dropout(dropout)
        # Projection des représentations cachées du LSTM (de dimension hid_dim) vers l'espace du vocabulaire de sortie (de dimension output_dim).
        self.fc_out = nn.Linear(hid_dim, output_dim)
    # cur_tok: Tenseur d'indices de tokens (entiers) représentant le token courant à décoder.
    # hidden: Tenseur représentant l'état caché du LSTM de l'étape précédente.
    # cell: Tenseur représentant l'état de la cellule (mémoire à long terme) du LSTM de l'étape précédente.
    def forward(self, cur_tok, hidden, cell):
        # Ajoute une dimension à cur_tok pour le mettre au format attendu par le LSTM.
        cur_tok = cur_tok.unsqueeze(1)
        # Convertit le tenseur de tokens (cur_tok) en vecteurs denses via la couche d'embedding.
        embedding = self.embedding(cur_tok)
        # Applique un dropout aux vecteurs denses pour régulariser le modèle et éviter le surapprentissage.
        embedded = self.dropout(embedding)
        # Passe les vecteurs denses dropped à travers le LSTM pour mettre à jour les états cachés et générer une sortie pour le mot courant.
        # output -> Sortie du LSTM pour le token courant.
        # (hidden, cell) -> États mis à jour du LSTM pour l'étape suivante.
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # Suppression de la dimention précédemment ajoutée
        output = output.squeeze(1)
        # Passe le vecteur de sortie du LSTM à travers une couche linéaire pour obtenir les scores/logits pour chaque mot du vocabulaire cible.
        prediction = self.fc_out(output)
        # Retourne les résultats pour l'étape de décodage courante.
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    # Définit comment les données traversent le réseau pendant l'entraînement.
    # src: Tenseur d'indices représentant la séquence source
    # trg: Tenseur d'indices représentant la séquence cible
    # teacher_forcing_ratio: Probabilité d'utiliser le teacher forcing
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # Récupère la taille du batch depuis la dimension 0 de src.
        batch_size = src.size(0)
        # Récupère la longueur de la séquence cible depuis la dimension 1 de trg.
        trg_len = trg.size(1)
        # Récupère la taille du vocabulaire cible depuis la couche linéaire finale du décodeur
        trg_vocab_size = self.decoder.fc_out.out_features

        # Initialise un tenseur outputs pour stocker les prédictions du modèle à chaque étape de décodage.
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # Passe la séquence source à travers l'encodeur pour obtenir les états initiaux (hidden, cell) du décodeur.
        hidden, cell = self.encoder(src)
        # Initialise l'entrée du décodeur avec le premier token de la séquence cible (trg[:,0]).
        cur_tok = trg[:,0]
        # Compléter avec autant de token attendu dans la séquence cible
        for t in range(1, trg_len):
            # Passe l'entrée courante (cur_tok) et les états précédents (hidden, cell) à travers le décodeur pour obtenir :
            # output -> Les scores/logits pour le mot courant.
            # hidden, cell : Les états mis à jour pour l'étape suivante.
            output, hidden, cell = self.decoder(cur_tok, hidden, cell)
            # Stocke la prédiction (output) pour l'étape t dans le tenseur outputs.
            outputs[:,t] = output

            # Décide si on utilise le teacher forcing pour cette étape
            teacher_force = random.random() < teacher_forcing_ratio
            # Trouve l'indice du token avec le score le plus élevé dans output (la prédiction du modèle)
            top1 = output.argmax(1)
            # Décide de l'entrée pour l'étape suivante (t+1) :
            # Si teacher forcing, utilise le mot réel (trg[:,t]).
            # Sinon, utilise le mot prédit (top1).
            cur_tok = trg[:,t] if teacher_force else top1
        # Retourne le tenseur outputs contenant toutes les prédictions du modèle pour la séquence cible.
        return outputs

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import io
from IPython.display import Image, display

def showPlot(title, pointsList, labels):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    cpt = 0
    for points in pointsList:
        plt.plot(points, label=(str(labels[cpt]) if cpt < len(labels) else "??"))
        cpt += 1
    plt.title("Évolution des métriques: "+title)
    plt.xlabel("Époques")
    plt.ylabel("Valeur")
    
    # Légende avec options personnalisées
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Améliorer la mise en forme
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder en mémoire et afficher dans Jupyter
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=75, bbox_inches='tight')
    buffer.seek(0)
    
    # Afficher dans Jupyter
    display(Image(buffer.getvalue()))
    
    plt.close()
    buffer.close()

# Entraîne le modèle pour une époque donnée.
# epoch_num : Numéro de l'époque actuelle (pour l'affichage).
# model : Modèle à entraîner.
# loader : Fournit les données d'entraînement par batches (paires (src, trg)).
# optimizer : Met à jour les poids du modèle en fonction des gradients.
# criterion : Calcule la perte entre les prédictions du modèle et les vraies valeurs.
# clip : Seuil pour le gradient clipping (évite les explosions de gradients).
def train(epoch_num, model, loader, optimizer, criterion, clip):
    # Passe le modèle en mode entraînement.
    model.train()
    # perte totale de l'époque
    epoch_loss = 0
    total_correct = 0  # Nombre total de mots correctement prédits
    total_tokens = 0   # Nombre total de mots (hors padding)
    pos=0
    # Itère sur les batches de données fournis par le DataLoader.
    for src, trg in loader:
        #print ("Training epoch "+str(epoch_num+1)+": "+str(int(10000*pos/len(loader))/100)+"%    ", end='\r')
        # Déplace les tenseurs src et trg sur le GPU (ou CPU).
        src, trg = src.to(device), trg.to(device)
        # Réinitialise les gradients de tous les paramètres du modèle à zéro.
        optimizer.zero_grad()
        # Passe les données (src, trg) à travers le modèle pour obtenir les prédictions.
        output = model(src, trg)
        # Récupère la taille du vocabulaire cible depuis la dernière dimension de output.
        output_dim = output.shape[-1]
        # Prépare les prédictions du modèle pour le calcul de la perte. Ignore le premier mot de la séquence cible (<sos>) et aplatit les dimensions en une seule dimension.
        output = output[:,1:].reshape(-1, output_dim)
        # Prépare les vraies valeurs (trg) pour le calcul de la perte, en les mettant au même format que output.
        trg = trg[:,1:].reshape(-1)
        # Calcule la perte (loss) entre les prédictions du modèle (output) et les vraies valeurs (trg).
        loss = criterion(output, trg)
        # Calcule les gradients de la perte par rapport aux poids du modèle (rétropropagation).
        loss.backward()
        # Applique le gradient clipping pour éviter les explosions de gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Met à jour les poids du modèle en utilisant les gradients calculés.
        optimizer.step()
        # Accumule la perte du batch courant dans epoch_loss.
        epoch_loss += loss.item()

        # Calcul de l'accuracy
        predicted = output.argmax(dim=1)  # Prédictions (indices des mots)
        correct = (predicted == trg)  # Comparaison avec les vraies valeurs

        # Ignorer les paddings (indice 0)
        non_pad_mask = trg != 0
        total_correct += correct[non_pad_mask].sum().item()
        total_tokens += non_pad_mask.sum().item()
        
        pos += 1
        
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    # Retourne la perte moyenne par batch et l'accuracy pour l'époque.
    return epoch_loss / len(loader), accuracy

# calcule la perte moyenne du modèle sur un jeu de données
# model : Modèle à évaluer.
# loader : Fournit les données de validation/test par batches (paires (src, trg)).
# criterion : Calcule la perte entre les prédictions du modèle et les vraies valeurs.
def evaluate(model, loader, criterion):
    # Passe le modèle en mode évaluation.
    model.eval()
    # perte totale pendant l'évaluation.
    epoch_loss = 0
    total_correct = 0  # Nombre total de mots correctement prédits
    total_tokens = 0   # Nombre total de mots (hors padding)
    # Désactive le calcul des gradients pendant l'évaluation.
    with torch.no_grad():
        # Itère sur les batches de données fournis par le DataLoader.
        for src, trg in loader:
            # Déplace les tenseurs src et trg sur le GPU (ou CPU).
            src, trg = src.to(device), trg.to(device)
            # Passe les données (src, trg) à travers le modèle pour obtenir les prédictions (output).
            output = model(src, trg, 0) # pas de teacher forcing
            # Récupère la taille du vocabulaire cible (trg_vocab_size) depuis la dernière dimension de output.
            output_dim = output.shape[-1]
            # Prépare les prédictions du modèle pour le calcul de la perte.
            output = output[:,1:].reshape(-1, output_dim)
            # Prépare les vraies valeurs (trg) pour le calcul de la perte, en les mettant au même format que output.
            trg = trg[:,1:].reshape(-1)
            # Calcule la perte (loss) entre les prédictions du modèle (output) et les vraies valeurs (trg).
            loss = criterion(output, trg)
            # Accumule la perte du batch courant dans epoch_loss.
            epoch_loss += loss.item()

            # Calcul de l'accuracy
            predicted = output.argmax(dim=1)  # Prédictions (indices des mots)
            correct = (predicted == trg)  # Comparaison avec les vraies valeurs

            # Ignorer les paddings (indice 0)
            non_pad_mask = trg != 0
            total_correct += correct[non_pad_mask].sum().item()
            total_tokens += non_pad_mask.sum().item()
            
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    # Retourne la perte moyenne par batch pour l'évaluation.
    return epoch_loss / len(loader), accuracy

# Retourne une liste qui est la moyenne des listes de values
def computeFoldMeans(allValues:list[list[float]]):
    means = [0]*max([len(values) for values in allValues])
    for values in allValues:
        for i in range(len(values)):
            means[i] += values[i]
    for i in range(len(means)):
        means[i] = means[i]/len(allValues)
    return means

def mySort(k):
    return results[k][0][-1]

# Encode les dataset src et trg en tensor numériques (<=> tableaux PyTorch)
dataset = TranslationDataset(SRC, TRG, SRC_VOCAB, TRG_VOCAB)

# 1. On définit la fonction d'entraînement compatible Ray Tune
def train_seq2seq(config, train_loader, val_loader):
    # Configuration du modèle à partir de la config courante
    enc = Encoder(INPUT_DIM, config["ENC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["ENC_DROPOUT"], config["BIDIRECTIONAL"])
    dec = Decoder(OUTPUT_DIM, config["DEC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["DEC_DROPOUT"])
    model = Seq2Seq(enc, dec, device).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    criterion = torch.nn.CrossEntropyLoss()
    
    CLIP = 1
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(config["N_EPOCHS"]):
        train_loss, train_accuracy = train(epoch, model, train_loader, optimizer, criterion, CLIP)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        # si les metriques ne sont pas meilleures que les meilleures des 5 dernières époques, arrêter
        if ((len(train_losses) > 10 and train_loss > min(train_losses[-5:]))
            or (len(val_losses) > 10 and val_loss > min(val_losses[-5:]))
            or (len(train_accuracies) > 10 and train_accuracy < max(train_accuracies[-5:]))
            or (len(val_accuracies) > 10 and val_accuracy < max(val_accuracies[-5:]))):
            break
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    return train_losses, val_losses, train_accuracies, val_accuracies
        

# 2. Définir l'espace de recherche des hyperparamètres
searchSpace = [
    {"Key":0, "BATCH_SIZE":16, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.440819, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.31331, "ENC_EMB_DIM": 128, "HID_DIM": 256, "LR": 0.00023299, "N_LAYERS": 2, "N_EPOCHS": 30},
    #{"Key":13, "BATCH_SIZE":32, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.161197, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.269328, "ENC_EMB_DIM": 64, "HID_DIM": 128, "LR": 0.000818817, "N_LAYERS": 2, "N_EPOCHS": 40},
    #{'Key': 15, 'BATCH_SIZE': 64, 'BIDIRECTIONAL': False, 'DEC_DROPOUT': 0.4523098700525877, 'DEC_EMB_DIM': 32, 'ENC_DROPOUT': 0.12384329060555066, 'ENC_EMB_DIM': 128, 'HID_DIM': 256, 'LR': 0.0027379544875619807, 'N_LAYERS': 3, 'N_EPOCHS': 40},
    #{'Key': 18, 'BATCH_SIZE': 32, 'BIDIRECTIONAL': True, 'DEC_DROPOUT': 0.43325387434248197, 'DEC_EMB_DIM': 128, 'ENC_DROPOUT': 0.42355256223853666, 'ENC_EMB_DIM': 128, 'HID_DIM': 256, 'LR': 0.00022905089776058659, 'N_LAYERS': 2, 'N_EPOCHS': 40},
    #{'Key': 19, 'BATCH_SIZE': 32, 'BIDIRECTIONAL': False, 'DEC_DROPOUT': 0.32843793749563344, 'DEC_EMB_DIM': 32, 'ENC_DROPOUT': 0.21262652633947465, 'ENC_EMB_DIM': 64, 'HID_DIM': 128, 'LR': 0.0011721132802202329, 'N_LAYERS': 2, 'N_EPOCHS': 40},
    #{'Key': 23, 'BATCH_SIZE': 32, 'BIDIRECTIONAL': False, 'DEC_DROPOUT': 0.29433368582798347, 'DEC_EMB_DIM': 128, 'ENC_DROPOUT': 0.43349293690802104, 'ENC_EMB_DIM': 256, 'HID_DIM': 128, 'LR': 0.0035036927184472575, 'N_LAYERS': 2, 'N_EPOCHS': 40},
    #{"Key":666, "BATCH_SIZE":64, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.440819, "DEC_EMB_DIM": 128, "ENC_DROPOUT": 0.31331, "ENC_EMB_DIM": 128, "HID_DIM": 512, "LR": 0.00023299, "N_LAYERS": 2, "N_EPOCHS": 40},
    #{'Key': 668, 'BATCH_SIZE': 16, 'BIDIRECTIONAL': True, 'DEC_DROPOUT': 0.2552553998721213, 'DEC_EMB_DIM': 64, 'ENC_DROPOUT': 0.3657303778805584, 'ENC_EMB_DIM': 256, 'HID_DIM': 128, 'LR': 0.0009012085107005741, 'N_LAYERS': 4, 'N_EPOCHS': 40},
    #{'Key': 670, 'BATCH_SIZE': 32, 'BIDIRECTIONAL': False, 'DEC_DROPOUT': 0.224323490472932, 'DEC_EMB_DIM': 32, 'ENC_DROPOUT': 0.16888275099579236, 'ENC_EMB_DIM': 128, 'HID_DIM': 512, 'LR': 0.0029438767377576295, 'N_LAYERS': 2, 'N_EPOCHS': 40},
    #{'Key': 671, 'BATCH_SIZE': 32, 'BIDIRECTIONAL': False, 'DEC_DROPOUT': 0.31978820050973733, 'DEC_EMB_DIM': 128, 'ENC_DROPOUT': 0.2877104784508834, 'ENC_EMB_DIM': 256, 'HID_DIM': 512, 'LR': 0.0001296001591769794, 'N_LAYERS': 2, 'N_EPOCHS': 40}
    #{"Key":1, "BATCH_SIZE":16, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.440819, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.31331, "ENC_EMB_DIM": 128, "HID_DIM": 256, "LR": 0.00023299, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":2, "BATCH_SIZE":32, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.179608, "DEC_EMB_DIM": 32, "ENC_DROPOUT": 0.389179, "ENC_EMB_DIM": 128, "HID_DIM": 64, "LR": 0.00243355, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":6, "BATCH_SIZE":32, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.161197, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.269328, "ENC_EMB_DIM": 64, "HID_DIM": 128, "LR": 0.000818817, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":7, "BATCH_SIZE":16, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.341104, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.162988, "ENC_EMB_DIM": 64, "HID_DIM": 256, "LR": 0.00164443, "N_LAYERS": 4, "N_EPOCHS": 20},
    #{"Key":9, "BATCH_SIZE":64, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.120318, "DEC_EMB_DIM": 32, "ENC_DROPOUT": 0.370962, "ENC_EMB_DIM": 128, "HID_DIM": 128, "LR": 0.00427946, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":12, "BATCH_SIZE":32, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.179608, "DEC_EMB_DIM": 32, "ENC_DROPOUT": 0.389179, "ENC_EMB_DIM": 128, "HID_DIM": 64, "LR": 0.00243355, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":14, "BATCH_SIZE":16, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.341104, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.162988, "ENC_EMB_DIM": 64, "HID_DIM": 256, "LR": 0.00164443, "N_LAYERS": 4, "N_EPOCHS": 20},
    #{"Key":15, "BATCH_SIZE":64, "BIDIRECTIONAL":True, "DEC_DROPOUT": 0.120318, "DEC_EMB_DIM": 32, "ENC_DROPOUT": 0.370962, "ENC_EMB_DIM": 128, "HID_DIM": 128, "LR": 0.00427946, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":3, "BATCH_SIZE":16, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.169886, "DEC_EMB_DIM": 128, "ENC_DROPOUT": 0.151859, "ENC_EMB_DIM": 64, "HID_DIM": 64, "LR": 0.00159483, "N_LAYERS": 3, "N_EPOCHS": 20},
    #{"Key":4, "BATCH_SIZE":64, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.248212, "DEC_EMB_DIM": 32, "ENC_DROPOUT": 0.392177, "ENC_EMB_DIM": 128, "HID_DIM": 128, "LR": 0.000198061, "N_LAYERS": 2, "N_EPOCHS": 20},
    #{"Key":5, "BATCH_SIZE":64, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.187711, "DEC_EMB_DIM": 64, "ENC_DROPOUT": 0.223163, "ENC_EMB_DIM": 32, "HID_DIM": 64, "LR": 0.000628239, "N_LAYERS": 4, "N_EPOCHS": 20},
    #{"Key":8, "BATCH_SIZE":32, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.109767, "DEC_EMB_DIM": 128, "ENC_DROPOUT": 0.433458, "ENC_EMB_DIM": 32, "HID_DIM": 128, "LR": 0.00603545, "N_LAYERS": 3, "N_EPOCHS": 20},
    #{"Key":10, "BATCH_SIZE":32, "BIDIRECTIONAL":False, "DEC_DROPOUT": 0.162927, "DEC_EMB_DIM": 128, "ENC_DROPOUT": 0.303756, "ENC_EMB_DIM": 32, "HID_DIM": 64, "LR": 0.000352071, "N_LAYERS": 2, "N_EPOCHS": 20}
]

# Compléter l'espace de recherche avec des valeurs aléatoires
keyMax = 0
for c in searchSpace:
    keyMax = c["Key"] if c["Key"] > keyMax else keyMax
for _ in range(150):
    keyMax += 1
    searchSpace.append({
        "Key":keyMax,
        "BATCH_SIZE":choice([16, 32, 64, 128, 256]),
        "BIDIRECTIONAL":choice([True, False]),
        "DEC_DROPOUT": choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        "DEC_EMB_DIM": choice([16, 32, 64, 128, 256]),
        "ENC_DROPOUT": choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        "ENC_EMB_DIM": choice([16, 32, 64, 128, 256]),
        "HID_DIM": choice([16, 32, 64, 128, 256, 512]),
        "LR": choice([0.0096, 0.0004, 0.0002, 0.0012, 0.0001, 0.0019, 0.0031, 0.0004, 0.0018, 0.0073, 0.0001, 0.0069, 0.002, 0.0041, 0.0012, 0.009, 0.0001, 0.0004, 0.0002, 0.0074, 0.0004, 0.0037, 0.0004, 0.0018, 0.0016, 0.0047, 0.0011, 0.0045, 0.0002, 0.0005, 0.0072, 0.001, 0.0067, 0.0015, 0.0004, 0.0004, 0.0001, 0.0026, 0.0007, 0.0001, 0.0011, 0.0025, 0.0082, 0.0049, 0.0003, 0.0039, 0.008, 0.0001, 0.0002, 0.0004]),
        "N_LAYERS": choice([2, 4, 8, 16]),
        "N_EPOCHS": 30
    })

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(kf.split(dataset))

results = {}
# Parcourir toutes les configurations
for config in searchSpace:
    start = time.time()
    # Résultats par fold
    all_train_losses, all_val_losses, all_train_accs, all_val_accs = [], [], [], []
    print ("Configuration", str(config["Key"])+"/"+str(len(searchSpace)-1), "                ")
    # traiter cette configuration en folds
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"=== Fold {fold+1}/{N_SPLITS} ===")
        # Création des sous-datasets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        # Construction des loaders pour charger les données par paquets homogènes
        train_loader = DataLoader(train_subset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False)
        
        # Initialisation d'un nouveau modèle pour chaque fold
        train_losses, val_losses, train_accuracies, val_accuracies = train_seq2seq(config, train_loader, val_loader)
        results[config["Key"]] = (train_losses, val_losses, train_accuracies, val_accuracies)
            
        # Sauvegarder les résultats de ce fold
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accuracies)
        all_val_accs.append(val_accuracies)
    mean_train_losses = computeFoldMeans(all_train_losses)
    mean_val_losses = computeFoldMeans(all_val_losses)
    mean_train_accs = computeFoldMeans(all_train_accs)
    mean_val_accs = computeFoldMeans(all_val_accs)

    results[config["Key"]] = (mean_train_losses, mean_val_losses, mean_train_accs, mean_val_accs)
    elapsed = time.time() - start
    print(f"Temps écoulé: {elapsed/60:.2f} minutes")

    # Rechercher parmis toutes les configurations testées les 5 meilleures
    bestConfigs = []
    for k in results:
        if len(bestConfigs) < 5:
            bestConfigs.append(k)
            bestConfigs.sort(key=mySort)
        else:
            # si le dernier mean_train_losses de key de la session courrante est meilleur (plus petit) que le moins bon (le dernier) des meilleures configs, le prendre en compte 
            if results[k][0][-1] < results[bestConfigs[-1]][0][-1]:
                bestConfigs = bestConfigs[:-1]
                bestConfigs.append(k)
                bestConfigs.sort(key=mySort)
    # Affichage des meilleures configurations
    print(str(len(bestConfigs))+ " meilleures configurations :")
    for k in bestConfigs:
        print(str(next(obj for obj in searchSpace if obj["Key"] == k)) + " | Loss (train): "+str(results[k][0][-1])+"; Loss (val): "+str(results[k][1][-1])+"; Accuracy (train): "+str(results[k][2][-1])+"; Accuracy (val): "+str(results[k][3][-1]))
