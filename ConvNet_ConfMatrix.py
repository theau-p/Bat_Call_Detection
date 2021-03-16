### CONSTRUCTION D'UN RESEAU DE NEURONE CONVOLUTIF POUR LA RECONNAISSANCE D'IMAGE ###
# On va apprendre à l'algorithme à reconnaitre le signal d'une pipistrelle ou du bruit 


# IMPORTER ET PREPROCESS LES DONNEES #

import torch
import os
import cv2
import numpy as np 
from tqdm import tqdm 
import time 


REBUILD_DATA = True 

class PipistrelleVSBruit(): 
    IMG_SIZE = 100
    PIPISTRELLES = "BlueBatImages/Bats_unique"
    BRUITS = "BlueBatImages/NoBats"
    LABELS = {PIPISTRELLES: 1, BRUITS: 0}
    training_data = []
    
    pipistrelle_count = 0
    bruit_count = 0

    def make_training_data(self): 
        for label in self.LABELS:               # label parcourt les clés du dictionnaires à savoir PIPISTRELLES et BRUITS
            print(label)
            for f in tqdm(os.listdir(label)):   # tqdm est une barre de progression et for f in os.listdir(label) permet de parcourir (lister) les images (attention f est juste le nom du fichier !!!) du directoire
                try:                            # On utilise des exceptions car il peut y avoir des images défaillantes ou simplement absentes  
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    # On lit l'image et on la convertit en gris (valeurs des pixels entre 0 et 1)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))     # On modifie les dimensions de l'image pour obtenir une image carrée (il faut écrire les deux dimensions)
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])   #Le np.eye(2) est une technique astucieuse pour afficher (1,0) ou (0,1) si c'est une pipistrelle ou du bruit. 
                    if label == self.PIPISTRELLES:
                        self.pipistrelle_count += 1
                    elif label == self.BRUITS:
                        self.bruit_count += 1 
                except Exception as e: 
                    pass

        np.random.shuffle(self.training_data)   # On "mélange" le training set. 
        np.save("training_data.npy", self.training_data)   # Comme la fonction précédente peut prendre du temps, on sauve le fichier pour éviter de refaire les calculs à chaque fois. 
        print("Pipistrelles:", self.pipistrelle_count)
        print("Bruits:", self.bruit_count)    
            
if REBUILD_DATA:                               # Permet de gérer plus simplement lorsqu'on change le code (du style la taille de l'image...)
    pipistrellevbruit = PipistrelleVSBruit()
    pipistrellevbruit.make_training_data() 

training_data = np.load("training_data.npy", allow_pickle=True)



# CONSTRUCTION DU RESEAU DE NEURONE CONVOLUTIF 

import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)    # L'entrée est 1 image, 32 sortie, et le kernel est de taille 5*5 (On utilise une fenêtre de 5 par 5 pixels pour faire les convolutions et elle se déplace sur toute l'image )
        self.conv2 = nn.Conv2d(4, 8, 5)

        self.fc = nn.Linear(3872, 2)  # flattening
         # 2 sorties pour pipistrelle ou bruit (2 classes)

    def forward(self, x):
        x = self.conv1(x)                 # On passe la première couche 
        x = F.max_pool2d(F.relu(x), (2,2))
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), (2,2))  
        x = x.view(x.shape[0], -1)        # On met le tenseur sous forme de tenseur ligne (flatenning)
        x = self.fc(x)                   # Comme c'est notre couche de sortie on n'utilise pas de fonction d'activation 
        return F.softmax(x, dim=1)        # On crée une distribution de proba: on obtient un tenseur de la forme (0.23, 0.77). 

net = Net() 

# ENTRAINEMENT DU MODELE 

import torch.optim as optim 

optimizer = optim.Adam(net.parameters(), lr = 0.001)  # La descente de gradient se fait "automatiquement" avec optimizer. On précise le learning rate (lr) et les paramètres du réseau (poids, biais) sont updatés. 
loss_function = nn.MSELoss()                          # Fonction de coût : MSE: Mean Squared Error 

X = torch.Tensor([i[0] for i in training_data]).view(-1,100,100)
X = X/255.0         # On normalise les pixels pour avoir des valeurs uniquement entre 0 et 1. 
y = torch.Tensor([i[1] for i in training_data])

# On veut désormais séparer les données que l'on utilise pour l'entrainement de celles utilisées pour tester l'accuracy (ou validation d'où VAL_PCT) 

VAL_PCT = 0.2                  # On réserve 20% de nos données pour la validation 
val_size = int(len(X)*VAL_PCT)
print(val_size)

# On slice nos données en différents groupes 

train_X = X[:-val_size]      # On prend pour l'entrainement les 90% premières données
train_y = y[:-val_size]

test_X = X[-val_size:]       # On prend pour le test les 10% restant 
test_y = y[-val_size:]

# Itérations 

BATCH_SIZE = 10
EPOCHS = 15 # On ne propage et rétropropage toutes le training data quinze fois 

start = time.time()

for epoch in range(EPOCHS): 
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):         # On itère sur une longueur len(train_X) avec un pas de BATCH_SIZE 
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 100, 100)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()      # On remet à zéro les gradients pour faire la rétropropagation à chaque boucle 

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)   # On calcule l'erreur entre le résultat attendu et celui obtenu 
        loss.backward()                          # Retropropagation pour loss
        optimizer.step()                         # Descente de gradient (Minimisation de loss)

    print(f"Epoch: {epoch}. Loss: {loss}")

# CALCUL DE LA PRECISION DU RESEAU DE NEURONE (ACCURACY)

correct = 0
total = 0 
real_label = []
predicted_label = []
with torch.no_grad():  # Attention à ne pas traquer le gradient 
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        real_label.append(real_class)
        net_out = net(test_X[i].view(-1,1,100,100))[0]     
        predicted_class = torch.argmax(net_out)
        predicted_label.append(predicted_class)

        if predicted_class == real_class: 
            correct += 1 
        total +=1

end = time.time() 

print("Accuracy: ", round(correct/total, 3))
print("Run time:", end-start, 'sec')


# Matrice de confusion pour représenter les résultats du réseau 

real_label = np.array(real_label)
predicted_label = np.array(predicted_label)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

array = confusion_matrix(real_label, predicted_label)

df_cm = pd.DataFrame(array, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
