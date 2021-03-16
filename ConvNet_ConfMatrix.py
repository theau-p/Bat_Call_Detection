### CONSTRUCTION OF A CONVOLUTIONAL NEURAL NETWORK FOR IMAGE RECOGNITION ###
# We will teach the algorithm to recognise the signal of a pipistrelle or noise 


# IMPORT AND PREPROCESS THE DATA #

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
        for label in self.LABELS:               
            print(label)
            for f in tqdm(os.listdir(label)):   # tqdm is a progress bar 
                try:                            # Exceptions are used because there may be faulty or simply missing images    
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    # The image is read and converted to grey (pixel values between 0 and 1)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))    # We modify the dimensions of the image to obtain a square image (we must write the two dimensions)
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])   # np.eye(2) allows to display (1,0) or (0,1) if it is a pipistrelle or noise. 
                    if label == self.PIPISTRELLES:
                        self.pipistrelle_count += 1
                    elif label == self.BRUITS:
                        self.bruit_count += 1 
                except Exception as e: 
                    pass

        np.random.shuffle(self.training_data)   # Shuffling the data 
        np.save("training_data.npy", self.training_data)   
        print("Pipistrelles:", self.pipistrelle_count)
        print("Bruits:", self.bruit_count)    
            
if REBUILD_DATA:                               # Allows to manage more easily when changing the code
    pipistrellevbruit = PipistrelleVSBruit()
    pipistrellevbruit.make_training_data() 

training_data = np.load("training_data.npy", allow_pickle=True)



# CONSTRUCTION OF THE CONVOLUTIONAL NEURAL NETWORK 

import torch.nn as nn 
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)    # The input is 1 image, 4 outputs (feature maps), and the kernel is of size 5*5 (We use a window of 5 by 5 pixels to make the convolutions and it moves on the whole image)
        self.conv2 = nn.Conv2d(4, 8, 5)

        self.fc = nn.Linear(3872, 2)  # flattening
         # 2 outputs for pipistrelle or noise (2 classes)

    def forward(self, x):
        x = self.conv1(x)                 # Going through the first layer 
        x = F.max_pool2d(F.relu(x), (2,2))
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x), (2,2))  
        x = x.view(x.shape[0], -1)        # We put the tensor in the form of a line tensor (flatenning)
        x = self.fc(x)                   # As this is our output layer we do not use an activation function  
        return F.softmax(x, dim=1)        # We create a probabilistic distribution

net = Net() 

# MODEL TRAINING

import torch.optim as optim 

optimizer = optim.Adam(net.parameters(), lr = 0.001)  # The gradient descent is done with the optimizer. The learning rate (lr) is specified and the network parameters (weights, bias) are updated. 
loss_function = nn.MSELoss()                          # Loss function : MSE (Mean Squared Error)

X = torch.Tensor([i[0] for i in training_data]).view(-1,100,100)
X = X/255.0         # The pixels are normalized to have values only between 0 and 1. 
y = torch.Tensor([i[1] for i in training_data])

# We now want to separate the data we use for training from the data used for testing accuracy (or validation, hence VAL_PCT)   

VAL_PCT = 0.2                  # We keep 20% of our data for validation 
val_size = int(len(X)*VAL_PCT)
print(val_size)

# We slice our data into different groups 

train_X = X[:-val_size]      # The first 90% of the data is taken for training 
train_y = y[:-val_size]

test_X = X[-val_size:]       # We take the remaining 10% for the test 
test_y = y[-val_size:]

# Iterations 

BATCH_SIZE = 10
EPOCHS = 15 # The training set is passed through the network 15 times  

start = time.time()

for epoch in range(EPOCHS): 
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):         # We iterate over a length len(train_X) with a step of BATCH_SIZE  
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 100, 100)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()      # The gradients are reset to zero to perform the backpropagation at each loop

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)   # Computing the error between the expected value and the predcited one
        loss.backward()                          # Backpropagation
        optimizer.step()                         # Gradient descent (Minimizing the loss function)

    print(f"Epoch: {epoch}. Loss: {loss}")

# COMPUTING THE ACCURACY OF THE ALGORITHM 

correct = 0
total = 0 
real_label = []
predicted_label = []
with torch.no_grad():  # Be careful not to track the gradient
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


# CONFUSION MATRIX TO REPRESENT THE NETWORK RESULTS  

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
