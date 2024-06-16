#!/usr/bin/env python
# coding: utf-8

# ## Car Model identification with bi-linear models (5 points)
# Images of 20 different models of cars.
# 
# Autor: Alejandro Furió Agustí
# 
# - Data loading
#     - Load images and labels in arrays of shape `(Nx3x224x224)` and `(Nx1)`
#     - Converto to `Tensor`, resize and normalize
#     - Organize into `Dataloader` for better trainig loop
# 
# 
# - Model Architecture
#     - Pre-trained Base: Utilize different sizes of ResNet model for feature extraction.
#     - Bilinear Pooling: Combine features with outer product using einops.
#     - Classification Layer: Map pooled features to num_classes using a fully connected layer.
# 
# - Freeze and Unfreeze
#     - Freezing (1st step): Freeze conv models and train only new layers. `20-40 epochs`
#         - epochs: `30`
#         - learnign rate: `1e-4`
#         - StepScheduler `step_size: 10`
#     - Unfreezing (2nd step):  Unfreeze all weights and train again. `50-60 epochs`
#         - epochs: `60`
#         - learnign rate: `1e-7`
#         - ReduceLRonPlateau: `patience=5` 
# 
# ### Results
# - Test Accuracy: 67%
# 
# *To train the model, just run all the cells up to the training loop. Run the last cell to generate plots with training metrics 
# 

# In[1]:


# check GPU
# !nvidia-smi


# In[2]:


# imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, DatasetDict
from einops import rearrange, einsum

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.models as models
from torchvision.models import resnet18
from torchvision import transforms

from sklearn.model_selection import train_test_split


# In[3]:


import yaml
import os
from pprint import pprint
from pathlib import Path

def load_and_pretty_print_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

current_file_dir = Path(os.getcwd()).resolve()
config = load_and_pretty_print_yaml(current_file_dir / 'config.yaml')


BATCH_SIZE = config['batch_size']
CNN_DROPOUT = config['cnn_dropout']
FEAT_DROPOUT = config['feat_dropout']
FREEZE_CONFIG = config['freeze_weights']
FREEZE_CONFIG['lr'] = float(FREEZE_CONFIG['lr'])
UNFREEZE_CONFIG = config['unfreeze_weights']
UNFREEZE_CONFIG['lr'] = float(UNFREEZE_CONFIG['lr'])

pprint(config)


# In[4]:


# internal_model = models.resnet18
# internal_weights = models.ResNet18_Weights.IMAGENET1K_V1

# internal_model = models.resnet34
# internal_weights = models.ResNet34_Weights.IMAGENET1K_V1

internal_model = models.resnet50
internal_weights = models.ResNet50_Weights.IMAGENET1K_V1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


# Load images
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')


# In[ ]:





# In[6]:


transform_fn = internal_weights.transforms()
convert_to_tensor = transforms.ToTensor()

print('Transforms:\n', transform_fn)

def process_image(image):
    image = convert_to_tensor(image)
    # image = rearrange(image, 'h w c -> c h w')
    image = transform_fn(image)
    return image

# transform images
x_train_transformed = list(map(process_image, x_train))
x_test_transformed = list(map(process_image, x_test))

# stack images
x_train_tensor = torch.stack(x_train_transformed) #.to(device)
x_test_tensor = torch.stack(x_test_transformed) #.to(device)

# convert labels to tensor
y_train_tensor = torch.tensor(y_train) #.to(device)
y_test_tensor = torch.tensor(y_test) #.to(device)

y_train_tensor = y_train_tensor - 1
y_test_tensor = y_test_tensor - 1

# split test set into validation and test
x_val_tensor, x_test_tensor, y_val_tensor, y_test_tensor = train_test_split(x_test_tensor, y_test_tensor, test_size=0.6, random_state=42)
# print('Train:', x_train_tensor.shape, y_train_tensor.shape)
# print('Val:', x_val_tensor.shape, y_val_tensor.shape)
# print('Test:', x_test_tensor.shape, y_test_tensor.shape)


# In[8]:


# TensorDataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)
val_data = TensorDataset(x_val_tensor, y_val_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# DataLoader
batch_size = BATCH_SIZE
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


# ### Model

# In[9]:


class BiLinearModel(nn.Module):
    def __init__(self, num_classes, internal_model=models.resnet50, internal_weights=models.ResNet50_Weights.IMAGENET1K_V1, cnn_dropout=0.0, feat_dropout=0.0):
        super(BiLinearModel, self).__init__()
        
        # Load internal model
        self.cnn1 = internal_model(weights=internal_weights)
        self.cnn2 = internal_model(weights=internal_weights)
        self.feature_size = self.cnn1.fc.in_features
        # print('Feature size:', self.feature_size)
        
        # remove last layers
        self.cnn1 = nn.Sequential(*list(self.cnn1.children())[:-2])
        self.cnn2 = nn.Sequential(*list(self.cnn2.children())[:-2])

        # add batchnorm
        # self.cnn1.add_module('BatchNorm', nn.BatchNorm2d(self.feature_size))
        # self.cnn2.add_module('BatchNorm', nn.BatchNorm2d(self.feature_size))
        
        # add dropout
        self.dropout1 = nn.Dropout(cnn_dropout)
        self.dropout2 = nn.Dropout(cnn_dropout)

        self.dropout_features = nn.Dropout(feat_dropout)

        # Define bilinear pooling
        self.fc = nn.Linear(self.feature_size**2, num_classes)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.feature_size**2, self.feature_size),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_size, num_classes)
        # )
    
    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        
        # bilinear pooling with einops
        x1 = rearrange(x1, 'b k h w -> b k (h w)')
        x2 = rearrange(x2, 'b k h w -> b k (h w)')

        # dropouts
        x1 = self.dropout1(x1)
        x2 = self.dropout2(x2)

        x = einsum(x1, x2, 'b i j, b k j -> b i k')
        x = rearrange(x, 'b i j -> b (i j)')
        x = self.dropout_features(x)

        x = self.fc(x)
        return x

model = BiLinearModel(num_classes=20, internal_model=internal_model, internal_weights=internal_weights, cnn_dropout=CNN_DROPOUT, feat_dropout=FEAT_DROPOUT)
model = model.to(device)

in_tensor = torch.randn(1, 3, 224, 224).to(device)
model(in_tensor).shape



# In[10]:


import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm 

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, frozen_weights=True):
    train_accuracy_list = []
    val_accuracy_list = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for sample in train_loader:
                image, label = sample
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * image.size(0)

                # accuracy
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == label.data)
                accuracy = corrects.double() / image.size(0)
                
                # update progress bar
                pbar.set_postfix(loss=running_loss/len(train_loader.dataset), accuracy=accuracy.item())
                pbar.update(1)
        train_accuracy_list.append(accuracy.item())

        # validation accuracy and loss
        model.eval()
        corrects = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for sample in val_loader:
                image, label = sample
                image, label = image.to(device), label.to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                val_loss += loss.item()*image.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == label.data)
                total += image.size(0)

        val_accuracy = corrects.double() / total
        val_loss = val_loss / len(val_loader.dataset)

        if frozen_weights:
            scheduler.step()
        else:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Validation accuracy: \t{val_accuracy:.4f} \t Validation loss: \t{val_loss:.4f}")
        val_accuracy_list.append(val_accuracy.item())
        model.train()


    res = {
        'model': model,
        'train_accuracy': train_accuracy_list,
        'val_accuracy': val_accuracy_list
        }

    return res

# Freeze the weights of the pre-trained models
# for param in model.parameters():
#     param.requires_grad = True
for param in model.cnn1.parameters():
    param.requires_grad = False
for param in model.cnn2.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=FREEZE_CONFIG['lr'], momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=FREEZE_CONFIG['step_lr'], gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Train the model
train_res_freeze = train_model(model, criterion, optimizer, scheduler, num_epochs=FREEZE_CONFIG['epochs'])
model = train_res_freeze['model']

# Unfreeze the weights and train again
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=UNFREEZE_CONFIG['lr'], momentum=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=UNFREEZE_CONFIG['step_lr'], gamma=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Train the model again
train_res_unfreeze = train_model(model, criterion, optimizer, scheduler, num_epochs=UNFREEZE_CONFIG['epochs'], frozen_weights=False)
model = train_res_unfreeze['model']

# evaluate the model
model.eval()
corrects = 0
total = 0
with tqdm(total=len(test_loader), desc=f"Evaluating") as pbar:
    with torch.no_grad():
        for sample in test_loader:
            image, label = sample
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == label.data)
            total += image.size(0)
            pbar.update(1)
print(f"Test Accuracy: {corrects.double()/total}")


# In[ ]:


# plot the training and validation accuracy
plt.plot(train_res_freeze['train_accuracy'], label='Train accuracy (Frozen weights)')
plt.plot(train_res_freeze['val_accuracy'], label='Validation accuracy (Frozen weights)')
plt.legend()
plt.savefig('train_val_acc.png')
plt.close()

plt.plot(train_res_unfreeze['train_accuracy'], label='Train accuracy (Unfrozen weights)')
plt.plot(train_res_unfreeze['val_accuracy'], label='Validation accuracy (Unfrozen weights)')
plt.legend()
plt.savefig('train_val_acc_unfreeze.png')
plt.close()

# combine frozen and unfrozen training accuracy
train_accuracy = train_res_freeze['train_accuracy'] + train_res_unfreeze['train_accuracy']
val_accuracy = train_res_freeze['val_accuracy'] + train_res_unfreeze['val_accuracy']

plt.plot(train_accuracy, label='Train accuracy')
plt.plot(val_accuracy, label='Validation accuracy')
plt.legend()
plt.savefig('train_val_acc_combined.png')
plt.close()


# In[ ]:





# In[ ]:


# bilinear pooling with einops
# x1 = rearrange(x1, 'b k h w -> b k (h w)')
# x2 = rearrange(x2, 'b k h w -> b k (h w)')
# x = einsum('b i j, b k j -> b i k', x1, x2)


# In[ ]:




