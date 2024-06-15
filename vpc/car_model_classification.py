#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# !nvidia-smi


# In[17]:


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


# In[18]:


# internal_model = models.resnet18
# internal_weights = models.ResNet18_Weights.IMAGENET1K_V1

internal_model = models.resnet34
internal_weights = models.ResNet34_Weights.IMAGENET1K_V1

# internal_model = models.resnet50
# internal_weights = models.ResNet50_Weights.IMAGENET1K_V1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[24]:


# Load images
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

## View some images
# plt.imshow(x_train[2,:,:,: ] )
# plt.axis('off')
# plt.show()

# # convert to torch
# x_train = torch.from_numpy(x_train)
# x_test = torch.from_numpy(x_test)

# y_train = torch.from_numpy(y_train)
# y_test = torch.from_numpy(y_test)

print('X_train shape:\t' ,x_train.shape)
print('Y_train shape\t' ,y_train.shape)

print('X_test shape\t' ,x_test.shape)
print('Y_test shape\t' ,y_test.shape)

train_cut = 100
x_train = x_train[:train_cut]
y_train = y_train[:train_cut]

test_cut = 100
x_test = x_test[:test_cut]
y_test = y_test[:test_cut]

# print('X_train shape:\t' ,x_train.shape)
# print('X_train dtype:\t' ,x_train.dtype)
# print('X_train type:\t' ,type(x_train))


# In[ ]:





# In[40]:


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

# add dimension to labels
# y_train_tensor = y_train_tensor.unsqueeze(1)
# y_test_tensor = y_test_tensor.unsqueeze(1)

# TensorDataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[41]:


sample, label = next(iter(train_loader))
print('Sample shape:', sample.shape)
print('Label shape:', label.shape)


# ### Model

# In[43]:


class BiLinearModel(nn.Module):
    def __init__(self, num_classes):
        super(BiLinearModel, self).__init__()
        
        self.cnn1 = internal_model(weights=internal_weights)
        self.cnn2 = internal_model(weights=internal_weights)
        

        self.cnn1 = nn.Sequential(*list(self.cnn1.children())[:-2])
        self.cnn2 = nn.Sequential(*list(self.cnn2.children())[:-2])
        
        # Define bilinear pooling
        self.fc = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        
        # # Bilinear pooling
        # batch_size = x1.size(0)
        # x1 = x1.view(batch_size, 512, 49)
        # x2 = x2.view(batch_size, 512, 49)
        # x = torch.bmm(x1, x2.transpose(1, 2)) / 49
        # x = x.view(batch_size, 512*512)
        
        # bilinear pooling with einops
        x1 = rearrange(x1, 'b c h w -> b c (h w)')
        x2 = rearrange(x2, 'b c h w -> b c (h w)')
        # print(x1.shape)
        # print(x2.shape)
        x = einsum(x1, x2, 'b c1 i, b c2 j -> b c1 c2')
        x = rearrange(x, 'b c1 c2 -> b (c1 c2)')
        # print(x.shape)

        x = self.fc(x)
        return x

model = BiLinearModel(num_classes=20).to(device)

in_tensor = torch.randn(1, 3, 224, 224).to(device)
model(in_tensor).shape


# In[ ]:





# In[45]:


import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm 

# Freeze the weights of the pre-trained models
for param in model.cnn1.parameters():
    param.requires_grad = False
for param in model.cnn2.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for sample in train_loader:
            image, label = sample
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            # update progress bar
            running_loss = loss.item()
            epoch_pbar.set_description(f"Epoch {epoch}/{num_epochs - 1}, Loss: {running_loss:.4f}")
            epoch_pbar.update()
        scheduler.step()
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {loss.item()}")
    
    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Unfreeze the weights and train again
for param in model.cnn1.parameters():
    param.requires_grad = True
for param in model.cnn2.parameters():
    param.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model again
model = train_model(model, criterion, optimizer, scheduler, num_epochs=15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




