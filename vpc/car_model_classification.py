#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[56]:


# !nvidia-smi


# In[74]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, Dataset, DatasetDict
from einops import rearrange, einsum

import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18


# In[75]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[76]:


# Load images
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')

y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

## View some images
# plt.imshow(x_train[2,:,:,: ] )
# plt.axis('off')
# plt.show()

# convert to torch
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

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


train_dataset = Dataset.from_dict({'image': x_train, 'label': y_train}).with_format('torch')
test_dataset = Dataset.from_dict({'image': x_test, 'label': y_test}).with_format('torch')

dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})


# In[77]:


transform_fn = models.ResNet18_Weights.IMAGENET1K_V1.transforms

def process_data(sample):
    sample['image'] = sample['image'].to(device)
    sample['label'] = sample['label'].to(device)
    sample = rearrange(sample, 'h w c -> c h w')
    sample = transform_fn(sample)
    return sample

# # cuda
# dataset = dataset.map(lambda x: {'image': x['image'].to(device)})

# # rearrange
# dataset = dataset.map(lambda x: {'image': rearrange(x['image'], 'h w c -> c h w')})

# # normalize
# dataset = dataset.map(lambda x: {'image': transform_fn(x['image'])})

dataset = dataset.map(process_data)

train_loader = torch.utils.data.DataLoader(dataset['traºin'], batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset['test'], batch_size=32, shuffle=True)


# ### Model

# In[68]:


internal_model = models.resnet18
internal_weights = models.ResNet18_Weights.IMAGENET1K_V1

class BiLinearModel(nn.Module):
    def __init__(self, num_classes):
        super(BiLinearModel, self).__init__()
        
        self.cnn1 = internal_model(weights=internal_weights)
        self.cnn2 = internal_model(weights=internal_weights)
        

        self.cnn1 = nn.Sequential(*list(self.cnn1.children())[:-2])
        self.cnn2 = nn.Sequential(*list(self.cnn2.children())[:-2])
        
        # Define bilinear pooling
        self.fc = nn.Linear(512*512, num_classes)
    
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
model(torch.randn(1, 3, 224, 224)).shape


# In[65]:





# In[66]:


import torch.optim as optim
from torch.optim import lr_scheduler

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
        for sample in train_loader:
            inputs, labels = sample['image'], sample['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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




