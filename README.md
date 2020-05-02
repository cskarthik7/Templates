# Templates

1. [kaggle api for colab](https://github.com/cskarthik7/Templates/#kaggle-api-for-colab)
2. [Dataloader Template](https://github.com/cskarthik7/Templates/#Dataloader)
3. [Training Template](https://github.com/cskarthik7/Templates/#Training-Template)
4. [Encoder Decoder of labels](https://github.com/cskarthik7/Templates/#Encoder-Decoder-of-labels)
5. [Import all Libraries at once](https://github.com/cskarthik7/Templates/#Libraries)




# kaggle api for colab
  
  Upload kaggle.json : 
    
    from google.colab import files
    files.upload()
    
  Then add the required files to initiate the kaggle.json
  
    !pip install -q kaggle
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
 
  Now search for the dataset you want : 
  e.g. : 
  
    !kaggle datasets list -s brain-image
    
  Download the dataset :
  e.g. : 
  
    !kaggle datasets download -d mateuszbuda/lgg-mri-segmentation    
    
  Unzip the dataset : 
  
    !unzip lgg-mri-segmentation.zip
    

# Dataloader  

  For Manual Class Dataset : 
  e.g. : 
  
    class Data(Dataset):
      def __init__(self,images,masks,transform=None):
        self.images=images
        self.masks=masks
        self.transform=transform
      def __getitem__(self,idx):
        image_name = self.images[idx]
        image = cv2.imread(image_name)
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        mask_name = self.masks[idx]
        mask = cv2.imread(mask_name)
        b, g, r = cv2.split(image)
        mask=cv2.merge((r, g, b))
        if self.transform:
          image=self.transform(image)
          mask=self.transform(mask)
        return image,mask
      def __len__(self):
        return len(self.images)
        
  For Transforms : 
  e.g. :
    
      transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((400,400)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
  For Trainloader and Testloader : 
  e.g. : 
  
    trainset=Data(train_images,train_masks,transform)
    testset=Data(test_images,test_masks,transform)
    train_loader=torch.utils.data.DataLoader(dataset=trainset,batch_size=5)
    test_loader=torch.utils.data.DataLoader(dataset=testset,batch_size=5)
    

# Training Template
  
    
    epochs=100
    for e in range(epochs):
      runningloss=0
      for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        output=alexnet(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        runningloss+=loss.item()
      else:
        testloss=0
        accuracy=0
        for images,labels in validation_loader:
          images,labels=images.to(device),labels.to(device)
          output=alexnet(images)
          testloss+=criterion(output,labels)
          ps=torch.exp(output)
          top_p,top_class=ps.topk(1,dim=1)
          equals=top_class==labels.view(*top_class.shape)
          accuracy+=torch.mean(equals.type(torch.FloatTensor))
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(runningloss/len(train_loader)),
              "Test Loss: {:.3f}.. ".format(testloss/len(validation_loader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
              
              
    
# Encoder Decoder of labels

Encoder : 

    def encode_labels(labels):
      encoder=dict()
      cnt=0
      for i in range(len(labels)):
        if labels[i] in encoder.keys():
          continue
        else:
          encoder[labels[i]]=cnt
          cnt=cnt+1



Decoder : 
     
    def decode_labels(encoder):
      decoder={}
      for keys in encoder.keys():
        decoder[encoder[keys]]=keys
        
        
        
# Libraries

    from __future__ import print_function
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import torch
    import torch.utils.data
    from torch import nn, optim
    from torch.autograd import Variable
    from torch.nn import functional as F
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from PIL import Image
    import numpy as np
    %matplotlib inline
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch import optim
    from torchvision import datasets,transforms
    import torch.nn.functional as F
    import cv2
    from torch.utils.data import Dataset, DataLoader
    import os
    from PIL import Image
    import numpy as np
    import torchvision
    import torch
    from torch.utils.data import sampler
    from torch.utils.data.sampler import SubsetRandomSampler 
 
  
    
