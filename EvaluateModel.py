import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import datetime
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,TensorDataset
from torchvision.transforms import v2
from torchsummary import summary
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image


NUM_CLASS = 7






def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=int, help='train model, 1: alexnet, 2: vgg, 3: resnet')
    argParser.add_argument('-p', metavar='Parameters', type=str, help='Trained Parameters')
    argParser.add_argument('-i', metavar='Image', type=str, help='Image to evaluate')
    
    args = argParser.parse_args()
    
    if args.m != None:
        if args.m == 1:
            model_name = "alexnet"
            
        elif args.m == 2:
            model_name = "vgg"
            
        elif args.m == 3:
            model_name = "resnet"

    if model_name == "":
        print("Error, no model specified!\n")
        return
    
    save_file = args.p
    
    print('\t\tinput model parameters = ', save_file)
    print('\t\ttraining model = ', model_name)
    
   
    
    
    # need to replace the final layer with a new nn.Linear layer matching the num of class
    if model_name == "alexnet":
        train_model = models.alexnet(weights=True)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_model.classifier[6] = nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
    elif model_name == "vgg":
        train_model = models.vgg16(weights=True)
        optimizer = optim.Adam(train_model.parameters(), lr=5e-5, weight_decay=1e-3)
        train_model.classifier[6] = nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
    elif model_name == "resnet":
        train_model = models.resnet18(weights=True)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-3, weight_decay=1e-7)
        train_model.fc = nn.Linear(train_model.fc.in_features, NUM_CLASS)
    else:
        print("Error, invalid model name!\n")
        return
    #Set model to evaluate
    train_model.load_state_dict(torch.load(save_file,weights_only=True))
    train_model.eval()
    
    
    image_path = args.i
    image = Image.open(image_path).convert('RGB')
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():  # Disable gradients for prediction
        output = train_model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        print(predicted_class)
if __name__ == '__main__':
    main()

