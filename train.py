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

# constant
NUM_CLASS = 100


# train function
def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, 
            parameter_file=None, save_file=None, plot_file=None):
    print('training ...')

    losses_train = []
    losses_val = []
    for epoch in range(1, n_epochs+1):
        # config the mode to train
        model.train()

        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, gt_val in train_loader:
            imgs = imgs.to(device=device)
            gt_val = gt_val.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, gt_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

        if save_file != None:
            torch.save(model.state_dict(), save_file)

        # save first sets of parameters: after 5 epochs
        if epoch == 5 and parameter_file != None:
            torch.save(model.state_dict(), f"{parameter_file}_5epochs.pth")

        # config the mode to evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        losses_val += [val_loss/len(val_loader)]

        print('{} Epoch {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, val_loss/len(val_loader)))

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def Img_loader(path):
    return Image.open(path).convert('RGB')

def main():
    save_file = './model.pth'
    n_epochs = 50
    # for vgg has more layers than alexnet and resnet, therefore, might need a smaller batch size
    batch_size = 64
    plot_file = 'loss.png'

    model_name = ""

    # handle arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=int, help='train model, 1: alexnet, 2: vgg, 3: resnet')
    argParser.add_argument('-s', metavar='batch size', type=int, help='batch size for training')
    argParser.add_argument('-e', metavar='epochs', type=int, help='number of epochs')

    args = argParser.parse_args()

    if args.m != None:
        if args.m == 1:
            model_name = "alexnet"
            save_file = "alexnet.pth"
        elif args.m == 2:
            model_name = "vgg"
            save_file = "vgg.pth"
        elif args.m == 3:
            model_name = "resnet"
            save_file = "resnet.pth"
    if args.s != None:
        batch_size = args.s
    if args.e != None:
        n_epochs = args.e

    if model_name == "":
        print("Error, no model specified!\n")
        return
        
    print('\t\toutput model name = ', save_file)
    print('\t\ttraining model = ', model_name)
    print('\t\tbatch size = ', batch_size)
    print('\t\tnumber of epochs = ', n_epochs)


    print('running main ...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)


    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Authenticate the Kaggle API
#     api = KaggleApi()
#     api.authenticate()
    
    # Define the dataset and download path
#     dataset = 'surajghuwalewala/ham1000-segmentation-and-classification'
#     download_path = './data/'
    
#     # Create the directory if it doesn't exist
#     os.makedirs(download_path, exist_ok=True)
    
    # Download the dataset
#     api.dataset_download_files(dataset, path=download_path, unzip=True)


    # Load HAM10000 dataset
    DatasetFolder_train = datasets.DatasetFolder(root='./data/train',loader = Img_loader, extensions=('JPG','.jpg','.JPG','jpg'),  transform=transform)
    DatasetFolder_val = datasets.DatasetFolder(root='./data/val',loader = Img_loader,extensions=('JPG','.jpg','.JPG','jpg'),  transform=transform)

    train_loader = DataLoader(DatasetFolder_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DatasetFolder_val, batch_size=batch_size, shuffle=False)


    # need to replace the final layer with a new nn.Linear layer matching the num of class
    if model_name == "alexnet":
        train_model = models.alexnet(pretrained=True)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_model.classifier[6] = nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
    elif model_name == "vgg":
        train_model = models.vgg16(pretrained=True)
        optimizer = optim.Adam(train_model.parameters(), lr=5e-5, weight_decay=1e-3)
        train_model.classifier[6] = nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
    elif model_name == "resnet":
        train_model = models.resnet18(pretrained=True)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-3, weight_decay=1e-7)
        train_model.fc = nn.Linear(train_model.fc.in_features, NUM_CLASS)
    else:
        print("Error, invalid model name!\n")
        return
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.CrossEntropyLoss()

    train_model.to(device)
    train_model.apply(init_weights)
    summary(train_model, input_size=(3, 224, 224))


    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=train_model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device,
            parameter_file=model_name,
            save_file=save_file,
            plot_file=plot_file)

if __name__ == '__main__':
    main()
