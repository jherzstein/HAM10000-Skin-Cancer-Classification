import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import datetime
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchsummary import summary
from PIL import Image

# Constants
NUM_CLASS = 7

# Train function
def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device,
          parameter_file=None, save_file=None, plot_file=None):
    print('training ...')

    losses_train = []
    losses_val = []
    best_val_loss = float('inf')
    early_stop_count = 0
    patience = 5  # Early stopping patience

    for epoch in range(1, n_epochs + 1):
        # Training mode
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

        # Step the scheduler
        scheduler.step()
        losses_train.append(loss_train / len(train_loader))

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))

        # Save model after every epoch
        if save_file is not None:
            torch.save(model.state_dict(), save_file)
            print(f"Model checkpoint saved at epoch {epoch} to {save_file}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        losses_val.append(val_loss / len(val_loader))
        print('{} Epoch {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, val_loss / len(val_loader)))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered!")
                break

        # Plot losses
        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

    # Save final model
    if save_file is not None:
        torch.save(model.state_dict(), save_file)
        print(f"Final model saved to {save_file}")


# Data Loader Helper
def Img_loader(path):
    return Image.open(path).convert('RGB')


# Main function
def main():
    save_file = './model.pth'
    n_epochs = 50
    batch_size = 64
    plot_file = 'loss.png'

    model_name = ""

    # Handle arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=int, help='train model, 1: alexnet, 2: vgg, 3: resnet, 4: googlenet')
    argParser.add_argument('-s', metavar='batch size', type=int, help='batch size for training')
    argParser.add_argument('-e', metavar='epochs', type=int, help='number of epochs')

    args = argParser.parse_args()

    if args.m is not None:
        if args.m == 1:
            model_name = "alexnet"
            save_file = "alexnet.pth"
        elif args.m == 2:
            model_name = "vgg"
            save_file = "vgg.pth"
        elif args.m == 3:
            model_name = "resnet"
            save_file = "resnet.pth"
        elif args.m == 4:
            model_name = "googlenet"
            save_file = "googlenet.pth"
    if args.s is not None:
        batch_size = args.s
    if args.e is not None:
        n_epochs = args.e

    if model_name == "":
        print("Error, no model specified!\n")
        return

    print('\t\toutput model name = ', save_file)
    print('\t\ttraining model = ', model_name)
    print('\t\tbatch size = ', batch_size)
    print('\t\tnumber of epochs = ', n_epochs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tusing device ', device)

    # Data augmentation
    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load datasets
    DatasetFolder_train = datasets.DatasetFolder(root='./data/train', loader=Img_loader,
                                                 extensions=('jpg', 'JPG'), transform=transform)
    DatasetFolder_val = datasets.DatasetFolder(root='./data/val', loader=Img_loader,
                                               extensions=('jpg', 'JPG'), transform=transform)

    train_loader = DataLoader(DatasetFolder_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DatasetFolder_val, batch_size=batch_size, shuffle=False)

    # Select and configure model
    if model_name == "alexnet":
        train_model = models.alexnet(pretrained=True)
        train_model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.6),  # Increased dropout
            nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
        )
        optimizer = optim.Adam(train_model.parameters(), lr=5e-5, weight_decay=1e-3)
    elif model_name == "vgg":
        train_model = models.vgg16(pretrained=True)
        train_model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(train_model.classifier[6].in_features, NUM_CLASS)
        )
        optimizer = optim.Adam(train_model.parameters(), lr=5e-5, weight_decay=1e-3)
    elif model_name == "resnet":
        train_model = models.resnet18(pretrained=True)
        train_model.fc = nn.Linear(train_model.fc.in_features, NUM_CLASS)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif model_name == "googlenet":
        train_model = models.googlenet(pretrained=True)
        train_model.fc = nn.Linear(train_model.fc.in_features, NUM_CLASS)
        optimizer = optim.Adam(train_model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        print("Error, invalid model name!\n")
        return

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    train_model.to(device)
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
        plot_file=plot_file
    )


if __name__ == '__main__':
    main()
