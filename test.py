import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import argparse
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader

alexnet_dir = "./alexnet_model.pth"
vgg_dir = "./vgg_model.pth"
resnet_dir = "./resnet_model.pth"

def ensemble(models, img, device, method="max_prob"):
    outputs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            raw_output = model(img.to(device))
            outputs.append(F.softmax(raw_output, dim=1))    # do softmax manually, apply through row

    top5_probs = []
    top5_indices = []
    if method == "max_prob":
        stacked_outputs = torch.stack(outputs)
        flattened_outputs = stacked_outputs.view(-1)

        topk_values, topk_flat_indices = torch.topk(flattened_outputs, len(models) * 5, dim=0)
        topk_class_indices = topk_flat_indices % stacked_outputs.size(2)
        unique_classes = set()
        unique_topk_probs = []
        unique_topk_indices = []
        for prob, class_idx in zip(topk_values, topk_class_indices):
            if class_idx.item() not in unique_classes:
                unique_classes.add(class_idx.item())
                unique_topk_probs.append(prob)
                unique_topk_indices.append(class_idx)
            if len(unique_topk_probs) == 5:
                break

        top5_probs = torch.tensor(unique_topk_probs)
        top5_indices = torch.tensor(unique_topk_indices)
        top5_probs = top5_probs[:5]
        top5_indices = top5_indices[:5]
    elif method == "avg_prob":
        ensemble_probs = torch.stack(outputs).mean(dim=0)
        top5_probs, top5_indices = ensemble_probs.topk(5, dim=1)
        top5_probs = top5_probs.view(-1)
        top5_indices = top5_indices.view(-1)
    elif method == "majority_vote":
        votes = [torch.argmax(softmax, dim=1).long() for softmax in outputs]
        votes_stacked = torch.stack(votes)
        num_classes = outputs[0].shape[1]
        vote_counts = torch.bincount(votes_stacked.view(-1), minlength=num_classes)
        top5_probs, top5_indices = torch.topk(vote_counts, 5)
        top5_probs = top5_probs.view(-1)
        top5_indices = top5_indices.view(-1)
    else:
        raise ValueError(f"Invalid ensemble method: {method}")

    return top5_probs, top5_indices

def main():
    base_dir = "./"
    method = ""
    alexnet_flag = False
    vgg_flag = False
    resnet_flag = False
    googlenet_flag = False
    early_model = False

    NUM_CLASS = 100

    # handle arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-b', metavar='base dir', type=str, help='parameter file base dir (.pth)')
    argParser.add_argument('-a', metavar='alexnet', type=int, help='alexnet flag, 1 for true, 0 for false')
    argParser.add_argument('-v', metavar='vgg', type=int, help='vgg flag, 1 for true, 0 for false')
    argParser.add_argument('-r', metavar='resnet', type=int, help='resnet flag, 1 for true, 0 for false')
    argParser.add_argument('-g', metavar='googlenet', type=int, help='googlenet flag, 1 for true, 0 for false')
    argParser.add_argument('-m', metavar='ensemble method', type=int, help='emsemble method, 1: max prob, 2: avg prob, 3: majority vote')
    argParser.add_argument('-e', metavar='5 epoch model', type=int, help='using 5 epoch model flag, 1 for True, 0 for False')

    args = argParser.parse_args()

    if args.b != None:
        base_dir = args.b
    if args.a != None:
        if args.a == 1:
            alexnet_flag = True
    if args.v != None:
        if args.v == 1:
            vgg_flag = True
    if args.r != None:
        if args.r == 1:
            resnet_flag = True
    if aggs.r != None:
        if args.g == 1:
            googlenet_flag = True
    if args.m != None:
        if args.m == 1:
            method = "max_prob"
        elif args.m == 2:
            method = "avg_prob"
        elif args.m == 3:
            method = "majority_vote"
        else:
            method = "NONE"
    if args.e != None:
        if args.e == 1:
            early_model = True

    print('\t\tbase directory = ', base_dir)
    print('\t\tapply alexnet = ', alexnet_flag)
    print('\t\tapply vgg = ', vgg_flag)
    print('\t\tapply resnet = ', resnet_flag)
    print('\t\tapply googlenet = ', googlenet_flag)
    print('\t\tensemble method = ', method)
    print('\t\tusing 5 epochs model = ', early_model)


    print('running main ...')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    loaded_models = []

    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    resnet18 = models.resnet18(pretrained=True)
    googlenet = models.googlenet(pretrained=True)

    if alexnet_flag:
        # modify output layer
        in_features = alexnet.classifier[6].in_features
        alexnet.classifier[6] = nn.Linear(in_features, NUM_CLASS)

        if early_model:
            alexnet.load_state_dict(torch.load(base_dir + "alexnet_5epochs.pth"), strict=False)
        else:
            alexnet.load_state_dict(torch.load(base_dir + "alexnet.pth"), strict=False)

        alexnet.to(device)
        loaded_models.append(alexnet)
    
    if vgg_flag:
        # modify output layer
        in_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(in_features, NUM_CLASS)

        if early_model:
            vgg16.load_state_dict(torch.load(base_dir + "vgg_5epochs.pth"), strict=False)
        else:
            vgg16.load_state_dict(torch.load(base_dir + "vgg.pth"), strict=False)

        vgg16.to(device)
        loaded_models.append(vgg16)

    if resnet_flag:
        # modify output layer
        in_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(in_features, 100)
        
        if early_model:
            resnet18.load_state_dict(torch.load(base_dir + "resnet_5epochs.pth"), strict=False)
        else:
            resnet18.load_state_dict(torch.load(base_dir + "resnet.pth"), strict=False)
        
        resnet18.to(device)
        loaded_models.append(resnet18)    

    if googlenet_flag:
        # modify output layer
        in_features = googlenet.fc.in_features
        googlenet.fc = nn.Linear(in_features, NUM_CLASS)
        
        if early_model:
            googlenet.load_state_dict(torch.load(base_dir + "googlenet_5epochs.pth"), strict=False)
        else:
            googlenet.load_state_dict(torch.load(base_dir + "googlenet.pth"), strict=False)
        
        googlenet.to(device)
        loaded_models.append(googlenet)    
    
    

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    DatasetFolder_val = datasets.DatasetFolder(root='./data/test', loader = Img_loader, extensions=('JPG', '.jpg', '.JPG', 'jpg'), train=False, transform=transform)
    val_loader = DataLoader(DatasetFolder_val, batch_size=1, shuffle=False)

    top5_correct = 0
    top1_correct = 0
    total_imgs = len(val_loader)

    with torch.no_grad():
        for img, gt_val in val_loader:
            top5_count, top5_idx = ensemble(loaded_models, img, device, method)
            gt_val = gt_val.to(top5_idx.device)
            if gt_val in top5_idx:
                curr_idx = torch.nonzero(top5_idx == gt_val)[0][0]
                if top5_count[curr_idx] > 0:
                    top5_correct += 1

            if gt_val == top5_idx[0]:
                top1_correct += 1
            
            # print(f"probability: {top5_count}\n")
            # print(f"idx: {top5_idx}\n")
            # print(f"gt idx: {gt_val}\n")

    top1_error_rate = 1 - top1_correct / total_imgs
    top5_error_rate = 1 - top5_correct / total_imgs

    print(f"top 1 error rate: {top1_error_rate}\n")
    print(f"top 5 error rate: {top5_error_rate}\n")





if __name__ == '__main__':
    main()
