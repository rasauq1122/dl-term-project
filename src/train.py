import argparse
import test

import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
from datetime import datetime

import torch
import torch.nn as nn # edited
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from run import ImageDataset, inference

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, val_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    max_val = 0.0
    model_id = str(datetime.now()).replace(" ","").replace(":", "").replace("-", "").replace(".", "")

    t_acc_list = []
    v_acc_list = []

    test_data = ImageDataset('test_images', transform=transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        epoch_str = f'{epoch:3d}'.replace(" ", "0")
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)
        for _, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        # scheduler.step()
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        
        print(f'{model_id}')
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        t_acc_list.append(epoch_train_acc*100)
        
        pred, true = test.test(args, val_loader, model)
        val_accuracy = (true == pred).sum() / len(pred)

        # test_result = inference(args, test_loader, model)
        # test_accuracy = 0
        # for i in range(100):
            # if test_result[i] == i//10:
                # test_accuracy = test_accuracy+1

        if max_val < val_accuracy:
            torch.save(model.state_dict(), f'{args.save_path}/{model_id}-{epoch_str}-best.pth')
            max_val = val_accuracy
        torch.save(model.state_dict(), f'{args.save_path}/{model_id}-model.pth')    
        print('origin val_accuracy : {:.3f}'.format(val_accuracy*100))

        v_acc_list.append(val_accuracy*100)

        print(v_acc_list)
        print(t_acc_list)
        # torch.save(model.state_dict(), f'{args.save_path}/{model_id} model.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    parser.add_argument('--load-model', default=None, type=str, help='load model')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    args.device = device
    num_classes = 10 # edited
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 100
    args.learning_rate = 0.00001
    args.batch_size = 16

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, val_loader = make_data_loader(args)

    # custom model
    torch.cuda.empty_cache()
    model = BaseModel(freeze=True)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
        for param in model.parameters():
            param.requires_grad = True


    # torchvision model
    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # num_features = model.fc.in_features # edited
    # model.fc = nn.Linear(num_features, num_classes) # edited

    # you have to change num_classes to 10
    # model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    # num_features = model._modules['classifier']._modules['1'].in_features
    # model._modules['classifier']._modules['1'] = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
    model.to(device)

    print(model)

    # Training The Model
    train(args, train_loader, val_loader, model)