import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel
from torchvision.models import resnet18, efficientnet_b3

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    
    model.eval()
    
    pbar = tqdm(data_loader)
    for i, (x, y) in enumerate(pbar):
        
        image = x.to(args.device)
        label = y.to(args.device)                

        output = model(image)
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()
        pred = np.append(pred,output, axis=0)
        
        label = label.detach().cpu().numpy()
        true =  np.append(true,label, axis=0)
    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--model-path', default='checkpointsmodel.pth', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # hyperparameters
    args.batch_size = 4
    
    # Make Data loader and Model
    _, test_loader = make_data_loader(args)

    # instantiate model
    model = BaseModel()
    # model = efficientnet_b3()
    # num_classes = 10
    # num_features = model._modules['classifier']._modules['1'].in_features
    # model._modules['classifier']._modules['1'] = torch.nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Test The Model
    pred, true = test(args, test_loader, model)
        
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))

    