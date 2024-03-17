from torchvision import datasets, transforms
import torch
from torchvision import models
# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# you can change input size(don't forget to change linear layer!)
train_transform = transforms.Compose([
    transforms.Resize(292),  # 이미지 크기 조정
    transforms.RandomCrop(260),  # 무작위로 이미지 자르기
    transforms.RandomHorizontalFlip(),  # 무작위로 좌우 반전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 조절
    transforms.RandomRotation(degrees=15),  # 무작위로 이미지 회전
    transforms.ToTensor(),  # Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
])

def make_data_loader(args):
    
    # Get Dataset
    dataset = datasets.ImageFolder(args.data, transform=test_transform)
    # mydataset = datasets.ImageFolder('mydata/', transform=models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms, split="train")
    
    print(args.data)

    # split dataset to train/test
    train_data_percentage = 0.8
    train_size = int(train_data_percentage * len(dataset))
    test_size = len(dataset) - train_size
    
    # you must set "seed" to get same test data
    # you can't compare different test set's accuracy
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataset.dataset.transform = models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms
    print(train_dataset.dataset.transform)
    test_dataset.dataset.transform = test_transform
    
    # combined_dataset = ConcatDataset([train_dataset, mydataset])
    # print(combined_dataset )
    # Get Dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader