import numpy as np
import imageio
import imgaug.augmenters as iaa
from PIL import Image
import os

# Augmentation 시퀀스 정의
seq = iaa.Sequential([
    iaa.color.AddToHueAndSaturation((-50, 50)),
    iaa.Affine(rotate=(-25, 25)),  # 이미지 회전 (-25도에서 25도 사이)
    iaa.Fliplr(0.5),  # 수평으로 뒤집기
    iaa.Flipud(0.2),  # 수직으로 뒤집기
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))),  # 가우시안 블러
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2*255), per_channel=0.5)),  # 노이즈 추가
    iaa.SomeOf((0, 5), [
        iaa.Crop(percent=(0, 0.1)),  # 이미지 일부 자르기
        iaa.Affine(rotate=(-45, 45)),  # 다른 각도로 회전
        iaa.Affine(scale=(0.5, 1.5)),  # 크기 조정
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # 이미지 선명도 변경
    ])
], random_order=True)  # 변환을 무작위로 적용

# 이미지 로드
pdata_directory = 'pdata'
for folder_name in os.listdir(pdata_directory):
    folder_path = os.path.join(pdata_directory, folder_name)
    
    # 폴더에 있는 이미지 파일들에 대해 반복
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_extension = filename.split('.')[-1].upper()
        image = Image.open(file_path)
        image = image.convert("RGB")
        # 이미지 증강 및 9개의 변형된 이미지 생성
        images_aug = [seq(image=np.array(image)) for _ in range(9)]
        
        # 변형된 이미지를 저장할 폴더 생성
        save_folder = os.path.join(folder_path, 'augmented')
        os.makedirs(save_folder, exist_ok=True)
        
        # 변형된 이미지를 파일로 저장
        for idx, img_aug in enumerate(images_aug):
            save_path = os.path.join(save_folder, f"{filename.split('.')[0]}_aug_{idx+1}.{file_extension.lower()}")
            img_aug = Image.fromarray(img_aug)
            img_aug.save(save_path)