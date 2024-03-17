# 💻 딥러닝 TERM PROJECT
```
2023년 2학기 CSE4048 딥러닝 강의에서 진행한 TERM PRJOCET
```

## Goal
[Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) 데이터셋을 이용하여 모델을 학습시킨다.  
**단, class에 속하는 동물의 그림, 소묘 등 다양한 domain에 대해서도 분류할 수 있도록 해야한다.**   
(`./src/test_images` 폴더 참고)

## Train
```
cd src
python3 train.py
```
`./src/data` 디렉토리에 학습을 위한 데이터가 존재해야 합니다.

## Run
```
cd src
python3 run.py
```
`result.txt`을 확인하여 어떻게 분류했는지 알 수 있습니다.

## Result
**최종 Accuracy가 92.32%로 수강생 중 1등**  
자세한 경과는 `./report/report.pdf`를 통해 확인할 수 있습니다.
