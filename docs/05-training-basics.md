# CIFAR-10 학습 파이프라인 구축

## 데이터셋

CIFAR-10 데이터를 사용하였다.

- 전체 train 데이터: 50,000장
- train set: 45,000장
- validation set: 5,000장
- test set: 10,000장
- seed: 42

## 데이터 확인

클래스별로 5장의 이미지를 추출하여 총 50장의 이미지를 확인하였다.

저장 위치:

`images/cifar10_samples.png`

## 정규화

CIFAR-10 train set 기준 평균과 표준편차를 사용하였다.

- Mean: (0.4914, 0.4822, 0.4465)
- Std: (0.2470, 0.2435, 0.2616)

## DataLoader

Batch size: 128
Train shuffle: True
Validation shuffle: False
Test shuffle: False
num_workers: 4
pin_memory: True
drop_last: False

## Batch 확인

x.shape: torch.Size([128, 3, 32, 32])
x.dtype: torch.float32
x.min(): -1.9894737005233765
x.max(): 2.12648868560791
y[:10]: tensor([3, 4, 2, 2, 9, 7, 1, 0, 5, 2])

한 batch에는 128장의 이미지가 포함되어 있으며, 각 이미지는 3×32×32 형태이다.

정규화 후 입력값의 범위가 약 -2에서 +2 사이로 나타나 정규화가 정상적으로 적용된 것을 확인하였다.