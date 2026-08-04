# CV 온보딩 기록 — 백서영

광운대학교 정보융합학부 학부연구생 컴퓨터비전 온보딩 과정 기록입니다.

## 환경

- OS: Ubuntu 20.04.6 LTS
- GPU: NVIDIA GeForce RTX 3090
- NVIDIA Driver: 535.183.01
- CUDA Toolkit: 11.3
- Python: 3.11.15
- PyTorch: 2.7.1+cu118
- Conda 환경: `cv`

## 목차

1. [Ubuntu 환경 확인](docs/01-ubuntu-install.md)
2. [NVIDIA 드라이버 및 CUDA 확인](docs/02-nvidia-cuda.md)
3. [Python 및 VS Code 설정](docs/03-vscode-setup.md)
4. [YOLO 데모 및 실험](docs/04-yolo-demo.md)

## Python 환경 실행

```bash
conda env create -f environment.yml
conda activate cv
python src/check_env.py
```
