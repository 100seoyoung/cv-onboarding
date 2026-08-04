# Python 환경 및 VS Code 설정 확인

## 환경
- GPU: NVIDIA GeForce RTX 3090
- OS: Ubuntu 20.04.6 LTS
- NVIDIA Driver: 535.183.01
- Driver 지원 CUDA Version: 12.2
- CUDA Toolkit: 11.3
- Conda: 24.9.2
- Python: 3.11.15
- 가상환경: cv
- 작성일: 2026-08-04

## 목표
연구실 컴퓨터에 기존 Python 개발 환경이 설치되어 있는지 확인한다.

기존에 설치된 프로그램은 재설치하지 않고 그대로 사용하며, 이번 과제에서 사용할 별도의 Conda 가상환경을 생성한다.

---

## 1. Conda 설치 상태 확인

현재 시스템에 Conda가 설치되어 있는지 확인하였다.

실행한 명령어:

```bash
conda --version
which conda
```

실행 결과:

```text
conda 24.9.2
/home/cv1/miniconda3/condabin/conda
```

확인 결과:

- Conda 버전: **24.9.2**
- Conda 설치 경로: **/home/cv1/miniconda3/condabin/conda**

Conda 버전과 설치 경로가 정상적으로 출력되어 Miniconda가 이미 설치되어 있음을 확인하였다.

---

## 2. 기존 Conda 가상환경 확인

현재 시스템에 생성되어 있는 Conda 가상환경 목록을 확인하였다.

실행한 명령어:

```bash
conda env list
```

실행 결과:

```text
# conda environments:
#
base                     /home/cv1/miniconda3
4D-humans                /home/cv1/miniconda3/envs/4D-humans
contho                   /home/cv1/miniconda3/envs/contho
dposerx                  /home/cv1/miniconda3/envs/dposerx
egowholemocap            /home/cv1/miniconda3/envs/egowholemocap
hdm                      /home/cv1/miniconda3/envs/hdm
joint                    /home/cv1/miniconda3/envs/joint
mega                     /home/cv1/miniconda3/envs/mega
metric3d                 /home/cv1/miniconda3/envs/metric3d
pare-env                 /home/cv1/miniconda3/envs/pare-env
pose                     /home/cv1/miniconda3/envs/pose
sceneego                 /home/cv1/miniconda3/envs/sceneego
smplerx                  /home/cv1/miniconda3/envs/smplerx
smplerx2                 /home/cv1/miniconda3/envs/smplerx2
smplify                  /home/cv1/miniconda3/envs/smplify
tkhmr                    /home/cv1/miniconda3/envs/tkhmr
```

확인 결과 연구실에서 사용 중인 여러 Conda 가상환경이 존재했다.

기존 연구용 가상환경은 변경하지 않고 과제 전용 `cv` 가상환경을 새로 생성하기로 하였다.

---

## 3. cv 가상환경 생성

Python 3.11을 사용하는 `cv` 가상환경을 생성하였다.

실행한 명령어:

```bash
conda create -n cv python=3.11 -y
```

---

## 4. cv 가상환경 활성화

생성한 `cv` 가상환경을 활성화하였다.

실행한 명령어:

```bash
conda activate cv
```

가상환경 활성화 후 Python 버전과 실행 경로를 확인하였다.

```bash
python --version
which python
```

실행 결과:

```text
Python 3.11.15
/home/cv1/miniconda3/envs/cv/bin/python
```
확인 결과 `cv` 가상환경이 정상적으로 활성화되었으며, 해당 환경에 설치된 Python을 사용하고 있음을 확인하였다.

---

## 4-2. PyTorch 설치

현재 NVIDIA Driver가 지원하는 CUDA 버전은 12.2이므로, 이보다 낮은 CUDA 11.8용 PyTorch를 설치한다. 
먼저 `pip`를 최신 버전으로 업데이트한다. 

실행할 명령어: 

```bash 
python -m pip install --upgrade pip 
```

실행 결과: 

```text 
Requirement already satisfied: pip in ./miniconda3/envs/cv/lib/python3.11/site-packages (26.1.2)
Collecting pip
  Downloading pip-26.2-py3-none-any.whl.metadata (4.6 kB)
Downloading pip-26.2-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 19.8 MB/s  0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 26.1.2
    Uninstalling pip-26.1.2:
      Successfully uninstalled pip-26.1.2
Successfully installed pip-26.2

```

이후 CUDA 11.8용 PyTorch를 설치하였다.

실행한 명령어:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

실행 결과:

```text
Successfully installed MarkupSafe-3.0.3 filelock-3.29.0 fsspec-2026.4.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.4 nvidia-cublas-cu11-11.11.3.6 nvidia-cuda-cupti-cu11-11.8.87 nvidia-cuda-nvrtc-cu11-11.8.89 nvidia-cuda-runtime-cu11-11.8.89 nvidia-cudnn-cu11-9.1.0.70 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.3.0.86 nvidia-cusolver-cu11-11.4.1.48 nvidia-cusparse-cu11-11.7.5.86 nvidia-nccl-cu11-2.21.5 nvidia-nvtx-cu11-11.8.86 pillow-12.2.0 sympy-1.14.0 torch-2.7.1+cu118 torchaudio-2.7.1+cu118 torchvision-0.22.1+cu118 triton-3.3.1 typing-extensions-4.15.0
```

설치가 완료된 후 PyTorch 버전을 확인하였다.

실행한 명령어:

```bash
python -c "import torch; print(torch.__version__)"
```

실행 결과:

```text
2.7.1+cu118
```

확인 결과 `cv` 가상환경에 PyTorch 2.7.1과 CUDA 11.8 지원 버전이 정상적으로 설치되었다.

---


## 4-3. PyTorch 및 GPU 동작 검증

PyTorch가 실제로 NVIDIA GPU를 사용할 수 있는지 확인하기 위해 `check_env.py` 파일을 작성한다.

먼저 실습 파일을 저장할 `workspace` 폴더를 생성하고 해당 폴더로 이동하였다.

실행한 명령어:

```bash
mkdir -p ~/workspace
cd ~/workspace
```

`workspace` 폴더를 VS Code에서 열었다.

실행한 명령어:

```bash
code .
```

VS Code 왼쪽 탐색기에서 `check_env.py` 파일을 생성하고 다음 코드를 입력하였다.

```python
import torch

print("torch version :", torch.__version__)
print("built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu name      :", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok     :", (x @ x).shape)
else:
    print("GPU를 사용할 수 없습니다. 멘토에게 이 출력 전체를 보여주세요.")
```

### check_env.py 실행

작성한 파일을 실행한다.

실행한 명령어:

```bash
python check_env.py
```

실행 결과:

```text
torch version : 2.7.1+cu118
built with cuda: 11.8
cuda available: True
gpu name      : NVIDIA GeForce RTX 3090
matmul ok     : torch.Size([4096, 4096])
```

## 4-4. VS Code 설치 확인

현재 연구실 컴퓨터에 VS Code가 설치되어 있는지 확인하였다.

실행한 명령어:
```bash 
code --version 
```
실행 결과:
```text
1.97.1
e249dada235c2083c83813bd65b7f4707fb97b76
x64
```
VS Code 버전이 정상적으로 출력되어 이미 설치되어 있음을 확인하였다.

## 4-5. VS Code 필수 확장 프로그램 설치

과제에서 요구한 VS Code 확장 프로그램을 설치하였다.

현재 연구실 컴퓨터는 기존 사용자가 사용하던 환경과 파일이 남아 있으므로, 기존 VS Code 설정과 확장 프로그램을 변경하지 않고 별도의 개인용 VS Code 환경을 구성하였다.

### 개인용 VS Code 폴더 생성

개인용 VS Code 설정과 확장 프로그램을 저장할 폴더를 생성하였다.

실행한 명령어:

```bash
mkdir -p "$HOME/workspace/vscode-seoyoung/user-data"
mkdir -p "$HOME/workspace/vscode-seoyoung/extensions"
```

생성된 폴더:

```text
/home/cv1/workspace/vscode-seoyoung/user-data
/home/cv1/workspace/vscode-seoyoung/extensions
```
### Python 확장 프로그램 설치 중 발생한 문제

VS Code 확장 프로그램 화면과 터미널에서 Python 확장을 설치하려고 했으나 다음 오류가 발생하였다.

```text
Error while installing extensions: Unterminated string in JSON at position 4096
```

Python 확장의 VSIX 설치 파일을 직접 다운로드한 뒤 파일 내부의 `package.json`이 정상인지 확인하였다.

실행한 명령어:

```bash
unzip -p "/home/cv1/다운로드/ms-python.python-2026.4.0@linux-x64(1).vsix" extension/package.json | python3 -m json.tool > /dev/null && echo "VSIX JSON 정상"
```

실행 결과:

```text
VSIX JSON 정상
```

확인 결과 다운로드한 VSIX 파일 자체에는 문제가 없었다.

기존 사용자의 VS Code 환경을 변경하지 않기 위해 Python 확장을 개인용 확장 프로그램 폴더에 설치하였다.

실행한 명령어:

```bash
code \
  --user-data-dir "$HOME/workspace/vscode-seoyoung/user-data" \
  --extensions-dir "$HOME/workspace/vscode-seoyoung/extensions" \
  --install-extension "/home/cv1/다운로드/ms-python.python-2026.4.0@linux-x64(1).vsix" \
  --force
```

실행 결과:

```text
Installing extensions...
Extension 'ms-python.python-2026.4.0@linux-x64(1).vsix' was successfully installed.
```

Python 확장 설치 여부를 확인하였다.

실행한 명령어:

```bash
code \
  --user-data-dir "$HOME/workspace/vscode-seoyoung/user-data" \
  --extensions-dir "$HOME/workspace/vscode-seoyoung/extensions" \
  --list-extensions | grep ms-python.python
```

실행 결과:

```text
ms-python.python
```

Python 확장이 개인용 VS Code 환경에 정상적으로 설치된 것을 확인하였다.

### 나머지 필수 확장 프로그램 설치

Python 확장과 동일한 개인용 VS Code 환경에 다음 확장 프로그램을 설치하였다.

```bash
code \
  --user-data-dir "$HOME/workspace/vscode-seoyoung/user-data" \
  --extensions-dir "$HOME/workspace/vscode-seoyoung/extensions" \
  --install-extension [확장 프로그램 ID]
```
### 개인용 VS Code 실행

설치한 확장 프로그램을 사용하려면 다음 명령어로 개인용 VS Code를 실행해야 한다.

```bash
code "$HOME/sywhite/2026_ 학연생/cv-onboarding" \
  --user-data-dir "$HOME/workspace/vscode-seoyoung/user-data" \
  --extensions-dir "$HOME/workspace/vscode-seoyoung/extensions"
```

## 4-6. VS Code Python 인터프리터 선택

VS Code가 `cv` 가상환경의 Python을 사용하도록 인터프리터를 선택하였다.

VS Code에서 다음 순서로 실행하였다.

```text
Ctrl+Shift+P
Python: Select Interpreter
cv 환경 선택
```

선택한 Python 인터프리터 경로:

```text
/home/cv1/miniconda3/envs/cv/bin/python
```

VS Code 하단 상태 표시줄에 `cv` 환경이 표시되는지 확인하였다.

VS Code에서 `check_env.py`를 다시 실행하였다.

실행한 명령어:

```bash
cd ~/workspace
python check_env.py
```

실행 결과:

```text
torch version : 2.7.1+cu118
built with cuda: 11.8
cuda available: True
gpu name      : NVIDIA GeForce RTX 3090
matmul ok     : torch.Size([4096, 4096])
```

확인 결과:

- PyTorch 버전: **2.7.1+cu118**
- PyTorch CUDA 버전: **11.8**
- CUDA 사용 가능 여부: **True**
- GPU 이름: **NVIDIA GeForce RTX 3090**
- 행렬 곱셈 결과: **torch.Size([4096, 4096])**

VS Code에서도 `cv` 가상환경의 PyTorch가 정상적으로 실행되고 NVIDIA GPU를 인식하는 것을 확인하였다.

## 4-7. VS Code 디버거 사용

VS Code에서 `check_env.py` 파일을 열었다.

다음 줄의 왼쪽 줄 번호 영역을 클릭하여 중단점을 설정하였다.

```python
print("matmul ok     :", (x @ x).shape)
```

줄 번호 왼쪽에 빨간 점이 표시되어 중단점이 설정된 것을 확인하였다.

`F5`를 눌러 디버깅을 시작하였다.

프로그램이 중단점에서 멈춘 후 VS Code 왼쪽의 `VARIABLES` 패널에서 변수 `x`가 생성된 것을 확인하였다.

하단의 `DEBUG CONSOLE`에서 텐서의 크기, 자료형, 생성 장치를 확인하였다.

입력 및 실행 결과:

```text
x.shape
torch.Size([4096, 4096])

x.dtype
torch.float32

x.device
device(type='cuda', index=0)
```

확인 결과 텐서 `x`는 크기가 `4096 × 4096`이고, 자료형은 `torch.float32`이며, 첫 번째 CUDA GPU에서 생성된 것을 확인하였다.

## 4-8. Conda 가상환경 저장

현재 `cv` 가상환경에 설치된 Python과 패키지 버전 정보를 `environment.yml` 파일로 저장하였다.

먼저 과제 저장소 폴더로 이동하였다.

실행한 명령어:

```bash
cd "$HOME/sywhite/2026_ 학연생/cv-onboarding"
```

현재 활성화된 Conda 환경 정보를 `environment.yml` 파일로 저장하였다.

실행한 명령어:

```bash
conda env export > environment.yml
```

명령 실행 후 별도의 오류 메시지 없이 파일이 생성되었다.

파일이 정상적으로 생성되었는지 확인하였다.

실행한 명령어:

```bash
ls -lh environment.yml
```

실행 결과:

```text
ls -lh environment.yml
```

파일의 앞부분을 확인하였다.

실행한 명령어:

```bash
head -n 20 environment.yml
```

실행 결과:

```yaml
name: cv
channels:
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=52_gnu
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2026.7.16=h06a4308_0
  - ld_impl_linux-64=2.44=h9e0c5a2_3
  - libexpat=2.8.2=h7354ed3_1
  - libffi=3.4.8=h06d3fd0_3
  - libgcc=15.2.0=h69a1729_8
  - libgcc-ng=15.2.0=h166f726_8
  - libnsl=2.0.0=h5eee18b_0
  - libstdcxx=15.2.0=h39759b7_8
  - libuuid=1.41.5=h5eee18b_0
  - libxcb=1.17.0=h9b100fa_0
  - libzlib=1.3.2=h47b2149_0

```

현재 `cv` 가상환경의 Python 버전과 설치된 라이브러리 정보를 `environment.yml` 파일로 저장하였다.

---

## 배운 것 / 다음에 할 것

- Conda 가상환경마다 Python과 설치된 패키지가 독립적으로 관리된다는 것을 확인하였다.
- 기존 연구용 가상환경을 변경하지 않고 과제 전용 `cv` 가상환경을 새로 생성하였다.
- 새로 생성한 `cv` 가상환경에는 PyTorch를 별도로 설치해야 한다는 것을 확인하였다.
- NVIDIA Driver가 지원하는 CUDA 버전 이하의 PyTorch CUDA 빌드를 선택해야 한다는 것을 알게 되었다.
- `cv` 환경에 PyTorch 2.7.1과 CUDA 11.8 지원 버전을 설치하였다.
- `check_env.py`를 작성하고 PyTorch가 NVIDIA GeForce RTX 3090을 정상적으로 인식하는지 확인하였다.
- `torch.cuda.is_available()`이 `True`로 출력되어 CUDA를 사용할 수 있음을 확인하였다.
- GPU에서 `4096 × 4096` 크기의 텐서를 생성하고 행렬 곱셈이 정상적으로 수행되는 것을 확인하였다.
- 기존 사용자의 VS Code 환경을 변경하지 않고 개인용 설정 및 확장 프로그램 폴더를 별도로 구성하였다.
- Python, Pylance, Jupyter, GitLens, Ruff, Markdown All in One, Error Lens 확장 프로그램을 설치하였다.
- VS Code에서 `cv` 가상환경의 Python 인터프리터를 선택하였다.
- VS Code 통합 터미널에서도 `/home/cv1/miniconda3/envs/cv/bin/python`이 사용되는 것을 확인하였다.
- VS Code 디버거에서 중단점을 설정하고 `VARIABLES` 패널을 통해 변수 `x`를 확인하였다.
- `DEBUG CONSOLE`에서 텐서의 크기, 자료형 및 실행 장치를 확인하였다.
- 텐서 `x`의 크기는 `torch.Size([4096, 4096])`, 자료형은 `torch.float32`, 실행 장치는 첫 번째 CUDA GPU임을 확인하였다.
- `conda env export` 명령어를 사용하여 현재 `cv` 가상환경의 패키지 정보를 `environment.yml` 파일로 저장하였다.
