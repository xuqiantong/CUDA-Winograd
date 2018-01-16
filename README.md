## Introduction

This code is used to test for fast cuda kernels used in DNN inference, especially in ResNet. Specifically, the kernels combine three parts into one piece:
- Convolution (3 * 3 * 128, 3 * 3 * 256)
- Batch Nomalization (BN + Scale)
- Activation (ReLU)

## Usage
``` sh
mkdir data
python data_generator.py
make
./Test
```
- Set parameters in `data_generator.py`

## Results

### 3 * 3 Kernels
Kernals | Operations | 128 / 128 | 256 / 256 
--- | --- | --- | --- 
Cudnn | Gemm + BN + ReLU | 214us | 384us
Cudnn | Winograd + BN + ReLU  | 95us | 155us
Our Kernel | Winograd + BN + ReLU | 59us | 117us

### 1 * 1 Kernels
Kernals | 512 / 128 | 128 / 512 | 1024 / 256 | 256 / 1024 
--- | --- | --- | --- | --- 
Operations | Gemm + BN + ReLU | Gemm + BN | Gemm + BN + ReLU | Gemm + BN + ReLU
Cudnn  | 119us | 115us | 219us | 214us
Our Kernel | 58us | 55us | 186us | 181us