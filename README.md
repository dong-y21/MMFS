# MMFS 
Official PyTorch implementation of the PG 2023 paper: Multi-Modal Face Stylization with a Generative Prior

## 预训练模型
下载以下预训练模型并保存在pretrained_models文件夹下   
phase2_pretrain_90000.pth：预训练的第二阶段多模态模型（AAHQ）  [【baidu】](https://pan.baidu.com/s/1rz7rPjngmdcL28sWwPJmKw?pwd=2xsg)   [【google】](https://drive.google.com/file/d/1jPXIR5UqkWS7chsMZ-SR-yAs0WIGCO3p/view?usp=drive_link)  
phase3_pretrain_10000.pth：预训练的第三阶段one/zero-shot初始模型   [【baidu】](https://pan.baidu.com/s/1w2OLkAUSPQbwxXu_30naCw?pwd=ncm9)  [【google】](https://drive.google.com/file/d/12AfgFfOs8PjagtYwglmO9bquO2AEzPof/view?usp=drive_link)

## one/zero-shot预训练模型
下载以下预训练风格模型可以直接进行第四阶段的风格化推理测试
### one-shot
example/reference/01.png：[百度网盘](https://pan.baidu.com/s/1S2YCXh14hLq2bILW3asmQw?pwd=wjmd)  [google drive](https://drive.google.com/file/d/1nip981zqzASsPu6EiRRXBYvHOAosqPMj/view?usp=drive_link)  
example/reference/02.png：[百度网盘](https://pan.baidu.com/s/17uclEk1bPOmwjDDtU9rtuQ?pwd=qvjx)  [google drive](https://drive.google.com/file/d/1Lq1PqeHKWbNgoIFsCHCSzXPIKmHRDtlh/view?usp=drive_link)  
example/reference/03.png：[百度网盘](https://pan.baidu.com/s/1ma6ueCq0o45mWEC8uSnecg?pwd=37md)  [google drive](https://drive.google.com/file/d/1UCBpnT7BC4fd1l7vu7YalyPz8ugPbom8/view?usp=drive_link)  
example/reference/04.png：[百度网盘](https://pan.baidu.com/s/1Q60Jejc9EuE3lDr7-mPv1w?pwd=x8d4)  [google drive](https://drive.google.com/file/d/1qEjDFsX-z1anpDr54dP5VG2LmSS1DU3R/view?usp=drive_link)  
### zero-shot
pop art：[百度网盘](https://pan.baidu.com/s/1hkjJQrwIPHWEasZmL3aViA?pwd=4uxi)  [google drive](https://drive.google.com/file/d/17a0OJjF4PuSCIouDMnVuc5iiGLRPZhOx/view?usp=drive_link)  
watercolor painting：[百度网盘](https://pan.baidu.com/s/1kQHr0Plbcux9cZ9GOdfWNA?pwd=atve)  [google drive](https://drive.google.com/file/d/1QGgzsiXQgJt_gjRMFQbv5_qS0kgntzBV/view?usp=drive_link)  

## 1. 介绍
此仓库包含MMFS部分训练和测试代码，也提供了前三个阶段的预训练模型测试

## 2. 配置依赖项
安装requirements.txt中依赖项

## 3. 训练模型

### 3.1 训练指令及流程解释

训练代码（例）：
```bash
python train.py --cfg_file ./exp/sp2pII-phase4.yaml
```

流程解释：
`train.py`对指定的参数文件（--cfg_file）的读取过程如下：
1. 首先，[configs/base_config.py](configs/base_config.py)中的参数会被读取并添加到参数池中。这些参数为统一参数，被所有任务引用。
2. 在[configs](configs)下的同名任务参数文件会被读取。每个文件包含那个任务专属的参数。这些参数会被添加到参数池中。
3. [train.py](train.py)会读取指定的--cfg_file中的参数并覆盖参数池中已有的值。

在所有参数文件都被读取后，训练开始。

### 3.2 one/zero-shot训练
第四阶段finetune模型提高one/zero-shot效果使用[sp2pII-phase4.yaml](exp/sp2pII-phase4.yaml)：需要指定yaml文件中image_prompt（one-shot的引导图路径）或text_prompt（zero-shot的引导文本），以及pretrained_model（第三阶段保存的预训练模型路径）



## 4. 模型推理

### 4.1 多模态推理
[test.py](test.py)需要指定任务的config文件，需要推理的文件夹或图片，和使用的模型checkpoint。生成的结果会在results文件夹下，
或可通过`--overwrite_output_dir`来指定。  

对一整个文件夹进行推理的代码示例：
```bash
python test.py --cfg_file ./exp/sp2pII-phase2.yaml --test_folder path/to/your/test/folder 
--ckpt path/to/your/phase2/checkpoint --overwrite_output_dir path/to/save/results
```

对单张图片进行模型推理的代码示例：
```bash
python test.py --cfg_file ./exp/sp2pII-phase2.yaml --test_img path/to/your/test/image
--ckpt path/to/your/phase2/checkpoint --overwrite_output_dir path/to/save/result
```

### 4.2 one/zero-shot推理
[test_sp2pII.py](test_sp2pII.py)需要指定使用的模型checkpoint、需要推理的文件夹、保存生成结果的文件夹、参考风格图或参考风格文本、推理设备。  
one-shot进行推理的代码示例：
```bash
python test_sp2pII.py --ckpt path/to/your/phase4/checkpoint --in_folder path/to/your/test/folder 
--out_folder path/to/save/results --img_prompt path/to/guide/image --device "cpu/cuda:x"
```

zero-shot进行模型推理的代码示例：
```bash
python test_sp2pII.py --ckpt path/to/your/phase4/checkpoint --in_folder path/to/your/test/folder 
--out_folder path/to/save/results --txt_prompt "your style prompt" --device "cpu/cuda:x"
```
