# TinyML Course Project 9 - How closely can W4A4 be brought to its FP16 counter part in terms of accuracy?  

**Students**: Sizhe Lester Li and Chonghyuk Andrew Song

### Abstract
In this project, we investigate the challenges and opportunities of applying W4A4 quantization to Large Language Models. We explore how different techniques, under two different paths (AWQ and Mixed-Precision), can close gaps from FP16. Our experiments show that both paths require non-trivial considerations of design choices. In AWQ, the original optimal search objective does not consider activation quantization. In Mixed-Precision, \todo{ASK andrew}. Ablatively, we analyze how different quantization methods result in different performances, assess our design choices, and investigate whether different layers of the model play different roles in performance. Our best model demonstrates a small gap, valued at 0.98, between W4A4 and FP16. We hope that our results and analyses will inspire future research on closing the gap between W4A4 and FP16 even further. We deeply thank the course staff for helping us understand the challenges in this area.   
    
## Environment Setup
This is a repository forked from the original [AWQ repo](https://github.com/andrewsonga/llm-awq). Here, we attach their installation guides:

### New Files
Friendly disclaimer: the names of these files are not representative of the efforts of each member. 

```
--awq
   |
   |
   |---quantize
          |
          |------auto_clip_lester.py
          |------auto_scale_lester.py
          |------fake_quant_lester.py
          |------pre_quant_lester.py
   |----examples
          |
          |------andrew_test_mixed_precision.ipynb
          |------lester_test_activation_quant.ipynb

```

### Install

1. Clone this repository and navigate to AWQ folder
```
git clone https://github.com/andrewsonga/llm-awq.git
cd llm-awq
```

2. Install Package
```
conda create -n awq python=3.10 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

* For **edge devices** like Orin, before running the commands above, please:

    1. Modify [pyproject.toml](pyproject.toml) by commenting out [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#L17).
    2. Set [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#18) to transformers==4.32.0.
    3. Manually install precompiled PyTorch binaries (>=2.0.0) from [NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
    4. Set the appropriate Python version for conda environment (e.g., `conda create -n awq python=3.8 -y` for JetPack 5).
  
3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```
cd awq/kernels
python setup.py install
```
