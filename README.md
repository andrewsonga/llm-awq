# TinyML Course Project 9 - How closely can W4A4 be brought to its FP16 counter part in terms of accuracy?  

**Students**: Sizhe Lester Li and Chonghyuk Andrew Song

### Abstract (OOD)
In this project, we investigate the challenges and opportunities of applying W4A4 quantization to Large Language Models (LLMs). We explore how different techniques, under two different paths (mixed-precision approach and AWQ), can close gaps from FP16. Our experiments show that both approaches require non-trivial considerations of design choices. For the mixed-precision approach, we find the per-channel activation quantization is essential to achieving comparable accuracy to the Full FP-16 model, even when we perform salient weight and activation protection. In AWQ, the original optimal search objective does not consider activation quantization, and hence needs to be modified accordingly. With respect to the unquantized FP16 model, our best model (W4A4 + 1\% FP16) demonstrates a performance gap of 1.11, 1.17, 1.29 for OPT 1.3B, 2.7B, and 6.7B, respectively. We conduct a variety of experiments to validate our design choices and present a series of insights that will hopefully inspire future work on W4A4 quantization. We deeply thank the course staff for helping us understand the challenges in this area. 
    
## Environment Setup
This is a repository forked from the original [AWQ repo](https://github.com/andrewsonga/llm-awq). Here, we attach their installation guides:

### New Files

```
--awq
   |
   |
   |---quantize
          |
          |------auto_clip_new.py
          |------auto_scale_new.py
          |------fake_quant_new.py
          |------pre_quant_new.py
          |------wnan_salient.py
   |----examples
          |
          |------test_mixed_precision.ipynb
          |------test_activation_quant.ipynb
    |---scripts
          |
          |---tiny_ml_project
                   |
                   |
                   |-------test_awq.py

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

### Reproducing experiments

#### AWQ Experiments

(1) To perform evaluation and save awq results to local storage:
```{bash}
CUDA_VISIBLE_DEVICES=7 python3 test_awq.py --model_kwrd opt-1.3b --run_awq --save_awq_result --awq_path awq_results.pt
```

(2) To load `awq_results.pt` that is locally stored and perform evaluation:
```{bash}
CUDA_VISIBLE_DEVICES=7 python3 test_awq.py --model_kwrd opt-1.3b  --load_awq_result --awq_path awq_results.pt
```

(3) By default, we perform W4A4 with per-channel activation quantization, but we can modify it by
```{bash}
CUDA_VISIBLE_DEVICES=7 python3 test_awq.py --model_kwrd opt-1.3b  --load_awq_result --awq_path awq_results.pt --w_n_bits 8 --a_n_bits 8 --q_group_size --act_quant {per_token/per_tensor//per_channel/none}
```
, where `q_group_size` governs the groun size for weight quantization.


(4) To evaluate the original AWQ objective against the new objective, run:
```{bash}
CUDA_VISIBLE_DEVICES=7 python3 test_awq.py --model_kwrd opt-1.3b  --load_awq_result --awq_path awq_results.pt --act_quant none --act_quant_override per_channel
```
This will result in AWQ using using "none" activation quantization, defaulting to the original objective. However, per_channel activation quantization will still be performed on the inference model after AWQ results are applied.