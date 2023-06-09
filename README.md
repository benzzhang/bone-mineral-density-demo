<!--
 * @Author: Zhang Jian
 * @Date: 2023-02-16
 * @Description: 
 * 
-->
## Vertebra Segmentation for Chest CT Image

### Usage
1. training on single gpu:
```bash
 python train_fp16.py --config-file your_config_file --gpu-id '0'
```
e.g. ```python train_fp16.py --config-file experiments/template/config.yaml --gpu-id '0'```


2. training on multiple gpus (nodes):
```bash
python train_fp16.py
```

3. inferring:
```bash
python train_fp16.py --infer
```