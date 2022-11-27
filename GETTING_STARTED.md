# Getting Started

## Prepare pretrained model

If you want to re-train MDPR, you can download lup_moco_r101.pth in [BaiduYun](https://pan.baidu.com/s/17DgQtqwGyOqgwTr7F9ztpg) and can put it in [weights/]. (password：mdpr)

If you want to eval MDPR, you can download our pretrained weights in [BaiduYun](https://pan.baidu.com/s/17DgQtqwGyOqgwTr7F9ztpg) and put them in [weights/]. (password：mdpr)

## Compile with cython to accelerate evalution

```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Training & Evaluation in Command Line

we provide the shell scripts to train and evaluate MDPR on DukeMTMC-reID.

```bash
# 1 GPUs
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file configs/MDPR/DukeMTMC.yml OUTPUT_DIR logs/MDPR/DukeMTMC 

# 4 GPUs (the same setting as in our paper)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file configs/MDPR/DukeMTMC.yml --num-gpus 4 SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 16 OUTPUT_DIR logs/MDPR/DukeMTMC
```

To evaluate a model's performance, use

```bash
# 1 GPUs
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --eval-only --config-file configs/MDPR/DukeMTMC.yml OUTPUT_DIR logs/MDPR/DukeMTMC MODEL.WEIGHTS weights/DukeMTMC-reID.pth

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --eval-only --config-file configs/MDPR/DukeMTMC.yml --num-gpus 4 OUTPUT_DIR logs/MDPR/DukeMTMC MODEL.WEIGHTS weights/DukeMTMC-reID.pth
```

To visualize the retrieval results, use

```bash
CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py --config-file configs/MDPR/DukeMTMC.yml --parallel --vis-label --dataset-name DukeMTMC --output logs/MDPR/DukeMTMC/vis --opts MODEL.WEIGHTS weights/DukeMTMC-reID.pth
```

To visualize the attention headmap, use

```bash
# first, eval and save the attention map
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --eval-only --config-file configs/MDPR/DukeMTMC.yml OUTPUT_DIR logs/MDPR/DukeMTMC/ MODEL.WEIGHTS weights/DukeMTMC-reID.pth OUTPUT_ALL True

# second, visualize and save the attention map
python3 demo/visualize_attn.py --base=logs/MDPR/DukeMTMC/heatmap_attn
```

## acknolegement
This resipotry is based on [fast-reid](https://github.com/JDAI-CV/fast-reid).
