# AttentionGuied
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+AttentionGuied SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0.INPUT.RPT.ENABLED False INPUT.REA.ENABLED False MODEL.BACKBONE.WITH_IBN False MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.LAST_STRIDE 2
# + Stride=1
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+Stride=1 SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0. INPUT.RPT.ENABLED False INPUT.REA.ENABLED False MODEL.BACKBONE.WITH_IBN False MODEL.BACKBONE.WITH_NL False
# + NL
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+NL SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0. INPUT.RPT.ENABLED False INPUT.REA.ENABLED False MODEL.BACKBONE.WITH_IBN False
# + IBN
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+IBN SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0. INPUT.RPT.ENABLED False INPUT.REA.ENABLED False
# + REA
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+REA SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0. INPUT.RPT.ENABLED False
# + RPT
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+RPT SOLVER.WARMUP_ITERS 0 MODEL.LOSSES.CE.EPSILON 0. 
# + LS
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/+LS SOLVER.WARMUP_ITERS 0  
# + WarmUp(default)
# CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/STAGE/SBaseline  


# NORM_FEAT before after(default)
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/NORM_FEAT/NOT_BNNECK MODEL.HEADS.WITH_BNNECK False
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/NORM_FEAT/before MODEL.HEADS.NECK_FEAT before
# (default) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/NORM_FEAT/after MODEL.HEADS.NECK_FEAT after

# BatchDropBlock H_W
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BatchDropBlock/0.15_1.0 MODEL.DROP.ENABLED True

# LOSS_RATIO
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/LOSS_RATIO/ATTN_SCALE MODEL.LOSSES.CE.ATTN_SCALE 2.0
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/LOSS_RATIO/MAIN_SCALE MODEL.LOSSES.CE.MAIN_SCALE 2.0
# (TODO) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/LOSS_RATIO/MAIN_SCALE*10 MODEL.LOSSES.CE.MAIN_SCALE 10.0

# TripletMargin 0(default) 0.1 0.3 0.5
# (default) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TripletMargin/0  
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TripletMargin/0.1 MODEL.LOSSES.TRI.MARGIN 0.1
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TripletMargin/0.3 MODEL.LOSSES.TRI.MARGIN 0.3
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TripletMargin/0.5 MODEL.LOSSES.TRI.MARGIN 0.5
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TripletMargin/NORM_FEAT MODEL.LOSSES.TRI.NORM_FEAT True


# MODEL.HEADS.POOL_LAYER GlobalAvgPool GlobalMaxPool AdaptiveAvgMaxPool GeneralizedMeanPoolingP(default)
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/POOL_LAYER/GlobalAvgPool MODEL.HEADS.POOL_LAYER GlobalAvgPool
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/POOL_LAYER/GlobalMaxPool MODEL.HEADS.POOL_LAYER GlobalMaxPool
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/POOL_LAYER/AdaptiveAvgMaxPool MODEL.HEADS.POOL_LAYER AdaptiveAvgMaxPool
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/POOL_LAYER/GeneralizedMeanPoolingP MODEL.HEADS.POOL_LAYER GeneralizedMeanPoolingP

# BACKBONE OSNet ResNest50 ResNext101 ResNet101
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BACKBONE/ResNest50 MODEL.BACKBONE.NAME build_resnest_backbone

# ARCHITECTURE Sbaseline(default) Sbase Baseline
# (default) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ARCHITECTURE/Sbaseline MODEL.META_ARCHITECTURE Sbaseline
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ARCHITECTURE/Sbase MODEL.META_ARCHITECTURE Sbase
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ARCHITECTURE/Baseline MODEL.META_ARCHITECTURE Baseline


# ATTN 4 8 16 32(default) 48 64
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/4 MODEL.HEADS.ATTN 4
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/8 MODEL.HEADS.ATTN 8
# (default) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/32 MODEL.HEADS.ATTN 32
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/48 MODEL.HEADS.ATTN 48
# (TODO) CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/64 MODEL.HEADS.ATTN 64


# BACKBONE.EMBEDDING_DIM 256 512(default) 1024 2048 
# (TODO) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BACKBONE_EMBEDDING_DIM/2048 MODEL.BACKBONE.EMBEDDING_DIM 2048
# (TODO) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BACKBONE_EMBEDDING_DIM/1024 MODEL.BACKBONE.EMBEDDING_DIM 1024
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BACKBONE_EMBEDDING_DIM/512 MODEL.BACKBONE.EMBEDDING_DIM 512
# CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/BACKBONE_EMBEDDING_DIM/256 MODEL.BACKBONE.EMBEDDING_DIM 256


# TRAIN_SIZE 256*128 224*224 384*128(default) 384*192
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TRAIN_SIZE/256*128 INPUT.SIZE_TEST [256,128] INPUT.SIZE_TRAIN [256,128]
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TRAIN_SIZE/224*224 INPUT.SIZE_TEST [224,224] INPUT.SIZE_TRAIN [224,224]
# (default) CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TRAIN_SIZE/384*128 INPUT.SIZE_TEST [384,128] INPUT.SIZE_TRAIN [384,128]
# (TODO) CUDA_VISIBLE_DEVICES=1,2 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TRAIN_SIZE/384*192 INPUT.SIZE_TEST [384,192] INPUT.SIZE_TRAIN [384,192]


# K*N 8*4 8*8 16*4(default) 16*8 32*4 32*8
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/8*4 SOLVER.IMS_PER_BATCH 32 DATALOADER.NUM_INSTANCE 4
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/8*8 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 8
# (default) CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/16*4 SOLVER.IMS_PER_BATCH 64 DATALOADER.NUM_INSTANCE 4
# (TODO) CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/16*8 SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 8
# (TODO) CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/32*4 SOLVER.IMS_PER_BATCH 128 DATALOADER.NUM_INSTANCE 4
# (TODO) CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/K*N/32*8 SOLVER.IMS_PER_BATCH 256 DATALOADER.NUM_INSTANCE 8


# PRETRAIN ImageNet(default) Lup LupNL Lup_No_IBN LupNL_No_IBN
# (default) CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/PRETRAIN/ImageNet MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/.cache/torch/checkpoints/resnet50-19c8e357.pth'
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/PRETRAIN/Lup MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/fast-reid/weights/lup_moco_r50.pth'
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/PRETRAIN/LupNL MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/fast-reid/weights/lupws_r50.pth'
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/PRETRAIN/Lup_No_IBN MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/fast-reid/weights/lup_moco_r50.pth' MODEL.BACKBONE.WITH_IBN False
# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/PRETRAIN/LupNL_No_IBN MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/fast-reid/weights/lupws_r50.pth' MODEL.BACKBONE.WITH_IBN False

# SingleBranchEval b1 b2 b3 b4 (b1+b2+b3)/3 [b1,b2,b3] (b1+b2+b3+b4)/4 [b1,b2,b3,b4](default)


# DATASET Market1501(default) dukeMTMC cuhk03
# cuhk03 https://github.com/zhunzhong07/person-re-ranking
