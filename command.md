CUDA_VISIBLE_DEVICES=3 python3 tools/train_net.py --config-file ./configs/Market1501/sbs_R50-ibn.yml --num-gpus 1


CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 MODEL.BACKBONE.PRETRAIN_PATH '/mnt21t/home/mm2022/fast-reid/weights/lupws_r50.pth' OUTPUT_DIR logs/MarketT/Sbaseline_R50_lupNL

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/bap_R50.yml --num-gpus 2 

CUDA_VISIBLE_DEVICES=1,2 python3 tools/train_net.py --config-file ./configs/MarketT/WIP4.yml --num-gpus 2

CUDA_VISIBLE_DEVICES=3 python3 tools/train_net.py --config-file ./configs/MarketT/WIP5.yml --num-gpus 1

# run multi-process

CUDA_VISIBLE_DEVICES=4,5 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50198 MODEL.META_ARCHITECTURE PartSbaseline OUTPUT_DIR logs/MarketT_Quick/PartSbaseline2 MODEL.HEADS.NUM_CHUNK 2 SOLVER.IMS_PER_BATCH 128 TEST.EVAL_PERIOD 4

CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50199 MODEL.META_ARCHITECTURE PartSbaseline OUTPUT_DIR logs/MarketT_Quick/PartSbaseline3 MODEL.HEADS.NUM_CHUNK 3 TEST.EVAL_PERIOD 4

CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50197 MODEL.META_ARCHITECTURE PartSbaseline OUTPUT_DIR logs/MarketT_Quick/PartSbaseline4 MODEL.HEADS.NUM_CHUNK 4 TEST.EVAL_PERIOD 4

# attn output 
CUDA_VISIBLE_DEVICES=1,0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --eval-only ATTN_OUTPUT True MODEL.WEIGHTS path_to_weight

修改base -> python draw_attention.py  

# mismatch output
CUDA_VISIBLE_DEVICES=1 python3 tools/train_net.py --config-file ./configs/MarketT/WIP.yml --num-gpus 1 --eval-only MISMTACH_OUTPUT True MODEL.WEIGHTS path_to_weight

CUDA_VISIBLE_DEVICES=1,2 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/MultiStepLR1 MODEL.LOSSES.TRI.HARD_MINING False

CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/ATTN/8 MODEL.HEADS.ATTN 8

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_GAN.yml --num-gpus 2 --dist-url 'tcp://127.0.0.1:50178' OUTPUT_DIR logs/MarketT/TEST

CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --eval-only --dist-url 'tcp://127.0.0.1:50178' OUTPUT_DIR logs/MarketT/TEST


CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 OUTPUT_DIR logs/MarketT/TEST

TORCH_DISTRIBUTED_DEBUG=info

CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50188 OUTPUT_DIR logs/MarketT/STAGE/-ATTACH_ACTIVATION MODEL.BACKBONE.ATTACH_ACTIVATION False

CUDA_VISIBLE_DEVICES=4,5 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50189 OUTPUT_DIR logs/MarketT/STAGE/-ATTNSHAPED_ACTIVATION MODEL.BACKBONE.ATTNSHAPED_ACTIVATION False


CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50188 OUTPUT_DIR logs/MarketT/GanSbaseline MODEL.META_ARCHITECTURE GanSbaseline

CUDA_VISIBLE_DEVICES=4 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50134 OUTPUT_DIR logs/MarketT_Quick/RPT 

CUDA_VISIBLE_DEVICES=6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50198 --eval-only OUTPUT_DIR logs/MarketT_Quick/TEST MODEL.WEIGHTS /home/ckl/fast-reid/logs/MarketT_Quick/Sbaseline_R50/model_final.pth

CUDA_VISIBLE_DEVICES=7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50666 OUTPUT_DIR logs/MarketT_Quick/STAGE/SBaseline

CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50677 OUTPUT_DIR logs/MarketT_Quick/STAGE/120_MS_LOSS MODEL.LOSSES.NAME "('CrossEntropyLoss','MultiSimilarityLoss')" MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50655 --eval-only OUTPUT_DIR logs/MarketT_Quick/STAGE/120_SE_TEST MODEL.BACKBONE.WITH_SE True SOLVER.MAX_EPOCH 120 MODEL.BACKBONE.DEPTH 101x MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/se_resnet101_ibn_a-fabed4e2.pth MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN MODEL.WEIGHTS /home/ckl/fast-reid/logs/MarketT_Quick/STAGE/120_SE/model_best.pth 

CUDA_VISIBLE_DEVICES=4,5 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50688 OUTPUT_DIR logs/MarketT_Quick/STAGE/120_SBaseline MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50865 OUTPUT_DIR logs/MarketT_Quick/STAGE/120_SE_Last_not_only MODEL.BACKBONE.WITH_SE True SOLVER.MAX_EPOCH 120 MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/se_resnet101_ibn_a-fabed4e2.pth SOLVER.IMS_PER_BATCH 128 MODEL.LAST_EMBEDDING True MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN MODEL.BACKBONE.DEPTH 101x



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50666 OUTPUT_DIR logs/MarketT_Quick/STAGE/120_Resnext MODEL.BACKBONE.DEPTH 101x SOLVER.IMS_PER_BATCH 128 MODEL.BACKBONE.NAME build_resnext_backbone MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/resnext101_ibn_a-6ace051d.pth MODEL.BACKBONE.DEPTH 101x 

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50666 OUTPUT_DIR logs/MarketT_Quick/STAGE/ATTN32 MODEL.HEADS.ATTN 32 SOLVER.IMS_PER_BATCH 128 SOLVER.MAX_EPOCH 120

CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 OUTPUT_DIR logs/MarketT_Quick/STAGE/512*256 INPUT.SIZE_TEST [512,256] INPUT.SIZE_TRAIN [512,256] MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN SOLVER.MAX_EPOCH 120



CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50677 OUTPUT_DIR logs/MarketT_Quick/LAST_ONLY MODEL.LAST_EMBEDDING True SOLVER.IMS_PER_BATCH 128

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50676 OUTPUT_DIR logs/MarketT_Quick/LAST_NOT_ONLY MODEL.LAST_EMBEDDING True SOLVER.IMS_PER_BATCH 128 MODEL.LAST_ONLY False


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50678 OUTPUT_DIR logs/MarketT_Quick/Multi_STEP SOLVER.IMS_PER_BATCH 256

CUDA_VISIBLE_DEVICES=0 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50678 OUTPUT_DIR logs/MarketT_Quick/Multi_STEP_2048 MODEL.BACKBONE.EMBEDDING_DIM 2048 &

CUDA_VISIBLE_DEVICES=1 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50178 OUTPUT_DIR logs/MarketT_Quick/Multi_STEP_1024 MODEL.BACKBONE.EMBEDDING_DIM 1024 &

CUDA_VISIBLE_DEVICES=2 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50278 OUTPUT_DIR logs/MarketT_Quick/Multi_STEP_512 MODEL.BACKBONE.EMBEDDING_DIM 512 &

CUDA_VISIBLE_DEVICES=0 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50378 OUTPUT_DIR logs/MarketT_Quick/Cos_1024_90 SOLVER.MAX_EPOCH 90 &

CUDA_VISIBLE_DEVICES=1 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50578 OUTPUT_DIR logs/MarketT_Quick/Cos_1024_120 SOLVER.MAX_EPOCH 120 &

CUDA_VISIBLE_DEVICES=2 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50878 OUTPUT_DIR logs/MarketT_Quick/Cos_1024_60 SOLVER.MAX_EPOCH 60 &

CUDA_VISIBLE_DEVICES=2 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50888 OUTPUT_DIR logs/MarketT_Quick/SbaselineTT SOLVER.MAX_EPOCH 60 MODEL.META_ARCHITECTURE SbaselineT

CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50666 OUTPUT_DIR logs/MarketT_Quick/SbaselineTT2 SOLVER.MAX_EPOCH 60 MODEL.META_ARCHITECTURE SbaselineT2

CUDA_VISIBLE_DEVICES=4 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:50555 OUTPUT_DIR logs/MarketT_Quick/STEP_70_90_120

CUDA_VISIBLE_DEVICES=3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:56668 OUTPUT_DIR logs/MarketT_Quick/SGD_0.001 SOLVER.OPT SGD SOLVER.BASE_LR 0.001

CUDA_VISIBLE_DEVICES=5 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51234 OUTPUT_DIR logs/MarketT_Quick/repvgg MODEL.META_ARCHITECTURE RepVggSbaseline MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/RepVGG-B2g4-200epochs-train.pth MODEL.BACKBONE.DEPTH B2g4 MODEL.BACKBONE.NAME build_repvgg_backbone MODEL.BACKBONE.FEAT_DIM 2560

CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51554 OUTPUT_DIR logs/MarketT_Quick/repvgg_b1g2 MODEL.META_ARCHITECTURE RepVggSbaseline MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/RepVGG-B1g2-train.pth MODEL.BACKBONE.DEPTH B1g2 MODEL.BACKBONE.NAME build_repvgg_backbone MODEL.BACKBONE.FEAT_DIM 2048

CUDA_VISIBLE_DEVICES=2,1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 2 --dist-url tcp://127.0.0.1:50668 OUTPUT_DIR logs/MarketT_Quick/STAGE/120_SE_ML_STEP MODEL.BACKBONE.WITH_SE True SOLVER.MAX_EPOCH 160 MODEL.BACKBONE.DEPTH 101x MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/se_resnet101_ibn_a-fabed4e2.pth MODEL.BACKBONE.NORM syncBN MODEL.HEADS.NORM syncBN 

CUDA_VISIBLE_DEVICES=3 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51559 OUTPUT_DIR logs/MarketT_Quick/repvgg_a2 MODEL.META_ARCHITECTURE RepVggSbaseline MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/RepVGG-A2-train.pth MODEL.BACKBONE.DEPTH A2 MODEL.BACKBONE.NAME build_repvgg_backbone MODEL.BACKBONE.FEAT_DIM 1408


CUDA_VISIBLE_DEVICES=6 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:59987 OUTPUT_DIR logs/MarketT_Quick/repvgg_B2g4 MODEL.META_ARCHITECTURE RepVggSbaseline MODEL.BACKBONE.PRETRAIN_PATH /home/ckl/fast-reid/weights/RepVGG-B2g4-train.pth MODEL.BACKBONE.DEPTH B2g4 MODEL.BACKBONE.NAME build_repvgg_backbone MODEL.BACKBONE.FEAT_DIM 2560

CUDA_VISIBLE_DEVICES=0 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51234 OUTPUT_DIR logs/MarketT_Quick/MS_60_80_100 &

CUDA_VISIBLE_DEVICES=3 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51668 OUTPUT_DIR logs/MarketT_Quick/MS_50_70_90 &

CUDA_VISIBLE_DEVICES=4 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51666 OUTPUT_DIR logs/MarketT_Quick/MS_50_65_80 &

CUDA_VISIBLE_DEVICES=5 nohup python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51888 OUTPUT_DIR logs/MarketT_Quick/MS_50_70 &




CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51888 OUTPUT_DIR logs/MarketT_Quick/TEST SOLVER.MAX_EPOCH 60 MODEL.META_ARCHITECTURE SbaselineT

CUDA_VISIBLE_DEVICES=1 python3 tools/train_net.py --config-file ./configs/MarketT/SBaseline_Quick.yml --num-gpus 1 --dist-url tcp://127.0.0.1:51889 OUTPUT_DIR logs/MarketT_Quick/TEST_WO_NL SOLVER.MAX_EPOCH 60 MODEL.META_ARCHITECTURE SbaselineT MODEL.BACKBONE.WITH_NL False