conda activate DeCLIP_CATSeg

####Train on CS-7:
#@ Vit-B
CUDA_VISIBLE_DEVICES=1 nohup bash run.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512 > logs/cs7_eva_b16_r512.log 2>&1 &
#@ Vit-L
CUDA_VISIBLE_DEVICES=2 nohup bash run.sh configs/cs7_catseg_vitl.yaml 1 outputs/cs7_eva_L14_r448/ > logs/cs7_eva_L14_r448.log 2>&1 &


#### Train on GTA-7:
#@ Vit-B
CUDA_VISIBLE_DEVICES=3 nohup bash run.sh configs/gta5_seen7_catseg.yaml 1 outputs/gta_seen7_eva_b16_r512 > logs/gta_seen7_eva_b16_r512.log 2>&1 &
#@ Vit-L
CUDA_VISIBLE_DEVICES=4 nohup bash run.sh configs/gta5_seen7_catseg_vitl.yaml 1 outputs/gta_seen7_eva_L14_r448/ > logs/gta_seen7_eva_L14_r448.log 2>&1 &


#### Train on CS-19:
#@ Vit-L
CUDA_VISIBLE_DEVICES=5 nohup bash run.sh configs/cs19_catseg_vitl.yaml 1 outputs/cs19_eva_L14_r448/ > logs/cs19_eva_L14_r448.log 2>&1 &


#### Eval
CUDA_VISIBLE_DEVICES=6 nohup sh eval_7to50.sh configs/cs7_catseg.yaml 1 outputs/eval_7to50_cs7_eva_b16_r512/
CUDA_VISIBLE_DEVICES=7 nohup sh eval_7to19.sh configs/cs7_catseg_vitl.yaml 1 outputs/eval_7to19_cs7_eva_L14_r448/


#### Demo show
#segmentation results vis
CUDA_VISIBLE_DEVICES=2 nohup sh demo/vis.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512/ > logs/viz_cs7_eva_b16_r512.log 2>&1 &
#attention map vis
CUDA_VISIBLE_DEVICES=7 nohup sh demo/vis_atten.sh configs/cs7_catseg.yaml 1 outputs/cs7_eva_b16_r512/ > logs/viz_attention_cs7_eva_b16_r512.log 2>&1 &
