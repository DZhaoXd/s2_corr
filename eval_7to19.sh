#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

### Seen CS7;  Unseen: cityscapes19_val  outputs/gta5_eva_b16_512
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes_19.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"cityscapes19_val\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0019999.pth \
 $opts


### Seen CS7;  Unseen: ACDC_19  outputs/gta5_eva_b16_512
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes_19.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"ACDC_train_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0019999.pth \
 $opts


### Seen CS7;  Unseen: BDD_19  outputs/gta5_eva_b16_512
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes_19.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"bdd100k_val_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0019999.pth \
 $opts



### Seen CS7;  Unseen: Mapi_19  outputs/gta5_eva_b16_512
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes_19.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"mapillary_val_seg\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0019999.pth \
 $opts


cat $output/eval/log.txt | grep copypaste