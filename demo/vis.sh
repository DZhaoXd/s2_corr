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



# mapillary_show_seg_30
#python vis.py --config $config \
# --num-gpus $gpus \
# --dist-url "auto" \
# --eval-only \
# OUTPUT_DIR $output/vis/mapillary_show_seg_30/ \
# MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/mapi_30.json" \
# MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
# DATASETS.TEST \(\"mapillary_show_seg_30\"\,\) \
# TEST.SLIDING_WINDOW "True" \
# MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
# MODEL.WEIGHTS $output/model_0014999.pth \
# $opts

# roadwork_semseg_val_show
python vis.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/vis/roadwork_semseg_val_show/ \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/roadwork_14.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"roadwork_semseg_val_show\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0014999.pth \
 $opts


# acdc_inpaint41_test
python vis.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/vis/acdc_inpaint41_test/ \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/inpaint41_vocab.json" \
 MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
 DATASETS.TEST \(\"acdc_inpaint41_test\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_0014999.pth \
 $opts

# bdd_inpaint41_test
#python vis.py --config $config \
# --num-gpus $gpus \
# --dist-url "auto" \
# --eval-only \
# OUTPUT_DIR $output/vis/bdd_inpaint41_test/ \
# MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/inpaint41_vocab.json" \
# MODEL.SEM_SEG_HEAD.ENABLE_HIGH_RES "False"  \
# DATASETS.TEST \(\"bdd_inpaint41_test\"\,\) \
# TEST.SLIDING_WINDOW "True" \
# MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
# MODEL.WEIGHTS $output/model_0014999.pth \
# $opts


cat $output/eval/log.txt | grep copypaste