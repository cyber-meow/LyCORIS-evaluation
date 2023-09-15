#!/bin/bash

# Define common variables
REF_DIR="/media/bangbang/howdy/lycoris_paper/lyco-finetune-local/ref_features"
EVAL_DIR="/media/bangbang/howdy/lycoris_paper/lyco-finetune-local/main/generated_features/reweight/"
METRIC_CSV="../../results/metrics/reweight_metrics.csv"

# Run each python script with the appropriate parameters
python eval_image_similarity.py --ref_dir $REF_DIR --eval_dir $EVAL_DIR --encoder_name clip-L-14 --metric_csv $METRIC_CSV --extra_level 2
python eval_image_similarity.py --ref_dir $REF_DIR --eval_dir $EVAL_DIR --encoder_name convnextv2-l --metric_csv $METRIC_CSV --extra_level 2
python eval_image_similarity.py --ref_dir $REF_DIR --eval_dir $EVAL_DIR/nt_correct --encoder_name dinov2-l-fb --metric_csv $METRIC_CSV --extra_level 2
python eval_image_diversity.py --eval_dir $EVAL_DIR --encoder_name dinov2-l-fb --metric_csv $METRIC_CSV --extra_level 2
python eval_image_diversity.py --eval_dir $EVAL_DIR --encoder_name clip-L-14 --metric_csv $METRIC_CSV --extra_level 2
python eval_image_diversity.py --eval_dir $EVAL_DIR --encoder_name convnextv2-l --metric_csv $METRIC_CSV --extra_level 2
python eval_text_similarity.py --ref_dir $REF_DIR --eval_dir $EVAL_DIR --metric_csv $METRIC_CSV --extra_level 2
