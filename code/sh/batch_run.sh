DATA_ROOT=/mnt/data/ECCV_release_sample
OUTPUT_ROOT=/home/ubuntu/yaoyao/codes/neilf_eccv_release/model
GPU_INDEX=$2

case $1 in
synthetic)
DATASET=synthetic
for SCENE in city studio castel city_mix studio_mix castel_mix; do
    # CUDA_VISIBLE_DEVICES=$GPU_INDEX python training/train.py \
    #     $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
    #     --config_path configs/config_synthetic_data.json
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python evaluation/evaluate.py \
        $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
        --config_path configs/config_synthetic_data.json --eval_brdf --eval_nvs --export_brdf
done
;;
DTU)
DATASET=DTU
for SCENE in scan1 scan11 scan24 scan37 scan73 scan75 scan97 scan110 scan114; do
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python training/train.py \
        $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
        --config_path configs/config_dtu_data.json
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python evaluation/evaluate.py \
        $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
        --config_path configs/config_dtu_data.json --eval_nvs
done
;;
BlendedMVS)
DATASET=BlendedMVS
for SCENE in bull camera dog gold statue stone; do
    # CUDA_VISIBLE_DEVICES=$GPU_INDEX python training/train.py \
    #     $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
    #     --config_path configs/config_blendedmvs_data.json
    CUDA_VISIBLE_DEVICES=$GPU_INDEX python evaluation/evaluate.py \
        $DATA_ROOT/${DATASET}_$SCENE $OUTPUT_ROOT/${DATASET}_$SCENE \
        --config_path configs/config_blendedmvs_data.json --eval_nvs --export_brdf
done
;;
*)
;;
esac