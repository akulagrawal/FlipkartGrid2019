INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=model/ssd_mobilenet_v1_pets.config
TRAINED_CKPT_PREFIX=model/model.ckpt-5000
EXPORT_DIR=trained_model/
python3 export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
