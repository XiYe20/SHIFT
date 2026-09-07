CUDA_VISIBLE_DEVICES=0,1 python src/train.py --config train_config.yaml
CUDA_VISIBLE_DEVICES=0,1 python /path/to/eval_phyvid.py --master_port 12483
