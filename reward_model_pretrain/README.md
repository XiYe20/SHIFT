# Reward model pretraining for MotionAlignmentVDM
The Steps to pretrain the ic and lc motion reward models

## Usage

### Step 1. Generate the fake videos using pretrained base video models

#### SVD Example Generation
Can be used to generate video examples by SVD for training the LC discriminator, or for SVD performance evaluation.

1. Given a directory of real videos, read each real video, and extract the first frame as input for SVD.
2. Generate num_example_per_video videos for each initial frame (from a real video) by SVD.
3. Save the generated fake and real videos as pt files for LC discriminator training dataset. Or save the generated videos as video files for SVD performance evaluation.

```bash
python svd_example_generation.py --gpu_ids "0,1,2" --output_dir ./svd_examples --real_video_dir ./real_videos
```

### LC Discriminator Training
Training the LC discriminator with the generated fake and real videos.

1. Modify the config file to set the training parameters.
2. Run the following command to start training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 accelerate launch --num_processes 10 --main_process_port 29512 train_lc_discriminator.py --config /path/to/MotionAlignmentVDM/configs/lc_discriminator_train_config.yaml
```

