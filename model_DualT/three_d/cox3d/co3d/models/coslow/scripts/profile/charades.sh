#!/bin/bash
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

PROJECT=models/coslow
DATASET=charades
GPUS=1
PRECISION=32

# Run test sequence ######################

python $PROJECT/main.py \
    --id CoSlow_charades_clip_profile \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_charades.yaml \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode clip \
    --profile_model \


python $PROJECT/main.py \
    --id CoSlow_charades_frames_8_profile \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_charades.yaml \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --co3d_forward_mode frame \
    --profile_model \
  

python $PROJECT/main.py \
    --id CoSlow_charades_frames_64_profile \
    --dataset $DATASET \
    --seed 42 \
    --gpus $GPUS \
    --from_hparams_file models/coslow/hparams/slow_8x8_charades.yaml \
    --batch_size 1 \
    --benchmark True \
    --logging_backend wandb \
    --num_workers 4 \
    --precision $PRECISION \
    --temporal_window_size 64 \
    --co3d_forward_mode frame \
    --profile_model \

