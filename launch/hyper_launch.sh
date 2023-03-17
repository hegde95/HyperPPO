
#!/bin/bash


# run the line to run this bash script
# ./launch/hyper_launch.sh <ENV> <EXP> <GPU_DEVICE> <WANDB_TAG(Optional)>

WANDB_TAG=${4:-"NONE"}
WITH_WANDB=False

if [ $WANDB_TAG != "NONE" ]; then
    WITH_WANDB=True
fi


export CUDA_VISIBLE_DEVICES=$3 && python -m sf_examples.brax.train_hyper_brax --env $1 --experiment $2 --seed $((111*$3)) --wandb_tag $WANDB_TAG --with_wandb $WITH_WANDB