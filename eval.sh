torchrun --nproc_per_node=1 train.py --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth --model-type vit_l --output work_dirs/pa_sam_l --eval --restore-model work_dirs/pa_sam_l/epoch_20.pth
