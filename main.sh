# Sec 5.2
python main.py \
    --inet-dir /home/public/ImageNet \  # change to your ImageNet directory
    --gpu 6 7 \
    --expr rep_recover_acc \
    --batch-size 256 \
    --num-samples-per-class 5 \
    --num-batches 5 \
    --arch resnet50 \
    --eps 3     # 0 for vanilla and 3 for AT

# Sec 5.3
python main.py \
    --inet-dir /home/public/ImageNet \  # change to your ImageNet directory
    --gpu 6 7 \
    --expr img_recover \
    --arch resnet50 \
    --eps 3 \
    --batch-size 256 \
    --opt adam \
    --lr 0.1 \
    --steps 5000 \
    --tv-l2-reg 1e-6 \
    --restarts 5 \
    --use-best