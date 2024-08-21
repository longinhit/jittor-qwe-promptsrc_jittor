export DISABLE_MULTIPROCESSING=1

seed=3407
root_path=./data
backbone=./jclip/ViT-B-32.pkl

mkdir caches
mkdir save_model
python -u extract_features.py \
    --root_path $root_path

python -u PromptSRCCLIP.py \
    --seed $seed \
    --root_path $root_path \
    --backbone $backbone \
    --shots 4 \
    --lr 0.0001 \
    --eps 0.001 \
    --train_epoch 50 