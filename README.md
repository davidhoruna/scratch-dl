


```
pip install -e . 

python train.py \
    --model_name resnet50 \
    --folder_structure ImageFolder \
    --folder_name PokemonData \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-5 \
    --val_split 0.1 \
    --amp \
    --save_checkpoints \
    --checkpoint_dir ./Poke \
    --project pokemon-classification \

```