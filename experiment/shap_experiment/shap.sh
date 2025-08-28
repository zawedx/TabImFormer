#!/bin/bash -l

uv pip uninstall adrd -y
uv pip install .

prefix="/home/two/LMTDE"
data_path="${prefix}/data/datasets/nacc_new/intra/nacc_newest.csv"
train_path="${prefix}/data/datasets/nacc_new/naccImg_train_normed.csv"
vld_path="${prefix}/data/datasets/nacc_new/naccImg_validation_normed.csv"
test_path="${prefix}/data/datasets/nacc_new/naccImg_validation_normed.csv"
cnf_file="${prefix}/data/datasets/nacc_new/meta/conf_mri.toml"
imgnet_ckpt="${prefix}/dev/ckpt/model_swinvit.pt"

# Note for setting the flags
# 1. If training without MRIs
# img_net="NonImg"
# img_mode = -1
# 2. if training with MRI embeddings
# img_net: [ViTEmb, DenseNetEMB, SwinUNETREMB, NonImg]
# img_mode = 1

# img_net="NonImg"
# img_mode=-1

img_net="SwinUNETREMB"
img_mode=1

ckpt_path="${prefix}/dev/ckpt/Final.pt"
gpt_embedding="${prefix}/data/datasets/nacc_new/meta/column_info.pkl"


emb_path=$gpt_embedding
emb_droprate=0.00
export SHAP="True"


python "${prefix}/experiment/shap_experiment/extract_head.py"


SHAP="True" torchrun --nproc_per_node=6 "${prefix}/dev/shap.py" --data_path $data_path --train_path $train_path --vld_path $vld_path --test_path $test_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
                --num_epochs 1000 --batch_size 21 --lr 0.004 --gamma 0 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" --imgnet_ckpt ${imgnet_ckpt} \
                --patch_size 16 --ckpt_path $ckpt_path --train_imgnet --cnf_file ${cnf_file} --train_path ${train_path} --vld_path ${vld_path} --data_path ${data_path}  \
                --fusion_stage middle --imgnet_layers 4 --weight_decay 0.0005 --save_intermediate_ckpts --emb_path ${emb_path} --parallel --emb_droprate $emb_droprate  \
                --contrastive --name "shap--contrastive"


python "${prefix}/experiment/shap_experiment/plotad.py"
python "${prefix}/experiment/shap_experiment/plotmci.py"