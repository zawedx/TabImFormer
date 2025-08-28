# import sys
# sys.path.append('..')
import hashlib
import time
from IPython import embed
from SimpleITK import Rank
import pandas as pd
from sympy import true
import torch
import json
import argparse
import os
import monai
import nibabel as nib
import wandb
import numpy as np
import random
import sys

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel, GMModel, SCLAD# , MGDA #
from tqdm import tqdm
from collections import defaultdict
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.disable()
from torchvision import transforms
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Resized,
)
from shap_impl import collect_shap_value_pairs
# `my_trainer` is your trainer instance
# `ldr_vld` is your DataLoader with 512 samples

def parser():
    parser = argparse.ArgumentParser("Transformer pipeline", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--data_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the entire data.')
    parser.add_argument('--train_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--vld_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the validation data.')
    parser.add_argument('--test_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the testing data.')
    parser.add_argument('--cnf_file', default='/home/skowshik/ADRD_repo/adrd_tool/dev/data/toml_files/default_nacc_revised_labels.toml', type=str,
        help='Please specify path to the configuration file.')
    parser.add_argument('--img_mode', type=int, choices=[-1, 0, 1, 2])
    parser.add_argument('--img_net', type=str, choices=['ViTAutoEnc', 'ViTEMB', 'DenseNet', 'DenseNetEMB', 'SwinUNETR', 'SwinUNETREMB', 'NonImg'])
    parser.add_argument('--imgnet_ckpt', type=str, help="Path to Imaging model checkpoint")
    parser.add_argument('--fusion_stage', type=str, default="middle", help="Fusion stage of the image embeddings")
    parser.add_argument('--train_imgnet', action="store_true", help="Set to True to train imaging model along transformer.")
    parser.add_argument("--img_size", type=str, help="input size to the imaging model")
    parser.add_argument("--imgnet_layers", type=int, default=2, help="Number of layers of the downsampling block.")
    parser.add_argument("--patch_size", type=int, help="patch size")
    parser.add_argument('--ckpt_path', default='/home/skowshik/ADRD_repo/adrd_tool/dev/ckpt/revised_labels/ckpt.pt', type=str,
        help='Please specify the ckpt path')
    parser.add_argument('--load_from_ckpt', action="store_true", help="Set to True to load model from checkpoint.")
    parser.add_argument('--save_intermediate_ckpts', action="store_true", help="Set to True to save intermediate model checkpoints.")
    parser.add_argument('--wandb', action="store_true", help="Set to True to init wandb logging.")
    parser.add_argument('--balanced_sampling', action="store_true", help="Set to True for balanced sampling.")
    parser.add_argument('--ranking_loss', action="store_true", help="Set to True to apply ranking loss.")
    parser.add_argument('--parallel', action='store_true', default=False, help='Set True for DP training.')
    parser.add_argument('--d_model', default=64, type=int,
        help='Please specify the dimention of the feature embedding')
    parser.add_argument('--nhead', default=1, type=int,
        help='Please specify the number of transformer heads')
    parser.add_argument('--num_epochs', default=256, type=int,
        help='Please specify the number of epochs')
    parser.add_argument('--batch_size', default=128, type=int,
        help='Please specify the batch size')
    parser.add_argument('--lr', default=1e-4, type=float,
        help='Please specify the learning rate')
    parser.add_argument('--gamma', default=2, type=float,
        help='Please specify the gamma value for the focal loss')
    parser.add_argument('--weight_decay', default=0.0, type=float,
        help='Please specify the weight decay (optional)')
    parser.add_argument('--emb_path', default=None, type=str,
        help='Please specify the path to the initiate embeddings, None indicates linear layer')
    parser.add_argument('--emb_droprate', default=0.2, type=float,
        help='Please specify the dropout rate in embedding layers')
    parser.add_argument('--num_trf_layers', default=2, type=int,
        help='Please specify the number of transformer layers')
    parser.add_argument('--temp', default=0.1, type=float,
        help='Please specify the temperature for contrastive learning')
    parser.add_argument('--clr_ratio', default=0.1, type=float,
        help='Please specify the ratio of the contrastive loss')
    parser.add_argument('--contrastive', action='store_true',
        help='Set True for contrastive learning')
    parser.add_argument('--gaussian', action='store_true',
        help='Set True for C-GMVAE')
    parser.add_argument('--mgda', action='store_true',
        help='Set True for MGDA_UB')
    parser.add_argument('--name', default=None, type=str,
        help='Please specify the model name, used for wandb logging')
    args = parser.parse_args()
    return args

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DDP
def init_DDP():
    dist.init_process_group(backend='nccl')
    global local_rank
    local_rank= int(os.environ["LOCAL_RANK"])
    global rank
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    set_random_seed(42)

#-----------------------------------

args = parser()
if args.parallel:
    init_DDP()
    if local_rank != 0:
        pass
        sys.stdout = open(os.devnull, 'w')
else:
    rank = 0
    local_rank = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# debug
if args.emb_path is None:
    raise("Please specify the path to the embeddings")
# debug end
img_size = eval(args.img_size)
print(f"Image backbone: {args.img_net}")
if args.img_net == 'None':
    args.img_net = None
    


save_path = '/'.join(args.ckpt_path.split('/')[:-1])
if not os.path.exists(save_path):
    os.makedirs(save_path)

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.8, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                # Resized(keys=["image"], spatial_size=img_size),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        try:
            image_data = data["image"]
            check = nib.load(image_data).get_fdata()
            if len(check.shape) > 3:
                return None
            return self.transforms(data)
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
trn_filter_transform = FilterImages(dat_type='trn')
vld_filter_transform = FilterImages(dat_type='vld')


# initialize datasets
seed = 0
stripped = '_stripped_MNI'
print("Loading training dataset ... ")
dat_trn = CSVDataset(dat_file=args.train_path, cnf_file=args.cnf_file, mode=0, img_mode=args.img_mode, arch=args.img_net, transforms=FilterImages('trn'), stripped=stripped)
print("Done.\nLoading Validation dataset ...")
dat_vld = CSVDataset(dat_file=args.vld_path, cnf_file=args.cnf_file, mode=1, img_mode=args.img_mode, arch=args.img_net, transforms=FilterImages('vld'), stripped=stripped)
print("Done.\nLoading testing dataset ...")
dat_tst = CSVDataset(dat_file=args.test_path, cnf_file=args.cnf_file, mode=2, img_mode=args.img_mode, arch=args.img_net, transforms=FilterImages('tst'), stripped=stripped)
# print("Done.")

label_fractions = dat_trn.label_fractions

df = pd.read_csv(args.data_path)

label_distribution = {}
# for label in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
#     label_distribution[label] = dict(df[label].value_counts())
ckpt_path = args.ckpt_path

print(label_fractions)
print(label_distribution)

# initialize and save Transformer

if args.mgda:
    mdl = MGDA(
        src_modalities = dat_trn.feature_modalities,
        tgt_modalities = dat_trn.label_modalities,
        label_fractions = label_fractions,
        d_model = args.d_model,
        nhead = args.nhead,
        num_encoder_layers = args.num_trf_layers,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size, 
        batch_size_multiplier = 1,
        lr = args.lr,
        weight_decay = args.weight_decay,
        gamma = args.gamma,
        criterion = 'AUC (ROC)',
        device = 'cuda',
        cuda_devices = [0],
        img_net = args.img_net,
        imgnet_layers = args.imgnet_layers,
        img_size = img_size,
        fusion_stage= args.fusion_stage,
        imgnet_ckpt = args.imgnet_ckpt,
        patch_size = args.patch_size,
        ckpt_path = ckpt_path,
        train_imgnet = args.train_imgnet,
        load_from_ckpt = args.load_from_ckpt,
        save_intermediate_ckpts = args.save_intermediate_ckpts,
        data_parallel = args.parallel,
        verbose = 4,
        wandb_ = args.wandb,
        label_distribution = label_distribution,
        ranking_loss = args.ranking_loss,
        _amp_enabled = False,
        _dataloader_num_workers = 2,
        embedding_path = args.emb_path,
        emb_droprate = args.emb_droprate
    )
elif args.gaussian:
    mdl = GMModel(
        src_modalities = dat_trn.feature_modalities,
        tgt_modalities = dat_trn.label_modalities,
        label_fractions = label_fractions,
        d_model = args.d_model,
        nhead = args.nhead,
        num_encoder_layers = args.num_trf_layers,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size, 
        batch_size_multiplier = 1,
        lr = args.lr,
        weight_decay = args.weight_decay,
        gamma = args.gamma,
        criterion = 'AUC (ROC)',
        device = 'cuda',
        cuda_devices = [0],
        img_net = args.img_net,
        imgnet_layers = args.imgnet_layers,
        img_size = img_size,
        fusion_stage= args.fusion_stage,
        imgnet_ckpt = args.imgnet_ckpt,
        patch_size = args.patch_size,
        ckpt_path = ckpt_path,
        train_imgnet = args.train_imgnet,
        load_from_ckpt = args.load_from_ckpt,
        save_intermediate_ckpts = args.save_intermediate_ckpts,
        data_parallel = args.parallel,
        verbose = 4,
        wandb_ = args.wandb,
        label_distribution = label_distribution,
        ranking_loss = args.ranking_loss,
        _amp_enabled = False,
        _dataloader_num_workers = 2,
        embedding_path = args.emb_path,
        emb_droprate = args.emb_droprate,
        clr_temp = args.temp,
        clr_ratio = args.clr_ratio,
        name = args.name
    )
elif args.contrastive:
#     mdl = SCLAD(
#         src_modalities = dat_trn.feature_modalities,
#         tgt_modalities = dat_trn.label_modalities,
#         label_fractions = label_fractions,
#         d_model = args.d_model,
#         nhead = args.nhead,
#         num_encoder_layers = args.num_trf_layers,
#         num_epochs = args.num_epochs,
#         batch_size = args.batch_size, 
#         batch_size_multiplier = 1,
#         lr = args.lr,
#         weight_decay = args.weight_decay,
#         gamma = args.gamma,
#         criterion = 'AUC (ROC)',
#         device = 'cuda',
#         cuda_devices = [0],
#         img_net = args.img_net,
#         imgnet_layers = args.imgnet_layers,
#         img_size = img_size,
#         fusion_stage= args.fusion_stage,
#         imgnet_ckpt = args.imgnet_ckpt,
#         patch_size = args.patch_size,
#         ckpt_path = ckpt_path,
#         train_imgnet = args.train_imgnet,
#         load_from_ckpt = args.load_from_ckpt,
#         save_intermediate_ckpts = args.save_intermediate_ckpts,
#         data_parallel = args.parallel,
#         verbose = 4,
#         wandb_ = args.wandb,
#         label_distribution = label_distribution,
#         ranking_loss = args.ranking_loss,
#         _amp_enabled = False,
#         _dataloader_num_workers = 2,
#         embedding_path = args.emb_path,
#         emb_droprate = args.emb_droprate
#     )
# else:
    mdl = ADRDModel(
        src_modalities = dat_trn.feature_modalities,
        tgt_modalities = dat_trn.label_modalities,
        label_fractions = label_fractions,
        d_model = args.d_model,
        nhead = args.nhead,
        num_encoder_layers = args.num_trf_layers,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size, 
        batch_size_multiplier = 1,
        lr = args.lr,
        weight_decay = args.weight_decay,
        gamma = args.gamma,
        criterion = 'AUC (ROC)',
        device = 'cuda',
        cuda_devices = [0],
        img_net = args.img_net,
        imgnet_layers = args.imgnet_layers,
        img_size = img_size,
        fusion_stage= args.fusion_stage,
        imgnet_ckpt = args.imgnet_ckpt,
        patch_size = args.patch_size,
        ckpt_path = ckpt_path,
        train_imgnet = args.train_imgnet,
        load_from_ckpt = args.load_from_ckpt,
        save_intermediate_ckpts = args.save_intermediate_ckpts,
        data_parallel = args.parallel,
        verbose = 4,
        wandb_ = args.wandb,
        label_distribution = label_distribution,
        ranking_loss = args.ranking_loss,
        _amp_enabled = False,
        _dataloader_num_workers = 2,
        embedding_path = args.emb_path,
        emb_droprate = args.emb_droprate,
        clr_temp = args.temp,
        clr_ratio = args.clr_ratio,
        name = args.name
    )

# def get_model_hash(model):
#     params = torch.cat([p.view(-1) for p in model.parameters()])
#     return hashlib.md5(params.detach().cpu().numpy().tobytes()).hexdigest()
    
if args.parallel == True:
    mdl.to(local_rank)
    if local_rank == 0:
        start_time = time.time()
        while True:
            if time.time() - start_time >= 10:
                break
            
    # sys.stderr.write(f"Rank {rank}: Model hash: {get_model_hash(mdl.net_)}\n")
        # raise("shit")
    mdl.net_ = DDP(mdl.net_, device_ids=[local_rank], output_device=local_rank
                #    ,find_unused_parameters=True
                )
dev1ce = f'cuda:{local_rank}'
if args.img_mode == 0 or args.img_mode == 2:
    mdl.load(filepath=args.ckpt_path, map_location=dev1ce)
else:
    mdl.load(filepath=args.ckpt_path, map_location=dev1ce)

ldr_trn, ldr_vld = mdl._init_dataloader(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels)

# 1. 各个进程独立运行函数，生成自己的 shap_pairs
shap_pairs = collect_shap_value_pairs(mdl, ldr_vld, target_labels=['MCI', 'AD'])
# shap_pairs = {
#     "MCI": {
#         "age": [(25, 0.1), (30, 0.2)],
#         "gender": [(1, 0.3), (0, 0.4)]
#     },
#     "AD": {
#         "age": [(60, 0.5), (65, 0.6)],
#         "gender": [(1, 0.7), (0, 0.8)]
#     }
# }


# 2. 使用 gather_object 聚合所有进程的结果
if args.parallel == True:
    torch.distributed.barrier()

    # 初始化一个列表，用于在主进程上接收所有对象
    # 对于非主进程，此参数可以是 None
    all_shap_pairs = [None for _ in range(torch.distributed.get_world_size())] if local_rank == 0 else None

    # gather_object 会将每个进程的 shap_pairs 对象收集到主进程的 all_shap_pairs 列表中
    torch.distributed.gather_object(
        obj=shap_pairs,
        object_gather_list=all_shap_pairs,
        dst=0  # 指定 rank 0 为接收方
    )

    # 清理分布式进程组
    torch.distributed.destroy_process_group()

    # 此时，在主进程 (local_rank == 0) 中:
    # all_shap_pairs 是一个列表，其中 all_shap_pairs[i] 包含了来自 rank i 进程的 shap_pairs 数据。
    # 你需要自己将这个列表中的数据合并成你需要的最终格式。
    if local_rank == 0:
        # 例如，如果每个进程的 shap_pairs 是一个 list，你可以将它们合并
        final_shap_pairs = {}
        for item_list in all_shap_pairs: 
            # 将每个进程的结果合并到 final_shap_pairs 中
            # 一级目录是 target label (e.g., 'MCI', 'AD')
            # 二级目录是 feature name (e.g., 'age', 'gender')
            # 三级目录是 (feature_value, shap_value) 的元组列表
            # 期望行为是把一级二级路径相同的三级目录里的元组合并成一个列表
            for target_label, features in item_list.items():
                if target_label not in final_shap_pairs:
                    final_shap_pairs[target_label] = {}
                for feature_name, pairs in features.items():
                    if feature_name not in final_shap_pairs[target_label]:
                        final_shap_pairs[target_label][feature_name] = []
                    final_shap_pairs[target_label][feature_name].extend(pairs)
        # 现在 final_shap_pairs 包含了所有进程的完整结果
    else:
        exit(0)  # 非主进程退出
else:
    # 如果不是并行模式，shap_pairs 就是最终结果
    final_shap_pairs = shap_pairs

# 2. Inspect the results
# The output `shap_pairs` is a nested dictionary.
# Let's see the data for the 'MCI' target and a feature named 'age' (example name)

# This assumes 'age' is one of your feature names.
# Replace with a real feature name from your dataset.
# feature_name = 'age' 
# mci_age_pairs = shap_pairs['MCI'][feature_name]

# print(f"Data for MCI target and feature '{feature_name}':")
# print(mci_age_pairs[:5]) # Print first 5 pairs: [(age1, shap1), (age2, shap2), ...]

# 3. (Recommended) Convert to Pandas DataFrame for easier analysis and plotting
# Example for the 'MCI' target
mci_df = pd.DataFrame()
mci_results = final_shap_pairs['MCI']

with open('shap_data.json', 'w') as f:
    json.dump(mci_results, f, indent=4)

ad_results = final_shap_pairs['AD']

with open('shap_data_ad.json', 'w') as f:
    json.dump(ad_results, f, indent=4)

