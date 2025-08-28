import torch
import numpy as np
from .. import nn
from typing import Any, Dict
from torch import Tensor
import pickle
from icecream import ic

class NumBinGeneralEmb(torch.nn.Module):
    ''' version 1 of the embedding layer '''
    def __init__(self,
                 src_modalities: Dict[str, Dict[str, Any]],
                 d_model: int,
                 device: str = 'cpu',
                 img_net: str | None = None,
                 fusion_stage: str = 'middle',
                 emb_path: str | None = None,
                 dropout_rate: float = 0.3,
                 imgnet_layers: int = 4,
                 img_size: int = 128,
                 patch_size: int | None = 16,
                 imgnet_ckpt: str | None = None,
                 train_imgnet: bool = False
                 ) -> None:
        ''' ... '''
        super().__init__()
        self.d_model = d_model
        self.img_net = img_net
        self.src_modalities = src_modalities
        self.device = device
        self.fusion_stage = fusion_stage
        self.dropout_rate = dropout_rate
        self.layers = imgnet_layers
        self.img_size = img_size
        self.patch_size = patch_size
        self.imgnet_ckpt = imgnet_ckpt
        self.train_imgnet = train_imgnet

        ic.enable()
        #ic.disable()
        # ic("Imgnet_ckpt:", imgnet_ckpt)

        if emb_path is not None and emb_path.lower() not in ["", "none"]:
            with open(emb_path, 'rb') as f:
                emb_dict = pickle.load(f)
                # new pkl: embedding dimension=3072
                emb_dict['dimension'] = 3072
                self.emb_dict = emb_dict
                source = 'GPT 3.5' if 'gpt' in emb_path.lower() else 'OP model'
                print(f'Embedding dict loaded from: {source}')
            for k in src_modalities.keys():
                # if "img_MRI" in k:
                #     print("img_MRI in ", k, "skipped.")
                #     continue
                # Convert embeddings to tensors
                for i in range(len(self.emb_dict[k]['embeddings'])):
                    if (isinstance(self.emb_dict[k]['embeddings'][i], dict)):
                        for key in self.emb_dict[k]['embeddings'][i].keys():
                            self.emb_dict[k]['embeddings'][i][key] = torch.tensor(
                                self.emb_dict[k]['embeddings'][i][key], device=self.device
                            )
                    else:
                        self.emb_dict[k]['embeddings'][i] = torch.tensor(
                            self.emb_dict[k]['embeddings'][i], device=self.device
                        )
                # self.emb_dict[k]['embedding'] = torch.tensor(
                #     self.emb_dict[k]['embedding'], device=self.device
                # )
        else:
            self.emb_dict = None
            print('Simple linear embedding implemented.')

        # Embedding modules for source modalities
        self.modules_emb_src = torch.nn.ModuleDict()

        if self.img_net.lower() != "nonimg":
            print('Downsample layers: ', self.layers)
            self.img_model = nn.ImagingModelWrapper(arch=self.img_net, img_size=self.img_size, patch_size=self.patch_size, ckpt_path=self.imgnet_ckpt, train_backbone=self.train_imgnet, layers=self.layers, out_dim=self.d_model, device=self.device, fusion_stage=self.fusion_stage)

        if self.emb_dict is None:
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    self.modules_emb_src[k] = torch.nn.Embedding(info['num_categories'], d_model)
                elif info['type'] == 'numerical':
                    self.modules_emb_src[k] = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(info['shape'][0]),
                        torch.nn.Linear(info['shape'][0], d_model)
                    )
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    # unrecognized
                    raise ValueError('{} is an unrecognized data modality'.format(k))
        else:
            print("------------------Experiment is on GPT embeddings-----------------------")
            # 对于二元特征，创建共享的线性层
            self.binary_emb_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.emb_dict['dimension'], self.d_model)
                for _ in range(2)
            ])
            self.num_emb_layers = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dict['dimension'], self.d_model*4),
                torch.nn.GELU(),
                torch.nn.Linear(self.d_model*4, self.d_model),
                torch.nn.GELU()
            )
            self.category_emb_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.emb_dict['dimension'], self.d_model)
            ])
            # # self.num2miu = torch.nn.Linear(self.emb_dict['dimension'], 1)
            # self.num_norms = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=False)

            self.emb_drop = torch.nn.Dropout(p=dropout_rate)
            ################# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.binary_emb_layers.requires_grad_(False)
            ###################################################
            self.cached_embeddings = {}
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    num_categories = info['num_categories']
                    if num_categories == 2:
                        # 对于二元特征，使用共享层
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                    else:
                        # continue
                        # 为每个类别创建一个线性层
                        # self.modules_emb_src[k] = torch.nn.ModuleList([
                        #     torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], self.d_model)
                        #     for _ in range(num_categories)
                        # ])
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                elif info['type'] == 'numerical':
                    self.cached_embeddings[k] = None
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    raise ValueError(f'{k} is an unrecognized data modality')
            #####################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.modules_emb_src.requires_grad_(False)

    def clear_cache(self):
        """Clear cached embeddings. Should be called at the start of each training epoch."""
        for k in self.cached_embeddings.keys():
            self.cached_embeddings[k] = None

    def forward(self,
                x: Dict[str, Tensor],
                mask: Dict[str, Tensor],
                skip_embedding: Dict[str, bool] | None = None,
                ) -> Dict[str, Tensor]:
        """ ... """
        # for i in range(3):
        #     assert(not torch.isnan(self.modules_emb_src['his_NACCREAS'][i]))
        out_emb = {}
        batch_size = next(iter(x.values())).shape[0]  # Assume all inputs have the same batch size

        for k in self.src_modalities.keys():
            if skip_embedding is None or not skip_embedding.get(k, False):
                if "img_MRI" in k:
                    # print("img_MRI in ", k)
                    if torch.all(~mask[k]):
                        if "swinunetr" in self.img_net.lower() and self.fusion_stage == 'late':
                            out_emb[k] = torch.zeros((1,768,4,4,4))
                        else:
                            out_emb[k] = torch.zeros((mask[k].shape[0], self.d_model)).to(self.device, non_blocking=True)
                        # print("mask is True, out_emb[k]: ", out_emb[k].size())
                    else:
                        out_emb[k] = self.modules_emb_src[k](x[k])

                elif self.src_modalities[k]['type'] == 'categorical':
                    num_categories = self.src_modalities[k]['num_categories']
                    if num_categories == 2: # For binary features
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        # check if cache available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            # compute intermidiate embeddings and cache them
                            # pick first value in dict
                            assert(isinstance(self.emb_dict[k]['embeddings'][0], dict))
                            # print(self.emb_dict[k]['embeddings'][0].values())
                            bin_emb = next(iter(self.emb_dict[k]['embeddings'][0].values()))
                            transformed_embeddings = torch.stack([
                                self.binary_emb_layers[i](bin_emb) for i in range(2)
                                # self.binary_emb_layers[i](self.emb_dict[k]['embedding']) for i in range(2)
                            ])  # [2, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        
                        transformed_embeddings = self.emb_drop(transformed_embeddings)
                        
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]

                    else: # For non-binary categorical features
                        # Check if cached embeddings are available
                        # continue
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            #try:
                            #    assert(isinstance(self.emb_dict[k]['embeddings'][0], dict))
                            #if not isinstance(self.emb_dict[k]['embeddings'][0], dict):
                            #    ic(f"{k} is fucked up")
                            #    ic(self.emb_dict[k]['embeddings'][0])
                            #    raise("fucked")
                            #print(self.emb_dict[k]['embeddings'][0].values())
                            emb = list(self.emb_dict[k]['embeddings'][0].values())
                            # emb = self.emb_dict[k]['embedding'].to(self.device)
                            # assert(isinstance(emb, torch.Tensor))
                            # assert(emb.shape[-1] == self.emb_dict['dimension'])
                            # assert(emb.shape[0] == num_categories)
                            assert(isinstance(emb, list))
                            if len(emb) != num_categories:
                                ic(k, len(emb), num_categories)
                                assert(num_categories > len(emb))
                                emb = emb + [emb[-1]] * (num_categories - len(emb))
                            # assert(len(emb) == num_categories)
                            transformed_embeddings = torch.stack([
                                self.category_emb_layers[0](emb[i]) for i in range(num_categories)
                            ])  # [num_categories, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                            # emb = self.emb_dict[k]['embedding'].to(self.device)
                            # transformed_embeddings = torch.stack([
                            #     self.modules_emb_src[k][i](emb) for i in range(num_categories)
                            # ])  # [num_categories, d_model]
                            # self.cached_embeddings[k] = transformed_embeddings

                        transformed_embeddings = self.emb_drop(transformed_embeddings)
                        # Get indices
                        if torch.isnan(transformed_embeddings).any():
                            print(f'Nan in {k}')
                            raise('nan')
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                elif self.src_modalities[k]['type'] == 'numerical':
                    # For numerical features
                    transformed_embedding = self.cached_embeddings.get(k)
                    if transformed_embedding is None:
                        # emb = self.emb_dict[k]['embedding'] # OLD PKL
                        emb = self.emb_dict[k]['embeddings'][0] # NEW PKL [0]rewrite version
                        transformed_embedding = self.num_emb_layers(emb)  # 形状: [d_model]
                        self.cached_embeddings[k] = transformed_embedding
                    # Dropout
                    transformed_embedding = self.emb_drop(transformed_embedding)
                    # Multiply by the normalized feature values
                    out_emb[k] = x[k] * transformed_embedding
                else:
                    out_emb[k] = x[k]
            else:
                out_emb[k] = x[k]
            # print(k)
            # print(x[k])
            try:
                assert(not torch.isnan(out_emb[k]).any())
            except:
                ic(f'Nan in {k}')
                exit(0)
        return out_emb


class ImageEmbeddingLayer(torch.nn.Module): # don't modify this class yet
    ''' version 1 of the embedding layer '''
    def __init__(self,
                 src_modalities: Dict[str, Dict[str, Any]],
                 img_size = None,
                 patch_size = None,
                 imgnet_ckpt = None,
                 train_imgnet = None,
                 down_dim_layers = None,
                 d_model: int = 128,
                 device: str = 'cpu',
                 img_net: str | None = None,
                 fusion_stage: str = 'middle',
                 emb_path: str | None = None,
                 dropout_rate: float = 0.3,
                 imgnet_layers: int = 4
                 ) -> None:
        ''' ... '''
        super().__init__()
        self.d_model = d_model
        self.img_net = img_net
        self.src_modalities = src_modalities
        self.device = device
        self.fusion_stage = fusion_stage
        self.dropout_rate = dropout_rate
        self.layers = imgnet_layers

        if emb_path is not None and emb_path.lower() not in ["", "none"]:
            with open(emb_path, 'rb') as f:
                emb_dict = pickle.load(f)
                self.emb_dict = emb_dict
                source = 'GPT 3.5' if 'gpt' in emb_path.lower() else 'OP model'
                print(f'Embedding dict loaded from: {source}')
            for k in src_modalities.keys():
                # Convert embeddings to tensors
                self.emb_dict[k]['embedding'] = torch.tensor(
                    self.emb_dict[k]['embedding'], device=self.device
                )
        else:
            self.emb_dict = None
            print('Simple linear embedding implemented.')

        # Embedding modules for source modalities
        self.modules_emb_src = torch.nn.ModuleDict()
        self.cached_embeddings = {}
        self.batch_norms = {}

        if self.img_net.lower() != "nonimg":
            print('Downsample layers: ', self.layers)
            self.img_model = nn.ImagingModelWrapper(arch=self.img_net, img_size=self.img_size, patch_size=self.patch_size, ckpt_path=self.imgnet_ckpt, train_backbone=self.train_imgnet, layers=self.layers, out_dim=self.d_model, device=self.device, fusion_stage=self.fusion_stage)

        if self.emb_dict is None:
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    self.modules_emb_src[k] = torch.nn.Embedding(info['num_categories'], d_model)
                elif info['type'] == 'numerical':
                    self.modules_emb_src[k] = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(info['shape'][0]).cuda(),
                        torch.nn.Linear(info['shape'][0], d_model)
                    )
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    # unrecognized
                    raise ValueError('{} is an unrecognized data modality'.format(k))
        else:
            print("------------------Experiment is on GPT embeddings-----------------------")
            # 对于二元特征，创建共享的线性层
            self.binary_emb_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.emb_dict['dimension'], self.d_model)
                for _ in range(2)
            ])

            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    num_categories = info['num_categories']
                    if num_categories == 2:
                        # 对于二元特征，使用共享层
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                    else:
                        # 为每个类别创建一个线性层
                        self.modules_emb_src[k] = torch.nn.ModuleList([
                            torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], self.d_model)
                            for _ in range(num_categories)
                        ])
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                elif info['type'] == 'numerical':
                    # 为每个数值特征创建一个线性层
                    self.modules_emb_src[k] = torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], self.d_model)
                    # 初始化缓存为 None
                    self.cached_embeddings[k] = None
                    self.batch_norms[k] = torch.nn.BatchNorm1d(info['shape'][0]).cuda()
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    raise ValueError(f'{k} is an unrecognized data modality')

    def clear_cache(self):
        """Clear cached embeddings. Should be called at the start of each training epoch."""
        for k in self.cached_embeddings.keys():
            self.cached_embeddings[k] = None

    def forward(self,
                x: Dict[str, Tensor],
                mask: Dict[str, Tensor],
                skip_embedding: Dict[str, bool] | None = None,
                ) -> Dict[str, Tensor]:
        """ ... """
        out_emb = {}
        batch_size = next(iter(x.values())).shape[0]  # Assume all inputs have the same batch size

        for k in self.src_modalities.keys():
            if skip_embedding is None or not skip_embedding.get(k, False):
                if "img_MRI" in k:
                    # print("img_MRI in ", k)
                    if torch.all(mask[k]):
                        if "swinunetr" in self.img_net.lower() and self.fusion_stage == 'late':
                            out_emb[k] = torch.zeros((1,768,4,4,4))
                        else:
                            if 'cuda' in self.device:
                                device = x[k].device
                                # print(device)
                            else:
                                device = self.device
                            out_emb[k] = torch.zeros((mask[k].shape[0], self.d_model)).to(device, non_blocking=True)
                        # print("mask is True, out_emb[k]: ", out_emb[k].size())
                    else:
                        # print("calling modules_emb_src...")
                        out_emb[k] = self.modules_emb_src[k](x[k])
                        # print("mask is False, out_emb[k]: ", out_emb[k].size())
                    
                elif self.src_modalities[k]['type'] == 'categorical':
                    num_categories = self.src_modalities[k]['num_categories']
                    if num_categories == 2: # For binary features
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        
                        # check if cache available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            # compute intermidiate embeddings and cache them
                            transformed_embeddings = torch.stack([
                                self.binary_emb_layers[i](self.emb_dict[k]['embedding']) for i in range(2)
                            ])  # [2, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=self.dropout_rate, training=self.training
                        )
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                    else: # For non-binary categorical features
                        # Check if cached embeddings are available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            emb = self.emb_dict[k]['embedding'].to(self.device)
                            transformed_embeddings = torch.stack([
                                self.modules_emb_src[k][i](emb) for i in range(num_categories)
                            ])  # [num_categories, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=self.dropout_rate, training=self.training
                        )
                        # Get indices
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                elif self.src_modalities[k]['type'] == 'numerical':
                    # For numerical features
                    transformed_embedding = self.cached_embeddings.get(k)
                    if transformed_embedding is None:
                        emb = self.emb_dict[k]['embedding']
                        transformed_embedding = self.modules_emb_src[k](emb)  # 形状: [d_model]
                        self.cached_embeddings[k] = transformed_embedding
                    # Dropout
                    x[k] = self.batch_norms[k](x[k])
                    transformed_embedding = torch.nn.functional.dropout(
                        transformed_embedding, p=self.dropout_rate, training=self.training
                    )
                    # Multiply by the normalized feature values
                    out_emb[k] = x[k] * transformed_embedding
                else:
                    out_emb[k] = x[k]
            else:
                out_emb[k] = x[k]
        return out_emb
        
class EmbeddingLayer(torch.nn.Module):
    ''' embedding layer: concate multi-class '''
    def __init__(self,
                 src_modalities: Dict[str, Dict[str, Any]],
                 d_model: int,
                 device: str = 'cpu',
                 img_net: str | None = None,
                 fusion_stage: str = 'middle',
                 emb_path: str | None = None,
                 dropout_rate: float = 0.3,
                 ) -> None:
        ''' ... '''
        super().__init__()
        self.d_model = d_model
        self.img_net = img_net
        self.src_modalities = src_modalities
        self.device = device
        self.fusion_stage = fusion_stage
        self.dropout_rate = dropout_rate

        if emb_path is not None and emb_path.lower() not in ["", "none"]:
            with open(emb_path, 'rb') as f:
                emb_dict = pickle.load(f)
                self.emb_dict = emb_dict
                source = 'GPT 3.5' if 'gpt' in emb_path.lower() else 'OP model'
                print(f'Embedding dict loaded from: {source}')
            for k in src_modalities.keys():
                # Convert embeddings to tensors
                self.emb_dict[k]['embedding'] = torch.tensor(
                    self.emb_dict[k]['embedding'], device=self.device
                )
        else:
            self.emb_dict = None
            print('Simple linear embedding implemented.')

        # Embedding modules for source modalities
        self.modules_emb_src = torch.nn.ModuleDict()

        if self.img_net.lower() != "nonimg":
            print('Downsample layers: ', self.layers)
            self.img_model = nn.ImagingModelWrapper(arch=self.img_net, img_size=self.img_size, patch_size=self.patch_size, ckpt_path=self.imgnet_ckpt, train_backbone=self.train_imgnet, layers=self.layers, out_dim=self.d_model, device=self.device, fusion_stage=self.fusion_stage)

        if self.emb_dict is None:
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    self.modules_emb_src[k] = torch.nn.Embedding(info['num_categories'], d_model)
                elif info['type'] == 'numerical':
                    self.modules_emb_src[k] = torch.nn.Sequential(
                        torch.nn.BatchNorm1d(info['shape'][0]),
                        torch.nn.Linear(info['shape'][0], d_model)
                    )
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    # unrecognized
                    raise ValueError('{} is an unrecognized data modality'.format(k))
        else:
            print("------------------Experiment is on GPT embeddings-----------------------")
            # 对于二元特征，创建共享的线性层
            self.binary_emb_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.emb_dict['dimension'], self.d_model)
                for _ in range(2)
            ])
            self.batch_norms = torch.nn.ModuleDict()
            ################# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.binary_emb_layers.requires_grad_(False)
            ###################################################
            self.cached_embeddings = {}
            for k, info in src_modalities.items():
                if info['type'] == 'categorical':
                    num_categories = info['num_categories']
                    if num_categories == 2:
                        # 对于二元特征，使用共享层
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                    else:
                        # 为每个类别创建一个线性层
                        self.modules_emb_src[k] = torch.nn.ModuleList([
                            torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], self.d_model)
                            for _ in range(num_categories)
                        ])
                        # 初始化缓存为 None
                        self.cached_embeddings[k] = None
                elif info['type'] == 'numerical':
                    # 为每个数值特征创建一个线性层
                    self.modules_emb_src[k] = torch.nn.Linear(self.emb_dict[k]['embedding'].shape[-1], self.d_model)
                    # 初始化缓存为 None
                    self.cached_embeddings[k] = None
                    self.batch_norms[k] = torch.nn.BatchNorm1d(info['shape'][0])
                elif info['type'] == 'imaging' or info['type'] == 'img_emb':
                    if self.img_net:
                        self.modules_emb_src[k] = self.img_model
                else:
                    raise ValueError(f'{k} is an unrecognized data modality')
            #####################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.modules_emb_src.requires_grad_(False)

    def clear_cache(self):
        """Clear cached embeddings. Should be called at the start of each training epoch."""
        for k in self.cached_embeddings.keys():
            self.cached_embeddings[k] = None

    def forward(self,
                x: Dict[str, Tensor],
                mask: Dict[str, Tensor],
                skip_embedding: Dict[str, bool] | None = None,
                ) -> Dict[str, Tensor]:
        """ ... """
        out_emb = {}
        batch_size = next(iter(x.values())).shape[0]  # Assume all inputs have the same batch size

        for k in self.src_modalities.keys():
            if skip_embedding is None or not skip_embedding.get(k, False):
                if "img_MRI" in k:
                    # Image processing (omitted for brevity)
                    pass
                elif self.src_modalities[k]['type'] == 'categorical':
                    num_categories = self.src_modalities[k]['num_categories']
                    if num_categories == 2: # For binary features
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        
                        # check if cache available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            # compute intermidiate embeddings and cache them
                            transformed_embeddings = torch.stack([
                                self.binary_emb_layers[i](self.emb_dict[k]['embedding']) for i in range(2)
                            ])  # [2, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=self.dropout_rate, training=self.training
                        )
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                    else: # For non-binary categorical features
                        # Check if cached embeddings are available
                        transformed_embeddings = self.cached_embeddings.get(k)
                        if transformed_embeddings is None:
                            emb = self.emb_dict[k]['embedding'].to(self.device)
                            transformed_embeddings = torch.stack([
                                self.modules_emb_src[k][i](emb) for i in range(num_categories)
                            ])  # [num_categories, d_model]
                            self.cached_embeddings[k] = transformed_embeddings
                        # 应用 Dropout
                        transformed_embeddings = torch.nn.functional.dropout(
                            transformed_embeddings, p=self.dropout_rate, training=self.training
                        )
                        # Get indices
                        indices = x[k].long().to(self.device)  # Shape: [batch_size]
                        # Select embeddings based on indices
                        out_emb[k] = transformed_embeddings[indices]  # Shape: [batch_size, d_model]
                elif self.src_modalities[k]['type'] == 'numerical':
                    # For numerical features
                    transformed_embedding = self.cached_embeddings.get(k)
                    if transformed_embedding is None:
                        emb = self.emb_dict[k]['embedding']
                        transformed_embedding = self.modules_emb_src[k](emb)  # 形状: [d_model]
                        self.cached_embeddings[k] = transformed_embedding
                    # Dropout
                    transformed_embedding = torch.nn.functional.dropout(
                        transformed_embedding, p=self.dropout_rate, training=self.training
                    )
                    x[k] = self.batch_norms[k](x[k])
                    # Multiply by the normalized feature values
                    out_emb[k] = x[k] * transformed_embedding
                else:
                    out_emb[k] = x[k]
            else:
                out_emb[k] = x[k]
        return out_emb

