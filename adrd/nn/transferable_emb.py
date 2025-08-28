import torch
import numpy as np
from .. import nn
from typing import Any, Dict
from torch import Tensor
import pickle

class GeneralEmbLayer(torch.nn.Module):
    ''' version 1 of the embedding layer '''
    def __init__(self,
                 src_modalities: Dict[str, Dict[str, Any]],
                 d_model: int,
                 device: str = 'cpu',
                 img_net: str | None = None,
                 fusion_stage: str = 'middle',
                 emb_path: str | None = None,
                 dropout_rate: float = 0.0,
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

            for k in emb_dict.keys():
                if k == 'dimension': continue
                # Convert embeddings to tensors
                self.emb_dict[k]['embedding'] = torch.tensor(
                    self.emb_dict[k]['embedding'], device=self.device
                )
            value = next(iter(self.emb_dict.values()))
            self.in_dim = len(value['embedding'])
            try:
                del self.emb_dict['dimension']
            except:
                pass
        else:
            self.emb_dict = None
            print('Simple linear embedding implemented.')

        # Embedding modules for source modalities
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
                torch.nn.Linear(self.in_dim, d_model)
                for _ in range(2)
            ])
            # for numerical feature, create 1 shared linear
            self.num_emb_layer = torch.nn.Linear(self.in_dim, d_model)
            self.num_norm = torch.nn.BatchNorm1d(num_features=1, affine=False, track_running_stats=False).to(self.device)

            for v in self.emb_dict.values():
                if isinstance(v, dict) and 'embedding' in v.values():
                    v['embedding'] = torch.stack([
                        torch.tensor(emb, device=self.device) for emb in v['embedding'].values()
                    ])

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

        # calculate cache tensors
        # for fea in x.keys():
        #     if fea not in self.emb_dict: continue
        #     if isinstance(self.emb_dict[fea], dict):
        #         typ = self.emb_dict[fea]['type']
        #         emb = self.emb_dict[fea]['embedding']
        #         if typ == 'binary':
        #             emb = self.binary_emb_layers[0](emb)
        #         elif typ == 'multiple':
        #             emb = self.binary_emb_layers[0](emb)
        #         elif typ == 'numerical':
        #             emb = 
        #         elif typ == 'imaging':
        #             pass
        #         else:
        #             raise('fuck')

        # for fea in x.keys():
        #     if fea not in self.emb_dict: continue
        #     typ = self.emb_dict[fea]['type']

        #     if typ == 'binary':

        #         self.cached_embeddings[fea] = torch.stack([
        #              torch.tensor
        #         ])
        #     elif typ == 'multi':
        #         embs = self.emb_dict[fea]['embedding'].values()
        #         self.cached_embeddings[fea] = torch.stack()
        #     elif typ == 'num':

        for fea in x.keys():
            if fea not in self.emb_dict: continue
            else:
                T = self.emb_dict[fea]['type'] 
                if T == 'binary' or T == 'multiple':
                    indices = x[fea].long().to(self.device)
                    out_emb[fea] = torch.index_select(self.emb_dict[fea]['embedding'], dim=0, index=indices)
                elif T == 'numerical':
                    x[fea] = self.num_norm(x[fea])
                    out_emb[fea] = x[fea] * self.emb_dict[fea]['embedding']

        return out_emb