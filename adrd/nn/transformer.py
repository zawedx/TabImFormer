from sys import modules
from .new_llm_finetune_emb import ImageEmbeddingLayer, NumBinGeneralEmb
from .new_llm_finetune_trf import TransformerLayerPT, TransformerLayerV2
import torch
import numpy as np
from .. import nn
# from ..nn import ImagingModelWrapper
from .net_resnet3d import r3d_18
from typing import Any, Type
import math
Tensor = Type[torch.Tensor]
from icecream import ic
import pickle
ic.disable()

class Transformer(torch.nn.Module):
    ''' ... '''
    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        device: str = 'cpu',
        img_net: str | None = None,
        imgnet_layers: int = 3,
        img_size: int | None = 128,
        patch_size: int | None = 16,
        imgnet_ckpt: str | None = None,
        train_imgnet: bool = False,
        fusion_stage: str = 'middle',
        emb_path: str | None = None,
        emb_droprate: float = 0.2,
        pretraining: bool = True
    ) -> None:
        ''' ... '''
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.img_net = img_net
        self.img_size = img_size
        self.patch_size = patch_size
        self.imgnet_ckpt = imgnet_ckpt
        self.train_imgnet = train_imgnet
        self.layers = imgnet_layers
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities
        self.device = device
        self.fusion_stage = fusion_stage
        self.pretraining = pretraining

        self.embedding_layer = NumBinGeneralEmb(
            src_modalities=src_modalities,
            d_model=d_model,
            device=device,
            img_net=img_net,
            fusion_stage=fusion_stage,
            emb_path=emb_path,
            dropout_rate=emb_droprate,
            imgnet_layers=imgnet_layers,
            img_size=img_size,
            patch_size=patch_size,
            imgnet_ckpt=imgnet_ckpt,
            train_imgnet=train_imgnet
        )

        self.dimension = self.embedding_layer.emb_dict['dimension']
        self.transformer_layer = TransformerLayerPT(
            tgt_modalities=tgt_modalities,
            d_model=d_model,
            nhead=nhead,
            device=device,
            # N=len(next(iter(out_emb.values()))),
            # S=len(mask_src),
            T=len(self.tgt_modalities),
            num_encoder_layers=num_encoder_layers,
            emb_path=emb_path,
            dimension = self.dimension
        )        

        # classifiers (binary only)
        self.modules_cls = torch.nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                self.modules_cls[k] = torch.nn.Linear(d_model, 1)
            else:
                # unrecognized
                raise ValueError

    def forward(self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        # x_img: dict[str, Tensor] | Any = None,
        skip_embedding: dict[str, bool] | None = None,
        detach: bool = False
    ) -> dict[str, Tensor]:
        """ ... """
        if self.training:
            self.embedding_layer.clear_cache()
        out_emb = self.embedding_layer(x, mask, skip_embedding)
        if self.fusion_stage == "late":
            out_emb = {k: v for k,v in out_emb.items() if "img_MRI" not in k}
            img_out_emb = {k: v for k,v in out_emb.items() if "img_MRI" in k}
            mask_nonimg = {k: v for k,v in mask.items() if "img_MRI" not in k}
            out_trf = self.transformer_layer(out_emb, mask_nonimg)
            out_trf = torch.concatenate()
        else:
            out_trf = self.transformer_layer(out_emb, mask)
        trf_detach = out_trf.detach()
        # out_cls = self.token0_cls(out_trf[0])
        if detach:
            out_cls = self.one2one_cls(trf_detach)
        else:
            # assert(not torch.isnan(out_trf).any())
            out_cls = self.one2one_cls(out_trf)
        
        return out_trf, out_cls

    def token0_cls(self,
        out_trf: Tensor,
    ) -> dict[str, Tensor]:
        """ ... """
        tgt_iter = self.modules_cls.keys()
        out_cls = {k: self.modules_cls[k](out_trf).squeeze(1) for k in tgt_iter}
        return out_cls
    
    def one2one_cls(self,
        out_trf # Tensor[13, batchsize, d_model],
    ) -> dict[str, Tensor]:
        """ ... """
        out_cls = {}
        for i, (k, mod) in enumerate(self.modules_cls.items()):
            out_cls[k] = mod(out_trf[i]).squeeze(1)
        
        return out_cls

if __name__ == '__main__':
    ''' for testing purpose only '''
    pass
