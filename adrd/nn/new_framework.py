from .new_llm_finetune_emb import V1EmbeddingLayer
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
        cuda_devices: list = [3],
        img_net: str | None = None,
        layers: int = 3,
        img_size: int | None = 128,
        patch_size: int | None = 16,
        imgnet_ckpt: str | None = None,
        train_imgnet: bool = False,
        fusion_stage: str = 'middle',
        emb_path: str | None = None,
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
        self.layers = layers
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities
        self.device = device
        self.fusion_stage = fusion_stage

        self.embedding_layer = V1EmbeddingLayer(
            src_modalities=src_modalities,
            d_model=d_model,
            device=device,
            img_net=img_net,
            fusion_stage=fusion_stage,
            emb_path=emb_path,
        )
            
        # positional encoding
        self.pe = PositionalEncoding(d_model)

        # auxiliary embedding vectors for targets
        self.emb_aux = torch.nn.Parameter(
            torch.zeros(len(tgt_modalities), 1, d_model),
            requires_grad = True,
        )
        
        # transformer
        enc = torch.nn.TransformerEncoderLayer(
            self.d_model, self.nhead,
            dim_feedforward = self.d_model,
            activation = 'gelu',
            dropout = 0.3,
        )
        self.transformer = torch.nn.TransformerEncoder(enc, self.num_encoder_layers)


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
        return_out_emb: bool = False,
    ) -> dict[str, Tensor]:
        """ ... """

        out_emb = self.embedding_layer.forward(x, mask, skip_embedding)
        '''
        for key, each in out_emb.items():
            if key!="bat_MOCATOTS":
                continue
            print(self.modules_emb_src[key][0].weight)
            print(f"Key: {key}")
            print(f"Shape: {each.shape}")
            print(f"Values: {each}")
            if torch.isnan(each).any():
                print("Embedding contains NaN values.")
        ##########
        '''
        if self.fusion_stage == "late":
            out_emb = {k: v for k,v in out_emb.items() if "img_MRI" not in k}
            img_out_emb = {k: v for k,v in out_emb.items() if "img_MRI" in k}
            mask_nonimg = {k: v for k,v in mask.items() if "img_MRI" not in k}
            out_trf = self.forward_trf(out_emb, mask_nonimg)
            out_trf = torch.concatenate()
        else:
            out_trf = self.forward_trf(out_emb, mask)
        
        # if isinstance(out_trf, dict):
        #     for key, each in out_trf.items():
        #         print(f"Key: {key}")
        #         print(f"Shape: {each.shape}")
        #         print(f"Values: {each}")
        #         if torch.isnan(each).any():
        #             print("Transformer contains NaN values.")
        # else:
        #     print(f"Shape: {out_trf.shape}")
        #     print(f"Values: {out_trf}")
        #     if torch.isnan(out_trf).any():
        #         print("Transformer contains NaN values.")

        out_cls = self.forward_cls(out_trf)

        # if isinstance(out_cls, dict):
        #     for key, each in out_cls.items():
        #         print(f"Key: {key}")
        #         print(f"Shape: {each.shape}")
        #         print(f"Values: {each}")
        #         if torch.isnan(each).any():
        #             print("Classifier contains NaN values.")
        # else:
        #     print(f"Shape: {out_cls.shape}")
        #     print(f"Values: {out_cls}")
        #     if torch.isnan(out_cls).any():
        #         print("Classifier contains NaN values.")

        if return_out_emb:
            return out_emb, out_cls
        return out_cls

    def forward_trf(self,
        out_emb: dict[str, Tensor],
        mask: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        # print('-----------forward_trf----------')
        N = len(next(iter(out_emb.values())))  # batch size
        S = len(self.embedding_layer.modules_emb_src)  # number of sources
        T = len(self.modules_cls)  # number of targets
        if self.fusion_stage == 'late':
            src_iter = [k for k in self.embedding_layer.modules_emb_src.keys() if "img_MRI" not in k]
            S = len(src_iter)  # number of sources

        else:
            src_iter = self.embedding_layer.modules_emb_src.keys()
        tgt_iter = self.modules_cls.keys()

        emb_src = torch.stack([o for o in out_emb.values()], dim=0)
        # print('emb_src: ', emb_src.size())

        self.pe.index = -1
        emb_src = self.pe(emb_src)
        # print('emb_src + pe: ', emb_src.size())
        
        # target embedding
        # print('emb_aux: ', self.emb_aux.size())
        emb_tgt = self.emb_aux.repeat(1, N, 1)
        # print('emb_tgt: ', emb_tgt.size())
        
        # concatenate source embeddings and target embeddings
        emb_all = torch.concatenate((emb_tgt, emb_src), dim=0)

        # combine masks
        mask_src = [mask[k] for k in src_iter]
        mask_src = torch.stack(mask_src, dim=1)
        
        # target masks
        mask_tgt = torch.zeros((N, T), dtype=torch.bool, device=self.emb_aux.device)
        
        # concatenate source masks and target masks
        mask_all = torch.concatenate((mask_tgt, mask_src), dim=1)
        
        # repeat mask_all to fit transformer
        mask_all = mask_all.unsqueeze(1).expand(-1, S + T, -1).repeat(self.nhead, 1, 1)

        # run transformer
        out_trf = self.transformer(
            src = emb_all,
            mask = mask_all,
        )[0]
        return out_trf

    def forward_cls(self,
        out_trf: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        tgt_iter = self.modules_cls.keys()
        out_cls = {k: self.modules_cls[k](out_trf).squeeze(1) for k in tgt_iter}
        return out_cls
    
class PositionalEncoding(torch.nn.Module):

    def __init__(self, 
        d_model: int, 
        max_len: int = 512
    ):
        """ ... """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.index = -1

    def forward(self, x: Tensor, pe_type: str = 'non_img') -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print('pe: ', self.pe.size())
        # print('x: ', x.size())
        if pe_type == 'img':
            self.index += 1
            return x + self.pe[self.index]
        else:
            self.index += 1
            return x + self.pe[self.index:x.size(0)+self.index]


if __name__ == '__main__':
    ''' for testing purpose only '''
    pass


