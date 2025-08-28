import torch
import numpy as np
from .. import nn
from typing import Any, Dict
from torch import Tensor
import pickle
import math

class GatedTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(GatedTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
        self.gate = torch.nn.Linear(d_model, dim_feedforward)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self._sa_block(src, src_mask, src_key_padding_mask)
        
        src = self.norm1(src + src2)
        latent = self.linear1(src)
        gated = self.sigmoid(self.gate(src))
        src2 = self.linear2(self.dropout(gated * latent))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
class TransformerLayerPT(torch.nn.Module):
    ''' version 1 of the TransformerLayer '''
    def __init__(self,
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        # N: int,
        # S: int,
        T: int,
        device = 'cuda',
        num_encoder_layers: int = 1,
        emb_path: str | None = None,
        pe: bool = True,
        dimension = 1536
    ) -> None:
        ''' ... '''
        super().__init__()
        self.N = None
        self.S = None
        self.T = T
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.tgt_modalities = tgt_modalities

        # positional encoding
        self.pe = PositionalEncoding(d_model)

        # auxiliary embedding vectors for targets
        self.emb_aux = torch.nn.Parameter(
            torch.zeros(len(tgt_modalities), 1, d_model),
            requires_grad = True,
        )
        
        # transformer with gated layers
        enc = GatedTransformerEncoderLayer(
            self.d_model, self.nhead,
            dim_feedforward = self.d_model,
            activation = 'gelu',
            dropout = 0.2, # left for tuning
        )
        self.transformer = torch.nn.TransformerEncoder(enc, self.num_encoder_layers)

    def forward(self,
        out_emb: dict[str, Tensor],
        mask: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        # print('-----------forward_trf----------')

        self.N = len(next(iter(out_emb.values())))
        emb_src = torch.stack([o for o in out_emb.values()], dim=0)

        self.pe.index = -1
        emb_src = self.pe(emb_src)
        
        # target embedding

        emb_tgt = self.emb_aux.repeat(1, self.N, 1)
        
        # concatenate source embeddings and target embeddings
        emb_all = torch.concatenate((emb_tgt, emb_src), dim=0)

        # combine masks
        mask_src = [mask[k] for k in out_emb.keys()]
        self.S = len(mask_src)
        mask_src = torch.stack(mask_src, dim=1)
        # target masks
        mask_tgt = torch.zeros((self.N, self.T), dtype=torch.bool, device=self.emb_aux.device)
        
        mask_all = torch.concatenate((mask_tgt, mask_src), dim=1)
        # repeat mask_all to fit transformer shape
        mask_all = mask_all.unsqueeze(1).expand(-1, self.S + self.T, -1).repeat(self.nhead, 1, 1)
        assert(not torch.isnan(emb_all).any())
        out_trf = self.transformer(
            src = emb_all,
            mask = mask_all,
        )[0:13]
        return out_trf

class TransformerLayerV2(torch.nn.Module):
    ''' version 1 of the TransformerLayer '''
    def __init__(self,
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        # N: int,
        # S: int,
        T: int,
        device: str = 'cuda',
        num_encoder_layers: int = 1, # should be announced in the train.sh
        emb_path: str | None = None,
        pe: bool = True,
        dimension = 1536 # small model
    ) -> None:
        ''' ... '''
        super().__init__()
        self.N = None
        self.S = None
        self.T = T
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.tgt_modalities = tgt_modalities
        self.device = device
        # positional encoding
        self.pe = PositionalEncoding(d_model)

        self.tgt_linear = torch.nn.Linear(dimension, d_model)
        self.tgt_bn = torch.nn.BatchNorm1d(num_features=d_model, track_running_stats=False, device=device)
        tgt_dict = pickle.load(open('data/embeddings/tgt_large.pkl', 'rb'))# 我先写屎山了 hard code路径在代码里。反正tgt不经常改的
        np_tgt = np.array([v['embedding'] for k,v in tgt_dict.items()])
        self.tgt_ori = torch.tensor(np_tgt, dtype=torch.float32, device=device)
        # transformer
        enc = torch.nn.TransformerEncoderLayer(
            self.d_model, self.nhead,
            dim_feedforward = self.d_model,
            activation = 'gelu',
            dropout = 0.3,
        )
        self.transformer = torch.nn.TransformerEncoder(enc, self.num_encoder_layers)

    def forward(self,
        out_emb: dict[str, Tensor],
        mask: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        # print('-----------forward_trf----------')

        self.N = len(next(iter(out_emb.values())))
        emb_src = torch.stack([o for o in out_emb.values()], dim=0)

        self.pe.index = -1
        emb_src = self.pe(emb_src)
        
        # target embedding

        # emb_tgt = self.emb_aux.repeat(1, self.N, 1)
        emb_tgt = self.tgt_linear(self.tgt_ori)
        emb_tgt = self.tgt_bn(emb_tgt)
        emb_tgt = emb_tgt.unsqueeze(1).repeat(1, self.N, 1)        
        # concatenate source embeddings and target embeddings
        emb_all = torch.concatenate((emb_tgt, emb_src), dim=0)

        # combine masks
        mask_src = [mask[k] for k in out_emb.keys()]
        self.S = len(mask_src)
        mask_src = torch.stack(mask_src, dim=1)
        # target masks
        mask_tgt = torch.zeros((self.N, self.T), dtype=torch.bool, device=self.device)
        
        mask_all = torch.concatenate((mask_tgt, mask_src), dim=1)
        # repeat mask_all to fit transformer shape
        mask_all = mask_all.unsqueeze(1).expand(-1, self.S + self.T, -1).repeat(self.nhead, 1, 1)

        out_trf = self.transformer(
            src = emb_all,
            mask = mask_all,
        )[:13]
        return out_trf

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
        if pe_type == 'img':
            self.index += 1
            assert(self.index == 0)
            return x + self.pe[self.index]
        else:
            self.index += 1
            assert(self.index == 0)
            return x + self.pe[self.index:x.size(0)+self.index]



class TransformerLayerGeneral(torch.nn.Module):
    ''' TransformerLayer receive general embeddings '''
    def __init__(self,
        tgt_modalities: dict[str, dict[str, Any]],
        d_model: int,
        nhead: int,
        T: int,
        device: str = 'cuda',
        num_encoder_layers: int = 1, 
        emb_path: str | None = None,
        dimension = 3072
    ) -> None:
        ''' ... '''
        super().__init__()
        self.T = T
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.tgt_modalities = tgt_modalities
        self.device = device
        # positional encoding

        self.tgt_linear = torch.nn.Linear(dimension, d_model)
        self.tgt_bn = torch.nn.BatchNorm1d(num_features=d_model, track_running_stats=False, device=device)
        tgt_dict = pickle.load(open('data/embeddings/tgt_large.pkl', 'rb'))
        np_tgt = np.array([v['embedding'] for k,v in tgt_dict.items()])
        self.tgt_ori = torch.tensor(np_tgt, dtype=torch.float32, device=device)
        # transformer
        enc = torch.nn.TransformerEncoderLayer(
            self.d_model, self.nhead,
            dim_feedforward = self.d_model,
            activation = 'gelu',
            dropout = 0.3,
        )
        self.transformer = torch.nn.TransformerEncoder(enc, self.num_encoder_layers)

        # auxiliary embedding vectors for targets
        self.emb_aux = torch.nn.Parameter(
            torch.zeros(1, 1, d_model),
            requires_grad = True,
        )

    def forward(self,
        out_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """ ... """
        # print('-----------forward_trf----------')

        self.N = out_emb.shape[0] # batchsize
        # target embedding
        emb_tgt = self.emb_aux.repeat(1, self.N, 1)
        emb_all = torch.concatenate((emb_tgt, out_emb), dim=0)

        # combine masks
        mask_src = mask
        self.S = mask_src.shape[0]
        # target masks
        mask_tgt = torch.zeros((self.N, self.T), dtype=torch.bool, device=self.device)
        
        mask_all = torch.concatenate((mask_tgt, mask_src), dim=1)
        # repeat mask_all to fit transformer shape
        mask_all = mask_all.unsqueeze(1).expand(-1, self.S + self.T, -1).repeat(self.nhead, 1, 1)

        out_trf = self.transformer(
            src = emb_all,
            mask = mask_all,
        )[:1]
        return out_trf