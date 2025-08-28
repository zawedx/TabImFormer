import sys
import os
import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import tqdm
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.special import expit
from copy import deepcopy
from contextlib import suppress
from typing import Any, Self, Type
from functools import wraps
from tqdm import tqdm
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]

# for DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import nn
from ..nn import Transformer
from ..utils import TransformerTrainingDataset, TransformerBalancedTrainingDataset, TransformerValidationDataset, TransformerTestingDataset, Transformer2ndOrderBalancedTrainingDataset
from ..utils.misc import ProgressBar
from ..utils.misc import get_metrics_multitask, print_metrics_multitask
from ..utils.misc import convert_args_kwargs_to_kwargs

from adrd.model.adrd_model import ADRDModel
from ..nn.latent_space import compute_loss
from types import SimpleNamespace

from torch.autograd import Variable
from adrd.model.min_norm_solvers import MinNormSolver, gradient_normalizers

class MGDA(ADRDModel):
    '''
    use MGDA to train the model
    '''
    def _init_net(self):
        """ ... """
        sys.stderr.write(f"I'm on {self.device}, Rank: {self.rank}\n")
        
        self.start_epoch = 0
        if self.load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                self.load(self.ckpt_path, map_location=self.device)
            except:
                print("Cannot load from checkpoint. Initializing new model...")
                self.load_from_ckpt = False

        if not self.load_from_ckpt:
            shared_model = SharedModule(
                src_modalities = self.src_modalities, 
                tgt_modalities = self.tgt_modalities, 
                d_model = self.d_model, 
                nhead = self.nhead, 
                num_encoder_layers = self.num_encoder_layers, 
                num_decoder_layers = self.num_decoder_layers, 
                device = self.device, 
                img_net = self.img_net, 
                img_size = self.img_size, 
                patch_size = self.patch_size, 
                imgnet_ckpt = self.imgnet_ckpt, 
                train_imgnet = self.train_imgnet,
                fusion_stage = self.fusion_stage,
                emb_path = self.emb_path
            )
            task_specific_model = TaskSpecificModule(
                src_modalities = self.src_modalities, 
                tgt_modalities = self.tgt_modalities, 
                d_model = self.d_model, 
                nhead = self.nhead, 
                num_encoder_layers = self.num_encoder_layers, 
                num_decoder_layers = self.num_decoder_layers, 
                device = self.device, 
                img_net = self.img_net, 
                img_size = self.img_size, 
                patch_size = self.patch_size, 
                imgnet_ckpt = self.imgnet_ckpt, 
                train_imgnet = self.train_imgnet,
                fusion_stage = self.fusion_stage,
                emb_path = self.emb_path
            )
            # if not self.load_from_ckpt:
            #     self.mgda_optimizer = self._init_optimizer()
            other_params = [param for param in shared_model.parameters()] + [param for param in task_specific_model.parameters()]
            self.mgda_optimizer = torch.optim.AdamW([
                    # {'params': par_pjt, 'lr': self.lr},
                    # {'params': par_cls, 'lr': self.lr},
                    {'params': other_params, 'lr': self.lr}
                ],
                betas = (0.9, 0.98),
                weight_decay = self.weight_decay
            )
            self.framework = MGDA_UB_framework(shared_model, task_specific_model, self.mgda_optimizer)
            # sys.stderr.write(f"Thread {self.rank}'s net_ initialized.\n")
            
            # intialize model parameters using xavier_uniform
            for name, p in self.framework.named_parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

    def train_one_epoch(self, ldr_trn, epoch, freeze_otr=False):
        # progress bar for batch loops
        if self.verbose > 1 and self.rank == 0: 
            pbr_batch = ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch))
        # set self.scheduler according freezing or not
        self.scheduler.step(epoch)
        # self.optimizer.param_groups[1]['lr'] = 0.0 if freeze_otr else self.lr
        # self.scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # set model to train mode
        torch.set_grad_enabled(True)
        self.net_.train()

        scores_trn, y_true_trn, y_mask_trn = [], [], []
        losses_trn = [[] for _ in self.tgt_modalities]
        
        iters = len(ldr_trn)
        for n_iter, (x_batch, y_batch, mask, y_mask) in enumerate(ldr_trn):
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}
            
            outputs = self.framework.train_one_iteration(x_batch, y_batch, mask, y_mask,
                skip_embedding=self.skip_embedding, detach=False)

            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_trn.append(outputs.detach().to(torch.float).cpu())
            y_true_trn.append(y_batch.cpu())
            y_mask_trn.append(y_mask.cpu())

            # update progress bar
            if self.verbose > 1 and self.rank == 0:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size*dist.get_world_size(), {})
                pbr_batch.refresh() 

            # clear cuda cache
            # if "cuda" in self.device:
            torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1 and self.rank == 0:
            pbr_batch.close()

        # calculate and print training performance metrics

        met_trn, self.thresholds = self.compute_metrics_ddp(
            scores_list=scores_trn,
            y_true_list=y_true_trn,
            y_mask_list=y_mask_trn,
            losses_list=losses_trn,
            training=True
        )
        
        # log metrics to 
        if self.wandb_ == 1 and self.rank == 0:
            wandb.log({f"Train loss {list(self.tgt_modalities)[i]}": met_trn[i]['Loss']  for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Train Balanced Accuracy {list(self.tgt_modalities)[i]}": met_trn[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
            
            wandb.log({f"Train AUC (ROC) {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Train AUPR {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

            if self.verbose > 2:
                print_metrics_multitask(met_trn)
            
        return met_trn
    
    def validate_one_epoch(self, ldr_vld, epoch):
        # # progress bar for validation
        if self.verbose > 1 and self.rank == 0:
            pbr_batch = ProgressBar(len(ldr_vld.dataset), 'Epoch {:03d} (VLD)'.format(epoch))

        # set model to validation mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        scores_vld, y_true_vld, y_mask_vld = [], [], []
        losses_vld = [[] for _ in self.tgt_modalities]
        for x_batch, y_batch, mask, y_mask in ldr_vld:
            # if len(next(iter(x_batch.values()))) < self.batch_size:
            #     break
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch} # if 'img' not in k}
            # x_img_batch = {k: x_img_batch[k].to(self.device) for k in x_img_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}

            # forward
            outputs = self.framework.validate_one_iteration(x_batch, y_batch, mask, y_mask,
                skip_embedding=self.skip_embedding)
            
            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_vld.append(outputs.detach().to(torch.float).cpu())
            y_true_vld.append(y_batch.cpu())
            y_mask_vld.append(y_mask.cpu())

            # update progress bar
            if self.verbose > 1 and self.rank == 0:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size*dist.get_world_size(), {})
                pbr_batch.refresh()

            # clear cuda cache
            # if "cuda" in self.device:
            #     torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1 and self.rank == 0:
            pbr_batch.close()

        # calculate and print validation performance metrics
        met_vld = self.compute_metrics_ddp(
            scores_list=scores_vld,
            y_true_list=y_true_vld,
            y_mask_list=y_mask_vld,
            losses_list=losses_vld,
            training=False
        )
            
        if self.wandb_ == 1 and self.rank == 0:
            wandb.log({f"Validation loss {list(self.tgt_modalities)[i]}": met_vld[i]['Loss'] for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Validation Balanced Accuracy {list(self.tgt_modalities)[i]}": met_vld[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
            
            wandb.log({f"Validation AUC (ROC) {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Validation AUPR {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

        if self.verbose > 2 and self.rank == 0:
            print_metrics_multitask(met_vld)
        
        return met_vld

class MGDA_UB_framework(torch.nn.Module):
    '''
    Multi-Gradient Descent Algorithm - Upper Bound (MGDA-UB) framework for training multiple tasks
        [input X]
            |
        shared_model
            |
        [representation Y]
            |
        task-specific models
            |
        [predictions Z_1, Z_2, ..., Z_n]
    '''
    def __init__(self, shared_model: Module, task_specific_model: Module, optimizer) -> None:
        self.shared_model = shared_model
        self.task_specific_model = task_specific_model
        self.optimizer = optimizer
        pass

    def train_one_iteration(self,
        x: dict[str, Tensor],
        y_batch: dict[str, Tensor],
        mask: dict[str, Tensor],
        y_mask: dict[str, Tensor],
        # x_img: dict[str, Tensor] | Any = None,
        skip_embedding: dict[str, bool] | None = None,
        detach: bool = True
    ) -> Tensor:
        ''' Forward pass '''
        y = self.shared_model(x, mask, skip_embedding=skip_embedding)
        # rep_variable = y.detach().clone()  # Detach and clone the tensor to avoid backprop through shared_model
        # rep_variable.requires_grad_()  # Ensure that rep_variable requires gradients

        rep_variable = y.clone().detach().requires_grad_(True)

        z, result = self.task_specific_model(rep_variable, y_batch, y_mask, detach=detach)

        # for each task, calculate gradient on representation y
        grads = dict()
        scale = dict()
        loss_data = dict()

        for key, value in z.items():
            self.optimizer.zero_grad()  # DEBUG: Check if this is necessary, maybe zero after the loop is enough
            value.backward(retain_graph=True)
            loss_data[key] = value.item()  # Use .item() to get a Python number from a tensor
            grads[key] = rep_variable.grad.clone().detach()  # Detach the gradient
            rep_variable.grad.data.zero_()

        # Normalize all gradients, this is optional and not included in the paper.
        gn = gradient_normalizers(grads, loss_data, "none")  # params['normalization_type'])
        for key, value in z.items():
            grads[key] = grads[key] / gn[key]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[key] for key, value in z.items()])
        for key, value in z.items():
            scale[key] = float(sol[key])
            
        # Scaled back-propagation
        self.optimizer.zero_grad()

        z, result = self.task_specific_model(y, detach=detach)
        # init loss
        loss = 0
        for key, value in z.items():
            loss = loss + scale[key] * value
        
        loss.backward()
        self.optimizer.step()

        # writer.add_scalar('training_loss', loss.data[0], n_iter)
        # for t in tasks:
        #     writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

        return result

    def validate_one_iteration(self,
        x: dict[str, Tensor],
        y_batch: dict[str, Tensor],
        mask: dict[str, Tensor],
        y_mask: dict[str, Tensor],
        # x_img: dict[str, Tensor] | Any = None,
        skip_embedding: dict[str, bool] | None = None,
        detach: bool = True
    ) -> Tensor:
        ''' Forward pass '''
        y = self.shared_model(x, mask, skip_embedding=skip_embedding)
        z, result = self.task_specific_model(y, y_batch, y_mask, detach=detach)
        return result


from sys import modules
from ..nn.new_llm_finetune_emb import V1EmbeddingLayer, V2EmbeddingLayer, RandomProjection
from ..nn.new_llm_finetune_trf import TransformerLayerPT, TransformerLayerV2
import torch
import numpy as np
from .. import nn
# from ..nn import ImagingModelWrapper
from typing import Any, Type
import math
from icecream import ic
import pickle
ic.disable()

class SharedModule(torch.nn.Module):
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
        layers: int = 3,
        img_size: int | None = 128,
        patch_size: int | None = 16,
        imgnet_ckpt: str | None = None,
        train_imgnet: bool = False,
        fusion_stage: str = 'middle',
        emb_path: str | None = None,
        emb_droprate: float = 0.2,
        pretraining: bool = True,
        detach_flag: bool = True
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
        self.pretraining = pretraining
        self.detach_flag = detach_flag

        self.embedding_layer = V1EmbeddingLayer(
            src_modalities=src_modalities,
            d_model=d_model,
            device=device,
            img_net=img_net,
            fusion_stage=fusion_stage,
            emb_path=emb_path,
            dropout_rate=emb_droprate
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

    def forward(self,
        x: dict[str, Tensor],
        mask: dict[str, Tensor],
        # x_img: dict[str, Tensor] | Any = None,
        skip_embedding: dict[str, bool] | None = None,
    ) -> Tensor:
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
        return out_trf


class TaskSpecificModule(torch.nn.Module):
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
        layers: int = 3,
        img_size: int | None = 128,
        patch_size: int | None = 16,
        imgnet_ckpt: str | None = None,
        train_imgnet: bool = False,
        fusion_stage: str = 'middle',
        emb_path: str | None = None,
        emb_droprate: float = 0.2,
        pretraining: bool = True,
        detach_flag: bool = True
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
        self.pretraining = pretraining   
        self.detach_flag = detach_flag

        # classifiers (binary only)
        self.modules_cls = torch.nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                self.modules_cls[k] = torch.nn.Linear(d_model, 1)
            else:
                # unrecognized
                raise ValueError

        self.Func_g = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model)
        )
            

    def forward(self,
        x: Tensor,
        y_batch: dict[str, Tensor],
        y_mask: dict[str, Tensor],
        detach: bool = True,
    ) -> dict[str, Tensor]:
        """ ... """
        x_detach = x.detach()
        # out_cls = self.token0_cls(out_trf[0])
        if self.detach:
            out_cls = self.one2one_cls(x_detach)
        else:
            out_cls = self.one2one_cls(x)
        
        emb_CL = self.Func_g(x)
        # return emb_CL, out_cls

        # calculate loss and return dict

        # 2-step: pretrain and contrastive learning
        representation, outputs = emb_CL, out_cls
        
        # calculate multitask loss
        loss = dict()

        for i, k in enumerate(self.tgt_modalities):
            loss_task = self.loss_fn[k](outputs[k], y_batch[k])
            msk_loss_task = loss_task * y_mask[k]
            msk_loss_mean = msk_loss_task.sum() / torch.sum(torch.stack(list(y_mask.values())))
            # loss += msk_loss_mean
            loss[k] = msk_loss_mean
            # losses_trn[i] += msk_loss_task.detach().cpu().numpy().tolist()

        nce_loss = 0
        for i, k in enumerate(self.tgt_modalities): 
            rept = representation[i]
            y_k = y_batch[k]
            y_mask_k = y_mask[k]
            rept_norm = torch.nn.functional.normalize(rept, p=2, dim=1, eps=1e-12)
            # Generate the similarity matrix using cosine similarity with temperature scaling
            sim_mat = torch.mm(rept_norm, rept_norm.t()) / self.clr_temp
            valid_pairs = y_mask_k.unsqueeze(1) * y_mask_k.unsqueeze(0)
            valid_pairs *= ~torch.eye(y_k.shape[0], device=self.device, dtype=torch.bool)
            sim_mat = torch.where(valid_pairs > 0, sim_mat, torch.tensor(float('-inf')).to(self.device))
            # InfoNCE calculation
            pos_pair_matrix = (y_k.unsqueeze(1) == 1) & (y_k.unsqueeze(0) == 1)
            pos_pair_matrix = pos_pair_matrix.type(torch.FloatTensor).to(self.device)
            # or -> pos_pair_matrix = (y_k.unsqueeze(1) == y_k.unsqueeze(0)).float()
            pos_pair_matrix *= valid_pairs
            log_softmax = torch.nn.functional.log_softmax(sim_mat, dim=1)
            # nce_loss += -(log_softmax * pos_pair_matrix).sum() / (pos_pair_matrix.sum() + 1e-10)
            loss["nce_loss_" + k] = -(log_softmax * pos_pair_matrix).sum() / (pos_pair_matrix.sum() + 1e-10)

        return loss, outputs
        

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