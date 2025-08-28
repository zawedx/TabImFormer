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

class GMModel(ADRDModel):

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
            
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled,
            ):
                # gaussian mixture loss: compute_loss(input_label, output, args)
                # input_label = y_batch
                input_label = torch.stack(list(y_batch.values()), dim=1)
                # print("input_label_shape: ", input_label.shape, "batch_size: ", self.batch_size)
                # assert(input_label.shape[0] == self.batch_size)
                assert(input_label.shape[1] == 13)
                outputs = self.net_(x_batch, mask, input_label=input_label, skip_embedding=self.skip_embedding)
                '''
                outputs structure:
        output['embs'] = embs
        output['label_out'] = label_out
        output['feat_out'] = feat_out
        output['feat_out2'] = feat_out2
        output['feat'] = feature
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2
        fe_output['label_emb'] = label_emb
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
                '''
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, _, pred_x = \
                    compute_loss(input_label, outputs, args=SimpleNamespace(**{
                        "nll_coeff": 0.1, "device": self.device, "clr_temp": self.clr_temp}))

            if self._amp_enabled:
                assert(False)
                self.scaler.scale(loss).backward()
            else:
                total_loss.backward()
            
            # update parameters
            if n_iter != 0 and n_iter % self.batch_size_multiplier == 0:
                if self._amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    flag = True
                    for name, param in self.net_.named_parameters():
                        if param.grad is None:
                            print(f"Parameter {name} did not receive gradients!")
                            flag = False
                            # print(f"Parameter {name} has gradients.")
                    # assert(flag)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            ''' TODO: change array to dictionary later '''
            # outputs = torch.stack(list(outputs.values()), dim=1)
            # assert(pred_x.shape[0] == self.batch_size)
            assert(pred_x.shape[1] == 13)
            outputs = pred_x
            sum_of_outputs = torch.sum(outputs, dim=0)
            # print("train-sum_of_outputs: ", sum_of_outputs)
            # if torch.isnan(sum_of_outputs[0]):
            #     print("output-nan-dump: ", outputs)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)
            # print(pred_x)
            print('---------------------------------------')
            # exit()
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
        print(scores_trn)
        met_trn = self.compute_metrics_ddp(
            scores_list=scores_trn,
            y_true_list=y_true_trn,
            y_mask_list=y_mask_trn,
            losses_list=losses_trn
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
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled
            ):
                # gaussian mixture loss: compute_loss(input_label, output, args)
                # input_label = y_batch
                input_label = torch.stack(list(y_batch.values()), dim=1)
                # assert(input_label.shape[0] == batch_size)
                assert(input_label.shape[1] == 13)
                outputs = self.net_(x_batch, mask, input_label=input_label, skip_embedding=self.skip_embedding)
                total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, _, pred_x = \
                    compute_loss(input_label, outputs, args=SimpleNamespace(**{
                        "nll_coeff": 0.1, "device": self.device, "clr_temp": self.clr_temp}))

            ''' TODO: change array to dictionary later '''
            # outputs = torch.stack(list(outputs.values()), dim=1)
            # assert(pred_x.shape[0] == batch_size)
            assert(pred_x.shape[1] == 13)
            outputs = pred_x
            sum_of_outputs = torch.sum(outputs, dim=0)
            print("val-sum_of_outputs: ", sum_of_outputs)
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
            losses_list=losses_vld
        )
            
        if self.wandb_ == 1 and self.rank == 0:
            wandb.log({f"Validation loss {list(self.tgt_modalities)[i]}": met_vld[i]['Loss'] for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Validation Balanced Accuracy {list(self.tgt_modalities)[i]}": met_vld[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
            
            wandb.log({f"Validation AUC (ROC) {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
            wandb.log({f"Validation AUPR {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

        if self.verbose > 2 and self.rank == 0:
            print_metrics_multitask(met_vld)
        
        return met_vld