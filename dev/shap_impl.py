import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
from typing import Any, Type, Dict, List

# Helper Dataset (无改动)
class TransformerTestingDataset(Dataset):
    def __init__(self, x_data_dict, src_modalities, is_embedding=None):
        self.x = x_data_dict
        self.src_modalities = src_modalities
        self.is_embedding = is_embedding if is_embedding is not None else {}
        self.num_samples = len(next(iter(x_data_dict.values())))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        smp = {k: self.x[k][idx] for k in self.src_modalities}
        mask = {k: torch.tensor(False) for k in self.src_modalities} 
        return smp, mask
    
    @staticmethod
    def collate_fn(batch):
        smps, masks = zip(*batch)
        smp_batch = {k: torch.stack([s[k] for s in smps]) for k in smps[0]}
        mask_batch = {k: torch.stack([m[k] for m in masks]) for k in masks[0]}
        return smp_batch, mask_batch

# ==============================================================================
# 1. Explainer Base Class (优化)
# ==============================================================================
class BaseExplainer(ABC):
    def __init__(self, model_trainer: Any) -> None:
        self.model_trainer = model_trainer
        self.model = model_trainer.net_
        self.src_modalities = list(model_trainer.net_.src_modalities)
        self.tgt_modalities = list(model_trainer.tgt_modalities)
        self.device = model_trainer.device

    def shap_values(self, x: Dict[str, torch.Tensor], is_embedding: Dict[str, bool] | None = None, mask: Dict[str, torch.Tensor] | None = None) -> List[Dict]:
        """
        计算一个批次（batch）的SHAP值。
        不再创建内部的DataLoader，而是直接处理传入的批次x。
        """
        batch_size = len(next(iter(x.values())))
        phi = [
            {
                tgt_k: {src_k: 0.0 for src_k in self.src_modalities}
                for tgt_k in self.tgt_modalities
            }
            for _ in range(batch_size)
        ]
        
        torch.set_grad_enabled(False)
        self.model.eval()
        
        print(f"Explaining a batch of {batch_size} sample(s)...")
        # 直接调用核心函数处理整个批次
        self._shap_values_core(x, phi, is_embedding, mask)

        return phi

    @abstractmethod
    def _shap_values_core(self, smp_batch: Dict, phi_batch: List[Dict], is_embedding: Dict | None, mask: Dict | None):
        pass

# ==============================================================================
# 2. Monte Carlo SHAP Explainer (核心优化)
# ==============================================================================
NUM_PERMUTATIONS = 256
DEBUG_FLAG = False

class MCExplainer(BaseExplainer):
    """
    使用蒙特卡洛近似法的具体实现。
    这个类现在被优化为可以处理整个批次的数据。
    """
    def _shap_values_core(self, smp_batch: Dict, phi_batch: List[Dict], is_embedding: Dict | None, mask: Dict | None):
        avail = self.src_modalities
        batch_size = len(next(iter(smp_batch.values())))
        
        # **核心改动 1: 批量重复**
        # 使用 repeat_interleave 将批次中的每个样本重复NUM_PERMUTATIONS次。
        # 例如，一个 [B, D] 的张量会变成 [B * NUM_PERMUTATIONS, D]。
        # 这是实现批量并行计算的关键。
        large_batch_smps = {k: v.repeat_interleave(NUM_PERMUTATIONS, dim=0) for k, v in smp_batch.items()}
        large_batch_smps = {k: large_batch_smps[k].to(self.device) for k in self.src_modalities}
        
        for src_k in tqdm(avail, desc="Explaining Features", leave=False):
            if DEBUG_FLAG:
                if src_k != "bat_UDSVERLN":
                    continue
            # 为每个排列组合生成前置子集
            to_uncover = []
            for _ in range(NUM_PERMUTATIONS):
                perm = avail.copy()
                random.shuffle(perm)
                to_uncover.append(perm[:perm.index(src_k)])

            # **核心改动 2: 构建更大的掩码（Masks）**
            # 掩码的大小现在是 (B * NUM_PERMUTATIONS)，对应扩展后的大批次。
            large_batch_size = batch_size * NUM_PERMUTATIONS
            
            masks_wo_src_k = {k: np.ones(large_batch_size, dtype=bool) for k in self.src_modalities}
            
            # 这个循环的逻辑保持不变，但现在它作用于一个更大的掩码数组
            for i, lst in enumerate(to_uncover):
                # 将掩码应用到批次中的每个对应样本上
                indices = range(i, large_batch_size, NUM_PERMUTATIONS)
                for k_uncover in lst:
                    for idx in indices:
                        masks_wo_src_k[k_uncover][idx] = False
            
            masks_wi_src_k = {k: v.copy() for k, v in masks_wo_src_k.items()}
            masks_wi_src_k[src_k][:] = False

            if DEBUG_FLAG:
                def print_mask(mask):
                    # print in hexadecimal format
                    pstr = ""
                    for i in range(0, len(mask), 16):
                        submask = mask[i:i+16]
                        bool_array_to_int = int(''.join(['1' if x else '0' for x in submask]), 2)
                        pstr += f"{bool_array_to_int:04x}"
                    print(pstr)

                indices = [1, 2, len(masks_wi_src_k[src_k]) - 2, len(masks_wi_src_k[src_k]) - 1]
                for idx in indices:
                    wi_bool_list = [masks_wi_src_k[k][idx] for k in masks_wi_src_k]
                    wo_bool_list = [masks_wo_src_k[k][idx] for k in masks_wo_src_k]
                    print(f"Index {idx}:")
                    print_mask(wi_bool_list)
                    print_mask(wo_bool_list)

                # while True:
                #     readnum = input("flip mask")
                #     readnum = int(readnum)
                #     if readnum > len(masks_wi_src_k[src_k]) - 1:
                #         print("Index out of range, try again.")
                #         continue
                #     key_name = list(masks_wi_src_k.keys())[readnum]
                #     masks_wi_src_k[key_name][0] = not masks_wi_src_k[key_name][0]
                #     masks_wo_src_k[key_name][0] = not masks_wo_src_k[key_name][0]
                #     print(f"Index {readnum} flipped.")
                #     wi_bool_list = [masks_wi_src_k[k][0] for k in masks_wi_src_k]
                #     wo_bool_list = [masks_wo_src_k[k][0] for k in masks_wo_src_k]
                #     print_mask(wi_bool_list)
                #     print_mask(wo_bool_list)
                #     _, out_wi_src_k = self.model(large_batch_smps, masks_wi_src_k, skip_embedding=self.model_trainer.skip_embedding)
                #     _, out_wo_src_k = self.model(large_batch_smps, masks_wo_src_k, skip_embedding=self.model_trainer.skip_embedding)
                #     diff = out_wi_src_k['MCI'][0].cpu() - out_wo_src_k['MCI'][0].cpu()
                #     print("wi = ", out_wi_src_k['MCI'][0].cpu(), "wo = ", out_wo_src_k['MCI'][0].cpu(), "diff = ", diff)

            masks_wi_src_k = {k: torch.tensor(v, device=self.device) for k, v in masks_wi_src_k.items()}
            masks_wo_src_k = {k: torch.tensor(v, device=self.device) for k, v in masks_wo_src_k.items()}

            is_embedding_arg = self.model_trainer.skip_embedding if is_embedding is None else is_embedding
            
            # **核心改动 3: 一次性在GPU上完成所有计算**
            # 模型现在处理的是一个大小为 [B * NUM_PERMUTATIONS] 的巨大批次
            # OR the mask
            or_masks_wi_src_k = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in masks_wi_src_k.items()}
            or_masks_wo_src_k = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in masks_wo_src_k.items()}
            for feature_name in mask.keys():
                expand_mask = mask[feature_name].repeat_interleave(NUM_PERMUTATIONS, dim=0).to(self.device)
                or_masks_wi_src_k[feature_name] = masks_wi_src_k[feature_name] | expand_mask
                or_masks_wo_src_k[feature_name] = masks_wo_src_k[feature_name] | expand_mask

            _, out_wi_src_k = self.model(large_batch_smps, or_masks_wi_src_k, skip_embedding=is_embedding_arg)
            _, out_wo_src_k = self.model(large_batch_smps, or_masks_wo_src_k, skip_embedding=is_embedding_arg)

            for tgt_k in self.tgt_modalities:
                diff = out_wi_src_k[tgt_k].cpu() - out_wo_src_k[tgt_k].cpu()
                if DEBUG_FLAG:
                    if tgt_k != 'MCI':
                        continue
                    print("diff shape:", diff.shape)
                    reshaped_diff = diff.view(batch_size, NUM_PERMUTATIONS, *diff.shape[1:])
                    print("reshaped diff shape:", reshaped_diff.shape)
                    # output first 10 permutations for each sample
                    for i in range(batch_size):
                        print(f"Sample {i} - Target {tgt_k}:")
                        print(reshaped_diff[i, :10])
                    raise ValueError("Debugging: Check diff shape before reshaping")

                # **核心改动 4: 重塑结果并按样本计算均值**
                # 将 [B * NUM_PERMUTATIONS, ...] 的结果重塑为 [B, NUM_PERMUTATIONS, ...]
                # 然后在 NUM_PERMUTATIONS 维度上求均值，得到每个样本的SHAP值。
                reshaped_diff = diff.view(batch_size, NUM_PERMUTATIONS, *diff.shape[1:])
                shap_values_for_batch = torch.nan_to_num(reshaped_diff).mean(dim=1)
                
                # 将计算出的SHAP值存入每个样本对应的phi字典中
                for i in range(batch_size):
                    phi_batch[i][tgt_k][src_k] = shap_values_for_batch[i].item()


# ==============================================================================
# 3. 主分析函数 (优化)
# ==============================================================================
def collect_shap_value_pairs(model_trainer, data_loader, target_labels=['MCI', 'AD']):
    """
    在整个数据集上运行SHAP分析，并为指定的目标标签收集 (特征值, SHAP值) 对。
    此版本经过优化，可以批量处理SHAP计算。
    """
    print("--- Starting SHAP Value-Pair Collection (Optimized for Batches) ---")
    
    explainer = MCExplainer(model_trainer)
    src_features = explainer.src_modalities
    
    
    # explainer.src_modalities.append("img_MRI_mIP")
    # explainer.src_modalities.append("img_MRI_FLAIR")
    # explainer.src_modalities.append("img_MRI_T1")
    # explainer.src_modalities.append("img_MRI_SWI")
    # explainer.src_modalities.append("img_MRI_Mag")
    # explainer.src_modalities["img_MRI_mIP"] = {"type": "img_emb", "shape": (1,768,4,4,4)}
    # explainer.src_modalities["img_MRI_FLAIR"] = {"type": "img_emb", "shape": (1,768,4,4,4)}
    # explainer.src_modalities["img_MRI_T1"] = {"type": "img_emb", "shape": (1,768,4,4,4)}
    # explainer.src_modalities["img_MRI_SWI"] = {"type": "img_emb", "shape": (1,768,4,4,4)}
    # explainer.src_modalities["img_MRI_Mag"] = {"type": "img_emb", "shape": (1,768,4,4,4)}

    for label in target_labels:
        if label not in explainer.tgt_modalities:
            print(f"Warning: Target label '{label}' not found, skipping.")
    
    results = {
        label: {feature: [] for feature in src_features}
        for label in target_labels if label in explainer.tgt_modalities
    }

    # **核心改动 5: 移除内层循环，直接处理批次**
    
    for x_batch, y_batch, mask, y_mask in tqdm(data_loader, desc="Processing Batches"):
        x_cuda = {k: v.to(model_trainer.device) for k, v in x_batch.items()}
        mask_cuda = {k: v.to(model_trainer.device) for k, v in mask.items()}
        _, y_pred = explainer.model(x_cuda, mask_cuda, skip_embedding=model_trainer.skip_embedding)
        y_pred = {k: v.cpu() for k, v in y_pred.items()}
        for label in target_labels:
            mmask = y_mask[label]
            yy_pred = y_pred[label].detach()
            y_true = y_batch[label]
            
            #取出mask为1的下标，筛选yy_pred和y_true。mask为01数组
            indices = torch.where(mmask == 1)
            yy_pred = yy_pred[indices]
            y_true = y_true[indices]
            #计算Area under ROC
            from sklearn.metrics import roc_auc_score
            if len(yy_pred) == 0 or len(y_true) == 0:
                print(f"Warning: No valid predictions for {label}, skipping AUC calculation.")
                continue
            auc = roc_auc_score(y_true.cpu().numpy(), yy_pred.cpu().numpy())
            print(f"ROC AUC for {label}: {auc:.4f}")
            #计算最佳阈值
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(y_true.cpu().numpy(), yy_pred.cpu().numpy())
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold = thresholds[np.argmax(f1_scores)]
            print(f"Best threshold for {label}: {best_threshold:.4f}")
            print(f"Best F1 score for {label}: {np.max(f1_scores):.4f}")

        # --- 关键步骤: 为整个批次计算SHAP值 ---
        phi_batch = explainer.shap_values(x_batch, mask=mask)

        batch_size = len(next(iter(x_batch.values())))
        
        # 对比x_predict和y_batch，计算指标

        # 循环遍历批次结果，以存储 (特征值, SHAP值) 对
        for i in range(batch_size):
            phi_sample = phi_batch[i]
            
            # 提取原始特征值
            feature_values = {}
            for k, v in x_batch.items():
                try:
                    feature_values[k] = v[i].item() if 'mri' not in k.lower() else -1
                except ValueError:
                    feature_values[k] = v[i].cpu().numpy()

            # 存储结果
            for label in results.keys():
                for feature in src_features:
                    if mask[feature][i] == 1:
                        continue
                    elif "mri" in feature.lower():
                        print(f"Detected MRI feature {feature} for sample:{i} in SHAP value collection.")
                    feat_val = feature_values[feature]
                    shap_val = phi_sample[label][feature]
                    results[label][feature].append((feat_val, shap_val))

    print("\n--- SHAP Value-Pair Collection Complete ---")
    return results



