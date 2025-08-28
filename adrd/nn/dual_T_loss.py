import torch
import torch.nn.functional as F
from icecream import ic

def dual_temperature_loss_func(
    rep: torch.Tensor,
    true_y: torch.Tensor,
    y_mask: torch.Tensor,
    temperature=0.1,
    dt_m=10,
) -> torch.Tensor:
    """
    rep: [b, dim] all representations,
    true_y: [b] with values in [0, 1] indicating the true positive label
    y_mask: [b] with values in [0, 1] indicating the mask for the valid labels
    """
    # query: [b, c]
    # key: [b, c]
    # pos_map: [b, b]
    # eye_map: [b, b] with diagonal 0 and others 1
    # sim = query @ key.T
    # assert(true_y.sum() > 0)
    # print(f'true_y = {true_y}')
    # print(f'y_mask = {y_mask}')
    # assert(not torch.isnan(rep).any())
    # assert(not torch.isnan(true_y).any())
    # assert(not torch.isnan(y_mask).any())
    ic.enable()
    ic.disable()
    b = rep.shape[0]
    # print("max element in rep", torch.max(rep))
    norm_rep = F.normalize(rep, p=2, dim=1, eps=1e-12)
    # print("max element in norm_rep", torch.max(norm_rep))

    sim_mat = norm_rep @ norm_rep.t() * (1 - torch.eye(b, device=rep.device))
    sim_mat = sim_mat - torch.max(sim_mat)
    # print("max element in sim_mat", torch.max(sim_mat))
    # assert(not torch.isinf(sim_mat).any())
    # set diagonal to -inf
    y_mask = y_mask.float()
    # assert(torch.max(y_mask))
    y_mask_map = y_mask.unsqueeze(1) @ y_mask.unsqueeze(0) * (1 - torch.eye(b, device=rep.device))
    # print(f'y_mask_map = \n{y_mask_map}')
    # sim_mat = sim_mat + (1 - y_mask_map) * -1e9
    # assert(not torch.isinf(sim_mat).any())
    y_row = true_y.unsqueeze(1).repeat(1, b)
    y_col = true_y.unsqueeze(0).repeat(b, 1)
    pos_mask = torch.eq(y_row, y_col).float().to(y_row.device)
    pos_mask = pos_mask * y_mask_map * (1 - torch.eye(b, device=rep.device))
    # print(f'pos_mask = \n{pos_mask}')
    mask_sum = pos_mask.sum(dim=1)
    # print("max element in sim_mat", torch.max(sim_mat))
    # print("temperature", temperature)
    # print("dt_m * temperature", dt_m * temperature)
    exp_a = (sim_mat / temperature).exp() * y_mask_map
    exp_b = (sim_mat / (dt_m * temperature)).exp() * y_mask_map
    # print("max element in exp_a", torch.max(exp_a))
    # assert(not torch.isinf(exp_a).any())
    # assert(not torch.isinf(exp_b).any())
    # print(f'pos_mask = \n{pos_mask}')
    pos_a = exp_a * pos_mask
    pos_b = exp_b * pos_mask
    # assert(not torch.isnan(pos_a).any())
    # assert(not torch.isnan(pos_b).any())
    # assert(torch.max(exp_a - pos_a) > 0)
    sum_a = exp_a.sum(dim=1)
    # ic(sum_a.shape)
    sum_b = exp_b.sum(dim=1)
    sum_pos_a = pos_a.sum(dim=1)
    sum_pos_b = pos_b.sum(dim=1)
    # assert(not torch.isnan(sum_b).any())
    # assert(not torch.isnan(sum_pos_b).any())
    fraction_over = (1 - pos_b/(sum_b + 1e-9) + 1e-9)
    fraction_under = (1 - pos_a/(sum_a + 1e-9) + 1e-9)
    # for i in range(fraction_over.shape[0]):
    #     if torch.isnan(fraction_over[i]):
    #         print(f"fraction_over[{i}] is nan")
    #         print(f"sum_pos_b[{i}]={sum_pos_b[i]}")
    #         print(f"sum_b[{i}]={sum_b[i]}")
    #         print(f"exp_b[{i}]={exp_b[i]}")
    #         print(f"pos_b[{i}]={pos_b[i]}")
    # assert(not torch.isnan(fraction_over).any())
    # assert(not torch.isnan(fraction_under).any())
    # print(f'fraction_over={fraction_over}, fraction_under={fraction_under}')
    coef = fraction_over / fraction_under
    coef = coef.detach()
    # print(f'coef={coef}')
    # assert(not torch.isnan(coef).any())

    nonzero = torch.nonzero(mask_sum).squeeze()
    mask_sum = mask_sum[nonzero]
    pos_a = pos_a[nonzero]
    sum_pos_a = sum_pos_a[nonzero]
    sum_a = sum_a[nonzero]
    coef = coef[nonzero]
    sum_a_uns = sum_a.unsqueeze(1).repeat(1, pos_a.shape[1])
    # ic(pos_a.shape, sum_a_uns.shape)

    pos_mask = pos_mask[nonzero]
    loss = -torch.log((pos_a + 1e-9) / (sum_a_uns + 1e-9)) * coef
    loss = torch.where(pos_mask > 0, loss, torch.zeros_like(loss))
    loss = loss.sum(dim=1) / mask_sum
    valid_samples = mask_sum.sum()
    if valid_samples == 0:
        return torch.tensor(0.0, device=rep.device)

    loss = loss.sum() / mask_sum.shape[0]
    return loss

    # intra-anchor hardness-awareness

def normed_InfoNCE(
    rept: torch.Tensor,
    true_y: torch.Tensor,
    y_mask: torch.Tensor,
    temperature=0.1    
):
    rept_norm = torch.nn.functional.normalize(rept, p=2, dim=1, eps=1e-12)
    # Generate the similarity matrix using cosine similarity with temperature scaling
    sim_mat = torch.mm(rept_norm, rept_norm.t())
    sim_mat = (sim_mat - torch.max(sim_mat)) / temperature

    y_mask = y_mask.float()
    valid_mask = (y_mask.unsqueeze(1) @ y_mask.unsqueeze(0)) > 0.5
    valid_mask &= ~torch.eye(len(true_y), device = rept.device, dtype = torch.bool)  # 直接使用布尔运算
    sim_mat = sim_mat.masked_fill(~valid_mask, -1e9)
    # InfoNCE calculation
    pos_pair_matrix = (true_y.unsqueeze(1) == 1) & (true_y.unsqueeze(0) == 1)
    pos_pair_matrix = pos_pair_matrix.type(torch.FloatTensor).to(rept.device)
    # or -> pos_pair_matrix = (true_y.unsqueeze(1) == true_y.unsqueeze(0)).float()
    pos_pair_matrix *= valid_mask.float()
    log_softmax = torch.nn.functional.log_softmax(sim_mat, dim=1)
    NCE = -(log_softmax * pos_pair_matrix).sum() / (pos_pair_matrix.sum() + 1e-10)
    return NCE