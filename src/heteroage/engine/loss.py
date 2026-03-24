import torch
import torch.nn as nn

class RankConsistentLoss(nn.Module):
    """
    [Loss]: Adaptive Rank-Consistent Optimization
    
    Dynamic adaptive ranking loss: The penalty intensity is proportional to the true age difference of the samples.
    For samples with a larger age gap, the model must widen the gap in their predictions.
    """
    def __init__(self, margin=1.0, max_margin=10.0, alpha=0.5):
        super(RankConsistentLoss, self).__init__()
        self.min_margin = margin
        self.max_margin = max_margin
        # alpha controls the scaling ratio of the margin
        self.alpha = alpha

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # 1. Pairwise Broadcast Expansion
        p_i, p_j = preds.unsqueeze(1), preds.unsqueeze(0)
        t_i, t_j = targets.unsqueeze(1), targets.unsqueeze(0)
        
        # 2. Extracting the actual age difference and its sign
        t_diff = t_i - t_j
        rank_target = torch.sign(t_diff)
        
        # 3. Select valid pairs (excluding samples with exactly the same real age).
        valid_mask = (rank_target != 0)
        
        if not valid_mask.any():
            return preds.sum() * 0.0
            
        p_i_flat = p_i.expand_as(t_diff)[valid_mask]
        p_j_flat = p_j.expand_as(t_diff)[valid_mask]
        target_flat = rank_target[valid_mask]
        t_diff_flat = t_diff[valid_mask]
        
        # 4. Computational Adaptive Margin
        # Adaptive Margin = clamp(|True Age Difference| * alpha, min, max)
        adaptive_margin = torch.clamp(torch.abs(t_diff_flat) * self.alpha, 
                                      min=self.min_margin, 
                                      max=self.max_margin)
        
        # 5. Hinge Loss
        # max(0, -y * (P_i - P_j) + Adaptive_Margin)
        loss = torch.relu(-target_flat * (p_i_flat - p_j_flat) + adaptive_margin)
        
        return loss.mean()


class HybridAgeLoss(nn.Module):
    """
    [Loss]: Multi-Objective Hybrid Aging Loss (Upgraded with Deep Supervision)
    
    Combines Absolute Regression Accuracy (MAE), Relative Ordering Consistency (Rank),
    and Auxiliary Branch Supervision (Aux) to force each hallmark to learn independent aging features.
    
    L_total = (w_mae * L_MAE) + (w_rank * L_Rank) + (w_aux * L_Aux)
    """
    def __init__(self, mae_weight=1.0, rank_weight=1.0, rank_margin=2.0, aux_weight=0.3):
        super(HybridAgeLoss, self).__init__()
        self.mae_weight = mae_weight
        self.rank_weight = rank_weight
        self.aux_weight = aux_weight 
        
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.rank_fn = RankConsistentLoss(margin=rank_margin)

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor or Tuple): Model predictions. If training, it's a tuple (final_pred, branch_preds).
            targets (Tensor): Ground truth [B, 1].
        """
        if isinstance(preds, tuple):
            final_pred, branch_preds = preds
        else:
            final_pred = preds
            branch_preds = None

        if final_pred.shape != targets.shape:
            targets = targets.view_as(final_pred)

        # 1. Regression Term: Absolute Alignment
        loss_mae = self.mae_fn(final_pred, targets)
        
        # 2. Ranking Term: Manifold Rectification
        if self.rank_weight > 0:
            loss_rank = self.rank_fn(final_pred, targets)
        else:
            loss_rank = torch.tensor(0.0, device=final_pred.device)
            
        # 3. Auxiliary Branch Loss
        if branch_preds is not None and self.aux_weight > 0:
            # targets [Batch, 1]，branch_preds [Batch, 12]
            targets_expanded = targets.view(-1, 1).expand_as(branch_preds)
            loss_aux = self.mae_fn(branch_preds, targets_expanded)
        else:
            loss_aux = torch.tensor(0.0, device=final_pred.device)
            
        # 4. Objective Fusion
        total_loss = (self.mae_weight * loss_mae) + \
                     (self.rank_weight * loss_rank) + \
                     (self.aux_weight * loss_aux)
        
        # Metrics dictionary for engine logging
        metrics = {
            "loss_mae": loss_mae.item(),
            "loss_rank": loss_rank.item(),
            "loss_aux": loss_aux.item(),
            "loss_total": total_loss.item()
        }
        
        return total_loss, metrics