import torch
import torch.nn as nn

class RankConsistentLoss(nn.Module):
    """
    [Loss]: Contrastive Rank-Consistent Optimization
    
    Enforces ordinal consistency: if Age(A) > Age(B), then Pred(A) > Pred(B) + Margin.
    This module rectifies the aging manifold by optimizing the pairwise relative 
    ordering within a mini-batch (O(N^2) complexity via vectorization).
    """
    def __init__(self, margin=2.0):
        super(RankConsistentLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): Predicted ages [B].
            targets (Tensor): Ground truth ages [B].
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # 1. Pairwise Broadcast Expansion
        # Create matrices of shape [B, B] representing all possible pairs
        p_i, p_j = preds.unsqueeze(1), preds.unsqueeze(0)
        t_i, t_j = targets.unsqueeze(1), targets.unsqueeze(0)
        
        # 2. Relationship Mapping
        # Compute ground truth differences and capture ordering sign (-1, 0, 1)
        t_diff = t_i - t_j
        rank_target = torch.sign(t_diff)
        
        # 3. Pair Selection Mask
        # Only pairs with distinct ages are utilized to prevent ambiguous gradients
        valid_mask = (rank_target != 0)
        
        if not valid_mask.any():
            # Return zero with gradient tracking to prevent computation graph break
            return preds.sum() * 0.0
            
        # 4. Flattened Computation
        # Extract distinct pairs for optimized MarginRankingLoss computation
        p_i_flat = p_i.expand_as(t_diff)[valid_mask]
        p_j_flat = p_j.expand_as(t_diff)[valid_mask]
        target_flat = rank_target[valid_mask]
        
        return self.loss_fn(p_i_flat, p_j_flat, target_flat)


class HybridAgeLoss(nn.Module):
    """
    [Loss]: Multi-Objective Hybrid Aging Loss
    
    Combines Absolute Regression Accuracy (MAE) with Relative Ordering Consistency (Rank).
    L_total = (w_mae * L_MAE) + (w_rank * L_Rank)
    """
    def __init__(self, mae_weight=1.0, rank_weight=1.0, rank_margin=2.0):
        super(HybridAgeLoss, self).__init__()
        self.mae_weight = mae_weight
        self.rank_weight = rank_weight
        
        self.mae_fn = nn.L1Loss(reduction='mean')
        self.rank_fn = RankConsistentLoss(margin=rank_margin)

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): Model predictions [B, 1].
            targets (Tensor): Ground truth [B, 1].
        """
        if preds.shape != targets.shape:
            targets = targets.view_as(preds)

        # 1. Regression Term: Absolute Alignment
        loss_mae = self.mae_fn(preds, targets)
        
        # 2. Ranking Term: Manifold Rectification
        if self.rank_weight > 0:
            loss_rank = self.rank_fn(preds, targets)
        else:
            loss_rank = torch.tensor(0.0, device=preds.device)
            
        # 3. Objective Fusion
        total_loss = (self.mae_weight * loss_mae) + (self.rank_weight * loss_rank)
        
        # Metrics dictionary for engine logging
        metrics = {
            "loss_mae": loss_mae.item(),
            "loss_rank": loss_rank.item(),
            "loss_total": total_loss.item()
        }
        
        return total_loss, metrics