import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BioSparseLinear, AttentionGate
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    """
    [Module]: Isomorphic Residual Block (Upgraded to 2-Layer MLP)
    """
    def __init__(self, dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), #############new##############
            nn.Dropout(dropout) ##############new##############
        )
        
    def forward(self, x):
        return x + self.net(x)

class HallmarkCascade(nn.Module):
    """
    [Module]: Hallmark Signaling Cascade (V2.1 Robust Version)
    - Replaced variable-depth while loops with fixed-depth isomorphic architecture.
    - Introduced immediate dimensional projection and LayerNorm for stability.
    """
    def __init__(self, input_dim, unified_dim, dropout=0.2):
        super(HallmarkCascade, self).__init__()
        
        #  Step 1: Universal Projection
        intermediate_dim = unified_dim * 2
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Step 2: Fixed-Depth Isomorphic Processing
        self.res_block1 = ResidualBlock(intermediate_dim, dropout)
        self.res_block2 = ResidualBlock(intermediate_dim, dropout)
        
        # Step 3: Latent Space Alignment
        self.compression = nn.Sequential(
            nn.Linear(intermediate_dim, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.compression(x)
        return x


class HeteroAgeHAB(nn.Module):
    """
    [Model]: Heterogeneity-Aware Hierarchical Attention Branch Network (HeteroAge-HAB)
    
    A deep learning framework for biological age prediction integrating tri-modal 
    epigenetic data (Beta, CHALM, CAMDA). It utilizes a biologically constrained 
    sparse encoder followed by hallmark-specific cascade networks and an attention 
    mechanism to model the heterogeneity of aging.
    
    Components:
        1. BioSparse Encoder: Maps raw CpGs to hallmark-specific pathways using a binary mask.
        2. Hallmark Cascades: Deep independent networks for pathway-specific feature extraction.
        3. Attention Aggregation: Computes dynamic weights to fuse hallmark embeddings.
        4. Prediction Head: Deep regression module for final age estimation.
    """
    def __init__(self, num_cpgs, branch_info, mask_matrix, unified_dim=64, dropout=0.2, active_modalities=None, modality_dropout=0.0):
        """
        Args:
            num_cpgs (int): Count of unique CpGs in the master feature list.
            branch_info (list): Metadata defining the slice indices for each hallmark branch.
            mask_matrix (Tensor): Binary mask enforcing biological sparsity constraints.
            unified_dim (int): Dimensionality of the latent bottleneck (Default: 64).
            dropout (float): Dropout probability for regularization (Default: 0.2).
            active_modalities (list): A list of switches used to control ablation experiments.
        """
        super(HeteroAgeHAB, self).__init__()
        
        # Records the currently active modalities; all are enabled by default.
        self.active_modalities = active_modalities or ['beta', 'chalm', 'camda']
        self.modality_dropout_prob = modality_dropout

        self.branch_info = branch_info
        self.num_hallmarks = len(branch_info)
        self.unified_dim = unified_dim
        
        # -----------------------------------------------------------
        # Layer 1: Bio-Sparse Encoder
        # -----------------------------------------------------------
        # Projects tri-modal input (N*3) into pathway-specific channels.
        total_hidden_dim = mask_matrix.shape[1]
        self.sparse_encoder = BioSparseLinear(
            in_features=num_cpgs * 3, 
            out_features=total_hidden_dim,
            mask=mask_matrix
        )
        #self.bn_sparse = nn.BatchNorm1d(total_hidden_dim)
        self.norm_sparse = nn.LayerNorm(total_hidden_dim)
        self.act_sparse = nn.GELU() 

        # -----------------------------------------------------------
        # Layer 2: Hallmark Signaling Cascades
        # -----------------------------------------------------------
        # Independent deep learning modules for each aging hallmark.
        # Depth is auto-configured based on the tiered input dimension (e.g., 1024/512/256).
        self.branches = nn.ModuleList()
        for info in branch_info:
            self.branches.append(
                HallmarkCascade(
                    input_dim=info['dim'],     
                    unified_dim=unified_dim,   
                    dropout=dropout            
                )
            )
        
        # -----------------------------------------------------------
        # Layer 3: Attention Aggregation
        # -----------------------------------------------------------
        # Dynamically weights hallmark contributions based on subject-specific profiles.
        self.attention = AttentionGate(unified_dim)
        
        # -----------------------------------------------------------
        # Layer 4: Deep Prediction Head
        # -----------------------------------------------------------
        # Regresses the aggregated latent embedding to biological age.
        # Updated Architecture: 64 -> 64 -> 32 -> 32 -> 1
        # This structure allows for more complex feature interaction before dimension reduction.
        self.head = nn.Sequential(
            # Stage 1: Isomorphic Refinement (64 -> 64)
            # Allows features to be mixed non-linearly before compression
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Stage 2: Feature Compression (64 -> 32)
            nn.Linear(unified_dim, unified_dim // 2),
            nn.LayerNorm(unified_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            # Stage 3: Deep Feature Abstraction (32 -> 1)
            nn.Linear(unified_dim // 2, 1)
        )

    def forward(self, beta, chalm, camda, return_breakdown=False):
        """
        Forward pass of the network with Ablation Switch and Modality Dropout.
        
        Args:
            beta, chalm, camda (Tensor): Input modalities [Batch, Num_CpGs].
            return_breakdown (bool): If True, returns branch-level scores and weights for interpretability.
        """
        # --- 1. Ablation Switch ---
        if 'beta' not in self.active_modalities:
            beta = torch.zeros_like(beta)
        if 'chalm' not in self.active_modalities:
            chalm = torch.zeros_like(chalm)
        if 'camda' not in self.active_modalities:
            camda = torch.zeros_like(camda)

        # =====================================================================
        # --- 2. Modality Dropout ---
        # =====================================================================
        if self.training and self.modality_dropout_prob > 0:
            if torch.rand(1).item() < self.modality_dropout_prob:
                mod_idx = torch.randint(0, 3, (1,)).item()
                if mod_idx == 0:
                    beta = torch.zeros_like(beta)
                elif mod_idx == 1:
                    chalm = torch.zeros_like(chalm)
                else:
                    camda = torch.zeros_like(camda)
        # =====================================================================

        x = torch.cat([beta, chalm, camda], dim=1)

        sparse_out = self.act_sparse(self.norm_sparse(self.sparse_encoder(x)))

        # 3. Hallmark Cascade Processing
        branch_feats = []
        for idx, info in enumerate(self.branch_info):
            seg_input = sparse_out[:, info['start']:info['end']]
            feat = self.branches[idx](seg_input)
            branch_feats.append(feat.unsqueeze(1))
            
        branch_feats = torch.cat(branch_feats, dim=1)
        
        # 4. Attention-Based Aggregation
        weighted_feat, att_weights = self.attention(branch_feats)
        
        # 5. Final Regression
        total_age = self.head(weighted_feat)
        
        if return_breakdown:
            raw_branch_preds = []
            for i in range(self.num_hallmarks):
                raw_branch_preds.append(self.head(branch_feats[:, i, :]))
            raw_branch_preds = torch.cat(raw_branch_preds, dim=1)
            
            return total_age, {
                'branch_scores': raw_branch_preds,
                'hallmark_weights': att_weights.squeeze(-1),
                'names': [b['name'] for b in self.branch_info]
            }
        
        return total_age, att_weights