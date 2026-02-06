import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BioSparseLinear, AttentionGate

class HallmarkCascade(nn.Module):
    """
    [Module]: Hallmark Signaling Cascade
    
    A hierarchical feature extraction module inspired by cellular signaling cascades.
    It implements a "Refinement-Compression" cycle to progressively distill high-dimensional 
    epigenetic signals into a compact latent representation.
    
    Architecture:
        - Iterative process: (Linear -> GELU -> Dropout) x 2 per stage.
        - Stage 1 (Refinement): Isomorphic transformation to model non-linear interactions at current resolution.
        - Stage 2 (Compression): Downsampling operation to extract abstract features.
        - The depth is dynamically determined by the input complexity (Tiered Dimension).
    """
    def __init__(self, input_dim, unified_dim, dropout=0.2):
        super(HallmarkCascade, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Dynamic depth construction based on input complexity
        while current_dim > unified_dim:
            # Phase A: Signal Refinement (Isomorphic)
            layers.append(nn.Linear(current_dim, current_dim))
            layers.append(nn.GELU()) 
            layers.append(nn.Dropout(dropout))
            
            # Phase B: Signal Compression (Downsampling)
            next_dim = max(unified_dim, current_dim // 2)
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.GELU()) 
            layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
            
        # Phase C: Latent Space Alignment
        # Ensures features are projected into the shared unified space for Attention
        layers.append(nn.Linear(unified_dim, unified_dim))
        layers.append(nn.GELU())     
        layers.append(nn.Dropout(dropout)) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    def __init__(self, num_cpgs, branch_info, mask_matrix, unified_dim=64, dropout=0.2):
        """
        Args:
            num_cpgs (int): Count of unique CpGs in the master feature list.
            branch_info (list): Metadata defining the slice indices for each hallmark branch.
            mask_matrix (Tensor): Binary mask enforcing biological sparsity constraints.
            unified_dim (int): Dimensionality of the latent bottleneck (Default: 64).
            dropout (float): Dropout probability for regularization (Default: 0.2).
        """
        super(HeteroAgeHAB, self).__init__()
        
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
        self.bn_sparse = nn.BatchNorm1d(total_hidden_dim)
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
        self.head = nn.Sequential(
            nn.Linear(unified_dim, unified_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(unified_dim // 2, unified_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(unified_dim // 4, 1)
        )

    def forward(self, beta, chalm, camda, return_breakdown=False):
        """
        Forward pass of the network.
        
        Args:
            beta, chalm, camda (Tensor): Input modalities [Batch, Num_CpGs].
            return_breakdown (bool): If True, returns branch-level scores and weights for interpretability.
        """
        # 1. Multi-Modal Fusion
        x = torch.cat([beta, chalm, camda], dim=1)
        
        # 2. Sparse Encoding & Normalization
        sparse_out = self.act_sparse(self.bn_sparse(self.sparse_encoder(x)))
        
        # 3. Hallmark Cascade Processing
        branch_feats = []
        for idx, info in enumerate(self.branch_info):
            # Isolate data for the specific hallmark pathway
            seg_input = sparse_out[:, info['start']:info['end']]
            # Extract deep features via Cascade
            feat = self.branches[idx](seg_input)
            branch_feats.append(feat.unsqueeze(1))
            
        # Shape: [Batch, Num_Hallmarks, Unified_Dim]
        branch_feats = torch.cat(branch_feats, dim=1)
        
        # 4. Attention-Based Aggregation
        weighted_feat, att_weights = self.attention(branch_feats)
        
        # 5. Final Regression
        total_age = self.head(weighted_feat)
        
        if return_breakdown:
            # Generate interpretability metrics (Branch Scores & Attention Weights)
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