import torch
import pytest
from heteroage.models.habnet import HeteroAgeHAB

def test_habnet_forward():
    """
    [Test]: Functional validation of HeteroAge-HAB forward pass.
    Verifies output dimensions for both standard regression and interpretability modes.
    """
    batch_size = 4
    num_cpgs = 100
    num_hallmarks = 3
    unified_dim = 64
    
    # 1. Mock biological metadata (Simulating branch_info from bio_utils)
    # Each hallmark has its own segment in the sparse hidden layer
    branch_info = [
        {'name': 'Epigenetic', 'dim': 1024, 'start': 0, 'end': 1024},
        {'name': 'Inflammation', 'dim': 512, 'start': 1024, 'end': 1536},
        {'name': 'Proteostasis', 'dim': 256, 'start': 1536, 'end': 1792}
    ]
    total_sparse_dim = 1792
    
    # 2. Mock inputs and Bio-Sparse Mask
    beta = torch.randn(batch_size, num_cpgs)
    chalm = torch.randn(batch_size, num_cpgs)
    camda = torch.randn(batch_size, num_cpgs)
    
    # Mask shape: [In_Features (3 * num_cpgs), Out_Features (total_sparse_dim)]
    mask = torch.ones(num_cpgs * 3, total_sparse_dim) 

    # 3. Model Initialization
    model = HeteroAgeHAB(
        num_cpgs=num_cpgs,
        branch_info=branch_info,
        mask_matrix=mask,
        unified_dim=unified_dim
    )

    # --- Case 1: Standard Regression ---
    age_pred, att_weights = model(beta, chalm, camda)
    
    assert age_pred.shape == (batch_size, 1), "Regression output dimension mismatch"
    assert att_weights.shape == (batch_size, num_hallmarks, 1), "Attention weights dimension mismatch"

    # --- Case 2: Interpretability (Breakdown) Mode ---
    total_age, breakdown = model(beta, chalm, camda, return_breakdown=True)
    
    assert 'branch_scores' in breakdown, "Breakdown mode should return pathway-specific scores"
    # branch_scores: [Batch, Num_Hallmarks]
    assert breakdown['branch_scores'].shape == (batch_size, num_hallmarks)
    # hallmark_weights: [Batch, Num_Hallmarks]
    assert breakdown['hallmark_weights'].shape == (batch_size, num_hallmarks)
    
    print("HeteroAge-HAB forward pass: PASSED")

if __name__ == "__main__":
    test_habnet_forward()