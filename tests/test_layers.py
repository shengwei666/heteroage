import torch
import torch.nn as nn
from heteroage.models.layers import BioSparseLinear, AttentionGate

def test_bio_sparse_linear_masking():
    """
    [Test]: Gradient Integrity Check for BioSparseLinear.
    Ensures that gradients are strictly zeroed for biologically invalid connections.
    """
    in_features = 10
    out_features = 5
    
    # Define Topology: Only the first 2 features connect to the output
    # Input Mask shape should match BioSparseLinear expectation: [in_features, out_features]
    mask = torch.zeros(in_features, out_features)
    mask[:2, :] = 1.0 
    
    layer = BioSparseLinear(in_features, out_features, mask=mask)
    
    # Dummy forward & backward pass
    x = torch.randn(2, in_features)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Access the weight gradient [out_features, in_features]
    grad = layer.weight.grad
    
    # 1. Verify Active Connections: Should have non-zero gradients
    # Note: BioSparseLinear transposes mask to [out, in], so indices are flipped
    assert torch.all(grad[:, :2] != 0), "Active pathways must propagate gradients."
    
    # 2. Verify Masked Connections: Gradients MUST be exactly zero
    assert torch.all(grad[:, 2:] == 0), "Masked connections must have zero gradient to ensure sparsity."
    
    print("BioSparseLinear Gradient Check: PASSED")

def test_attention_gate_aggregation():
    """
    [Test]: Functional check for AttentionGate.
    Verifies context-aware aggregation from [Batch, Num_Hallmarks, Dim] to [Batch, Dim].
    """
    batch = 4
    num_hallmarks = 12
    dim = 64
    
    # Input: Latent embeddings from 12 HallmarkCascades
    x = torch.randn(batch, num_hallmarks, dim)
    
    gate = AttentionGate(dim)
    weighted_sum, weights = gate(x)
    
    # 1. Verify Output Dimensions
    # Should collapse the Hallmark dimension via weighted sum
    assert weighted_sum.shape == (batch, dim), "Output should be a single latent vector per sample."
    
    # 2. Verify Weight Dimensions
    # Should provide one weight per hallmark
    assert weights.shape == (batch, num_hallmarks, 1), "Attention weights shape mismatch."
    
    # 3. Verify Softmax Constraints
    # Weights across hallmarks must sum to 1.0
    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums)), "Attention weights must sum to 1.0."

    print("AttentionGate Functional Check: PASSED")

if __name__ == "__main__":
    test_bio_sparse_linear_masking()
    test_attention_gate_aggregation()