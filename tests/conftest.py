import pytest
import torch
import numpy as np

@pytest.fixture
def dummy_input():
    """
    [Fixture]: Generates a micro-batch of tri-modal epigenetic data.
    - Batch Size: 4
    - Num CpGs: 100
    - Modalities: Beta, CHALM, CAMDA
    """
    batch_size = 4
    num_cpgs = 100
    
    # Simulate normalized epigenetic signals [0, 1]
    beta = torch.rand(batch_size, num_cpgs)
    chalm = torch.rand(batch_size, num_cpgs)
    camda = torch.rand(batch_size, num_cpgs)
    
    # Chronological age labels
    age = torch.tensor([20.0, 40.0, 60.0, 80.0])
    
    return beta, chalm, camda, age, num_cpgs

@pytest.fixture
def dummy_topology(dummy_input):
    """
    [Fixture]: Generates a tiered biological topology for testing.
    Simulates the output of 'construct_biosparse_topology' from bio_utils.
    
    Structure:
        - Hallmark A (Tier 3): 256 dim
        - Hallmark B (Tier 3): 256 dim
    Total Sparse Output: 512
    """
    _, _, _, _, num_cpgs = dummy_input
    input_dim = num_cpgs * 3  # Tri-modal input
    
    # Define simplified branch metadata for testing
    branch_info = [
        {'name': 'Pathway_A', 'dim': 256, 'start': 0, 'end': 256},
        {'name': 'Pathway_B', 'dim': 256, 'start': 256, 'end': 512}
    ]
    total_hidden = 512
    
    # Binary mask: [Input_Dim, Total_Hidden]
    # Encodes biological priors (Sparse connectivity)
    mask = torch.randint(0, 2, (input_dim, total_hidden)).float()
    
    return mask, branch_info