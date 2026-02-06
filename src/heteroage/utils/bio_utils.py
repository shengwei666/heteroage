# src/heteroage/utils/bio_utils.py

import torch
import json
import logging
import numpy as np
from collections import OrderedDict

logger = logging.getLogger("heteroage")

# === Pathway Complexity Configuration ===
# Revised thresholds based on actual Hallmark-CpG distribution.
# Grouping Logic:
# - Tier 1 (>12k): Epigenetic(35k) ... Proteostasis(14k) -> High Capacity (1024 dim)
# - Tier 2 (>4k):  Inflammation(10k) ... Nutrient(8.4k)  -> Med Capacity (512 dim)
# - Tier 3 (<4k):  Stem Cell(3.9k) ... Dysbiosis(1k)     -> Low Capacity (256 dim)
TIER_CONFIG = {
    'Tier1': {'threshold': 12000, 'dim': 1024, 'desc': 'Broad Systemic'},
    'Tier2': {'threshold': 4000,  'dim': 512,  'desc': 'Intermediate Process'},
    'Tier3': {'threshold': 0,     'dim': 256,  'desc': 'Specific Mechanism'}
}

def load_pathway_definitions(json_path):
    """
    Parses the Hallmark-CpG association map from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def construct_biosparse_topology(hallmark_dict, master_cpg_list):
    """
    Constructs the sparse connectivity matrix (Topology) for the BioSparse Layer.
    
    This function maps the biological prior knowledge (Hallmark -> CpGs) into a 
    binary mask tensor and determines the structural complexity (Tier) for each branch.

    Args:
        hallmark_dict (dict): Map of {Hallmark_Name: [CpG_List]}.
        master_cpg_list (list): The global ordered list of input CpGs.

    Returns:
        mask_tensor (Tensor): Binary mask [3*N_CpGs, Total_Hidden_Dim].
        branch_metadata (list): Configuration for each HallmarkCascade branch.
    """
    # 1. Indexing: Create a lookup map for global CpG positions
    cpg_to_idx = {cpg: i for i, cpg in enumerate(master_cpg_list)}
    num_cpgs = len(master_cpg_list)
    
    branch_metadata = []
    mask_parts = []
    
    current_start_idx = 0
    
    # 2. Topology Construction
    # Sort hallmarks by name to ensure deterministic structure
    for name in sorted(hallmark_dict.keys()):
        related_cpgs = hallmark_dict[name]
        
        # A. Determine Biological Complexity (Tiering)
        # Calculate overlap with master list to assess effective pathway size
        valid_cpgs = [c for c in related_cpgs if c in cpg_to_idx]
        count = len(valid_cpgs)
        
        if count >= TIER_CONFIG['Tier1']['threshold']:
            tier = TIER_CONFIG['Tier1']
        elif count >= TIER_CONFIG['Tier2']['threshold']:
            tier = TIER_CONFIG['Tier2']
        else:
            tier = TIER_CONFIG['Tier3']
            
        hidden_dim = tier['dim']
        
        # Log the classification for verification
        # logger.info(f"Hallmark '{name}' ({count} CpGs) -> {tier['desc']} (Dim: {hidden_dim})")
        
        # B. Metadata Logging
        branch_metadata.append({
            'name': name,
            'dim': hidden_dim,        # Input dimension for HallmarkCascade
            'start': current_start_idx,
            'end': current_start_idx + hidden_dim,
            'tier': tier['desc']
        })
        
        # C. Sparse Mask Generation
        # Create a binary matrix for this specific branch [3*N_CpGs, Branch_Dim]
        # We use a randomized sparse projection (conceptually similar to compressed sensing)
        branch_mask = torch.zeros(num_cpgs * 3, hidden_dim)
        
        if count > 0:
            # Map valid CpGs to indices
            cpg_indices = [cpg_to_idx[c] for c in valid_cpgs]
            
            # Apply to all 3 modalities (Beta, CHALM, CAMDA)
            # Modality offsets: 0, num_cpgs, 2*num_cpgs
            for mod_offset in [0, num_cpgs, 2 * num_cpgs]:
                # Connect each input CpG to ~3 random neurons in the hidden layer
                # This ensures robust signal propagation without full connectivity
                rows = [i + mod_offset for i in cpg_indices]
                
                # Randomly assign connections within the branch's allocated width
                cols = np.random.randint(0, hidden_dim, size=len(rows) * 3) 
                
                # Repeat rows to match the 1-to-3 connection ratio
                rows_expanded = np.repeat(rows, 3)
                
                # Set connections
                branch_mask[rows_expanded, cols] = 1.0
        
        mask_parts.append(branch_mask)
        current_start_idx += hidden_dim
        
    # 3. Assembly
    # Concatenate all branch masks to form the global BioSparse Matrix
    final_mask = torch.cat(mask_parts, dim=1)
    
    return final_mask, branch_metadata

def setup_logger(output_dir, name="heteroage"):
    """Configures the experimental logging system."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s', 
        datefmt='%m-%d %H:%M'
    )
    
    # Stream Handler (Console)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(f"{output_dir}/experiment.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger