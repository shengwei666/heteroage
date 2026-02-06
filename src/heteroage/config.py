import torch
import os
import platform

class HeteroAgeConfig:
    """
    [Registry]: Centralized configuration for HeteroAge-HAB.
    
    This class manages biological prior constants, architectural hyperparameters, 
    and hardware-dependent execution settings to ensure experimental consistency.
    """

    # --- 1. Biological Tiering Constraints ---
    # Defines how hallmark pathways are mapped to network complexity
    TIER_CONFIG = {
        'tier1': {'threshold': 12000, 'dim': 1024, 'label': 'High-Complexity'},
        'tier2': {'threshold': 4000,  'dim': 512,  'label': 'Medium-Complexity'},
        'tier3': {'threshold': 0,     'dim': 256,  'label': 'Specific-Pathway'}
    }

    # --- 2. Architectural Bottlenecks ---
    # The latent space where all hallmark features are fused via Attention
    UNIFIED_DIM = 64     
    DROPOUT = 0.2
    
    # Loss function constants
    RANK_MARGIN = 2.0    # Margin for ordinal ranking loss (unit: years)
    MAE_WEIGHT = 1.0     # Relative weight of regression loss
    RANK_WEIGHT = 1.0    # Relative weight of manifold consistency loss

    # --- 3. Compute Environment ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use Automated Mixed Precision for GPU, disable for CPU
    USE_AMP = True if torch.cuda.is_available() else False
    
    # Adaptive CPU worker allocation
    NUM_WORKERS = 4 if platform.system() == 'Windows' else 8

    # --- 4. Filesystem & Artifacts ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments")

    @classmethod
    def get_tier(cls, cpg_count: int) -> dict:
        """Dynamically assigns a hallmark to its corresponding TIER based on CpG count."""
        if cpg_count >= cls.TIER_CONFIG['tier1']['threshold']:
            return cls.TIER_CONFIG['tier1']
        elif cpg_count >= cls.TIER_CONFIG['tier2']['threshold']:
            return cls.TIER_CONFIG['tier2']
        return cls.TIER_CONFIG['tier3']

    @classmethod
    def summary(cls) -> str:
        """Returns a string representation of the current configuration for logging."""
        return (f"\n{'='*40}\n"
                f"HeteroAge-HAB Configuration Summary\n"
                f"{'-'*40}\n"
                f"Device:       {cls.DEVICE}\n"
                f"AMP:          {cls.USE_AMP}\n"
                f"Unified Dim:  {cls.UNIFIED_DIM}\n"
                f"Rank Margin:  {cls.RANK_MARGIN}\n"
                f"Tier Dims:    {[v['dim'] for v in cls.TIER_CONFIG.values()]}\n"
                f"{'='*40}")