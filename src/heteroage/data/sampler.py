import pandas as pd
import numpy as np
import gc
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def assemble_features(
    cpg_beta: pd.DataFrame, 
    chalm_data: pd.DataFrame, 
    camda_data: pd.DataFrame, 
    pc_data: pd.DataFrame, 
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrates multi-modal data into a unified DataFrame with memory-efficient precision.
    
    Strategy:
        - Early downcasting to float32.
        - In-place column renaming and overlap removal.
        - Strategic garbage collection.
    """
    # 1. Standardize Metadata
    df_base = metadata.copy() # Explicit copy of the small meta-df
    if 'sample_id' not in df_base.columns:
        df_base = df_base.reset_index().rename(columns={df_base.index.name: 'sample_id'})
    df_base['sample_id'] = df_base['sample_id'].astype(str)
    
    # 2. Modality definition with memory-safe suffixes
    modalities = [
        ('beta', cpg_beta, '_beta'),
        ('chalm', chalm_data, '_chalm'),
        ('camda', camda_data, '_camda'),
        ('pc', pc_data, '') 
    ]

    logger.info(f"Assembling features for {len(df_base)} samples...")
    meta_cols_set = {'sample_id', 'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded'}

    for name, df_mod, suffix in modalities:
        if df_mod is None or df_mod.empty or df_mod is metadata: 
            continue

        # Standardize sample_id for the modality
        if 'sample_id' not in df_mod.columns:
            df_mod = df_mod.reset_index().rename(columns={df_mod.index.name: 'sample_id'})
        df_mod['sample_id'] = df_mod['sample_id'].astype(str)

        # In-place Renaming and Downcasting
        rename_map = {
            col: f"{col}{suffix}" for col in df_mod.columns 
            if col not in meta_cols_set and not col.endswith(suffix)
        }
        if rename_map:
            df_mod.rename(columns=rename_map, inplace=True)

        # Downcast float64 to float32 to save 50% memory
        f64_cols = df_mod.select_dtypes(include=['float64']).columns
        if not f64_cols.empty:
            df_mod[f64_cols] = df_mod[f64_cols].astype(np.float32)

        # Drop overlaps except the key to prevent column duplication in merge
        overlap = list(set(df_base.columns).intersection(set(df_mod.columns)) - {'sample_id'})
        if overlap:
            df_mod.drop(columns=overlap, inplace=True)

        # Perform inner join
        df_base = pd.merge(df_base, df_mod, on='sample_id', how='inner')
        
        # Release memory of references
        gc.collect()

    # Final label standardization
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)
    
    return df_base

def filter_and_impute(
    df: pd.DataFrame, 
    features_to_keep: Optional[List[str]] = None, 
    impute_strategy: str = 'median'
) -> pd.DataFrame:
    """
    Filters features and performs chunked imputation to prevent memory exhaustion.
    """
    meta_cols = {'sample_id', 'age', 'project_id', 'Tissue', 'Sex', 'Is_Healthy', 'Sex_encoded'}
    
    # 1. Selective subsetting
    if features_to_keep:
        existing_meta = [c for c in meta_cols if c in df.columns]
        valid_features = [f for f in features_to_keep if f in df.columns]
        # Using list(set(...)) to ensure unique columns
        df_filtered = df[list(set(existing_meta + valid_features))].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty: 
        return df_filtered

    # 2. Identify Imputation Targets
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    cols_to_impute = [c for c in numeric_cols if c not in meta_cols]

    if not cols_to_impute:
        return df_filtered

    logger.info(f"Starting chunked imputation for {len(cols_to_impute)} features...")

    # 3. Chunked Imputation Logic
    chunk_size = 1000
    for i in range(0, len(cols_to_impute), chunk_size):
        chunk = cols_to_impute[i : i + chunk_size]
        
        try:
            # Calculate fill values per chunk
            if impute_strategy == 'median':
                fill_vals = df_filtered[chunk].median().astype(np.float32)
            elif impute_strategy == 'mean':
                fill_vals = df_filtered[chunk].mean().astype(np.float32)
            else:
                fill_vals = 0.0
            
            # fillna(0.0) covers both biological NaNs and columns where median/mean is NaN
            df_filtered.loc[:, chunk] = df_filtered[chunk].fillna(fill_vals).fillna(0.0)
            
        except Exception as e:
            logger.warning(f"Imputation error at chunk {i}: {e}")
            
        if i % 10000 == 0:
            gc.collect()

    logger.info("Feature imputation complete.")
    return df_filtered