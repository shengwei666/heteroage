import os
import glob
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TriModalDataset(Dataset):
    """
    Dataset class for HeteroAge-HAB handling Beta, CHALM, and CAMDA modalities.
    """
    def __init__(self, beta, chalm, camda, age, sample_ids, feature_names=None):
        self.beta = beta
        self.chalm = chalm
        self.camda = camda
        self.age = age
        self.sample_ids = sample_ids
        self.feature_names = feature_names
        
    def __len__(self):
        return len(self.beta)

    def __getitem__(self, idx):
        return self.beta[idx], self.chalm[idx], self.camda[idx], self.age[idx]

    @staticmethod
    def load_from_directory(data_root, split, ref_cpg_list=None, use_cache=True, source_paths=None):
        cache_dir = os.path.join(data_root, "cached_tensor_data")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"merged_{split}.pt")
        
        # --- 1. Cache Recovery ---
        if use_cache and os.path.exists(cache_file) and source_paths is None:
            logger.info(f"Loading cached dataset: {cache_file}")
            try:
                cached_data = torch.load(cache_file)
                if ref_cpg_list is not None and cached_data.get('feature_names') is not None:
                    if cached_data['feature_names'] != ref_cpg_list:
                        logger.warning("Cache feature mismatch. Re-assembling from source.")
                    else:
                        return TriModalDataset(**cached_data)
                else:
                    return TriModalDataset(
                        beta=cached_data['beta'], chalm=cached_data['chalm'], 
                        camda=cached_data['camda'], age=cached_data['age'], 
                        sample_ids=cached_data['sample_ids'], feature_names=ref_cpg_list
                    )
            except Exception as e:
                logger.warning(f"Cache access failed: {e}")

        # --- 2. Path Discovery & Mode Selection ---
        logger.info(f"Assembling data from modal sources for {split} split")
    
        if source_paths and os.path.isfile(source_paths['beta']):
            logger.info("Direct file paths detected. Bypassing directory search.")
            matched_files = [(source_paths['beta'], source_paths['chalm'], source_paths['camda'])]
        else:
            if source_paths:
                dir_beta = source_paths['beta']
                dir_chalm = source_paths['chalm']
                dir_camda = source_paths['camda']
            else:
                dir_beta = os.path.join(data_root, "Beta_Matrix", split)
                dir_chalm = os.path.join(data_root, "Chalm_Matrix", split)
                dir_camda = os.path.join(data_root, "Camda_Matrix", split)
                
            beta_files = sorted(glob.glob(os.path.join(dir_beta, "*.pkl")))
            if not beta_files:
                raise FileNotFoundError(f"No source .pkl files found in {dir_beta}")
                
            matched_files = []
            for f_beta in beta_files:
                fname = os.path.basename(f_beta)
                matched_files.append((f_beta, os.path.join(dir_chalm, fname), os.path.join(dir_camda, fname)))

        # --- 3. Multi-Modal Alignment Loop ---
        list_beta, list_chalm, list_camda = [], [], []
        list_ages, list_ids = [], []

        for f_beta, f_chalm, f_camda in tqdm(matched_files, desc=f"Aligning {split}"):
            if not (os.path.exists(f_chalm) and os.path.exists(f_camda)):
                logger.warning(f"Missing matching files for {f_beta}, skipping.")
                continue
                
            try:
                df_b, df_c, df_s = pd.read_pickle(f_beta), pd.read_pickle(f_chalm), pd.read_pickle(f_camda)
            except:
                continue
                
            # Modal intersection (Sample Alignment)
            common = df_b.index.intersection(df_c.index).intersection(df_s.index)
            if common.empty or 'Age' not in df_b.columns:
                continue
            
            df_b, df_c, df_s = df_b.loc[common], df_c.loc[common], df_s.loc[common]
            
            # Metadata extraction
            ages = df_b['Age'].astype(float).values
            ids = df_b.index.astype(str).values
            
            # Feature alignment (Hallmark Reindexing)
            meta_cols = {'project_id', 'Tissue', 'Age', 'Sex', 'Set', 'Is_Healthy', 'Set_Group', 'sample_id'}
            def clean(df): 
                m = df.drop(columns=[c for c in meta_cols if c in df.columns])
                if ref_cpg_list is not None:
                    m = m.reindex(columns=ref_cpg_list, fill_value=0.0)
                return m.values.astype(np.float32)

            list_beta.append(clean(df_b))
            list_chalm.append(clean(df_c))
            list_camda.append(clean(df_s))
            list_ages.extend(ages)
            list_ids.extend(ids)

        if not list_beta:
            raise RuntimeError("Data assembly failed: No aligned samples found.")

        # --- 4. Tensor Aggregation & Validation ---
        final_beta = torch.tensor(np.vstack(list_beta))
        final_chalm = torch.tensor(np.vstack(list_chalm))
        final_camda = torch.tensor(np.vstack(list_camda))
        final_age = torch.tensor(np.array(list_ages, dtype=np.float32))
        
        logger.info(f"Assembly Success: {final_beta.shape[0]} samples, {final_beta.shape[1]} features.")
        
        # --- 5. Persistence ---
        if use_cache:
            cache_dict = {
                'beta': final_beta, 'chalm': final_chalm, 'camda': final_camda,
                'age': final_age, 'sample_ids': list_ids, 'feature_names': ref_cpg_list
            }
            torch.save(cache_dict, cache_file)
            logger.info(f"Saved tensor cache to {cache_file}")

        return TriModalDataset(final_beta, final_chalm, final_camda, final_age, list_ids, ref_cpg_list)