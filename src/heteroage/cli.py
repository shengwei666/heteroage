# src/heteroage/cli.py

import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import logging
import optuna
from tqdm import tqdm
from functools import partial
from torch.utils.data import random_split

from .models import HeteroAgeHAB
from .engine import HeteroAgeTrainer, HybridAgeLoss
from .data import TriModalDataset
from .utils import load_hallmark_dict, construct_biosparse_topology, setup_logger

logger = logging.getLogger("heteroage")

def get_args():
    parser = argparse.ArgumentParser(description="HeteroAge-HAB CLI: Biologically-Inspired Cascade Architecture")
    
    # Global Arguments
    parser.add_argument('--data_root', type=str, required=True, help="Path to data/cache directory")
    parser.add_argument('--hallmark_json', type=str, required=True, help="Path to hallmark definitions")
    parser.add_argument('--output_dir', type=str, default='./output', help="Results directory")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--seed', type=int, default=42)

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Command: Assemble (Data Pre-processing)
    assemble_parser = subparsers.add_parser('assemble', help="Align and assemble tri-modal data into tensor cache")
    assemble_parser.add_argument('--beta_path', type=str, required=True, help="Path to Beta matrix source")
    assemble_parser.add_argument('--chalm_path', type=str, required=True, help="Path to CHALM matrix source")
    assemble_parser.add_argument('--camda_path', type=str, required=True, help="Path to CAMDA matrix source")
    assemble_parser.add_argument('--split', type=str, default='Train', choices=['Train', 'Test'])

    # Command: Train
    train_parser = subparsers.add_parser('train', help="Training mode")
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--patience', type=int, default=15)
    train_parser.add_argument('--weight_decay', type=float, default=1e-2)
    train_parser.add_argument('--hidden_dim', type=int, default=64, help="Unified latent dimension")
    train_parser.add_argument('--dropout', type=float, default=0.2)
    train_parser.add_argument('--mae_weight', type=float, default=1.0)
    train_parser.add_argument('--rank_weight', type=float, default=1.0)
    train_parser.add_argument('--rank_margin', type=float, default=2.0)
    train_parser.add_argument('--val_ratio', type=float, default=0.2)

    # Command: Tune (Optuna)
    tune_parser = subparsers.add_parser('tune', help="Hyperparameter tuning mode")
    tune_parser.add_argument('--n_trials', type=int, default=50)
    tune_parser.add_argument('--study_name', type=str, default="heteroage_optim")
    tune_parser.add_argument('--lr_range', type=float, nargs=2, default=[1e-5, 1e-3])
    tune_parser.add_argument('--unified_dim_choices', type=int, nargs='+', default=[32, 64])
    tune_parser.add_argument('--dropout_range', type=float, nargs=2, default=[0.1, 0.4])
    tune_parser.add_argument('--batch_size_choices', type=int, nargs='+', default=[32, 64])
    tune_parser.add_argument('--rank_weight_range', type=float, nargs=2, default=[0.1, 2.0])
    tune_parser.add_argument('--tune_epochs', type=int, default=20)
    tune_parser.add_argument('--val_ratio', type=float, default=0.2)

    # Command: Predict
    pred_parser = subparsers.add_parser('predict', help="Inference mode")
    pred_parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    pred_parser.add_argument('--split', type=str, default='Test', choices=['Train', 'Test'])
    pred_parser.add_argument('--batch_size', type=int, default=128)
    pred_parser.add_argument('--hidden_dim', type=int, required=True)
    pred_parser.add_argument('--dropout', type=float, default=0.0)

    return parser.parse_args()

def get_feature_list_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_cpgs = set()
    for cpg_list in data.values():
        all_cpgs.update(cpg_list)
    return sorted(list(all_cpgs))

def objective(trial, args, device, master_cpg_list, hallmark_dict, train_ds, val_ds):
    lr = trial.suggest_float('lr', args.lr_range[0], args.lr_range[1], log=True)
    unified_dim = trial.suggest_categorical('unified_dim', args.unified_dim_choices)
    dropout = trial.suggest_float('dropout', args.dropout_range[0], args.dropout_range[1])
    batch_size = trial.suggest_categorical('batch_size', args.batch_size_choices)
    rank_weight = trial.suggest_float('rank_weight', args.rank_weight_range[0], args.rank_weight_range[1])
    
    mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
    
    model = HeteroAgeHAB(
        num_cpgs=len(master_cpg_list), 
        branch_info=branch_info, 
        mask_matrix=mask.to(device), 
        unified_dim=unified_dim, 
        dropout=dropout
    ).to(device)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
    criterion = HybridAgeLoss(mae_weight=1.0, rank_weight=rank_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    for epoch in range(args.tune_epochs):
        model.train()
        for b, c, s, a in train_loader:
            b, c, s, a = b.to(device), c.to(device), s.to(device), a.to(device)
            optimizer.zero_grad()
            pred, _ = model(b, c, s)
            loss, _ = criterion(pred, a)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for b, c, s, a in val_loader:
                pred, _ = model(b.to(device), c.to(device), s.to(device))
                val_mae += torch.mean(torch.abs(pred.flatten() - a.to(device))).item()
        val_mae /= len(val_loader)
        
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_mae

def main():
    args = get_args()
    setup_logger(output_dir=args.output_dir, name="heteroage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Mode: {args.command.upper()} | Device: {device}")

    master_cpg_list = get_feature_list_from_json(args.hallmark_json)
    hallmark_dict = load_hallmark_dict(args.hallmark_json)

    # ==========================
    # Assemble Execution
    # ==========================
    if args.command == 'assemble':
        logger.info(f"Assembling {args.split} data from specified modal paths...")
        # Note: TriModalDataset.load_from_directory handles the logic
        TriModalDataset.load_from_directory(
            args.data_root, args.split, ref_cpg_list=master_cpg_list,
            source_paths={'beta': args.beta_path, 'chalm': args.chalm_path, 'camda': args.camda_path}
        )
        logger.info("Assembly complete. Cache generated in data_root.")

    # ==========================
    # Train / Tune Execution
    # ==========================
    elif args.command in ['train', 'tune']:
        full_train_ds = TriModalDataset.load_from_directory(args.data_root, 'Train', ref_cpg_list=master_cpg_list)
        
        val_size = int(len(full_train_ds) * args.val_ratio)
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        
        if args.command == 'tune':
            obj = partial(objective, args=args, device=device, master_cpg_list=master_cpg_list, 
                          hallmark_dict=hallmark_dict, train_ds=train_ds, val_ds=val_ds)
            study = optuna.create_study(direction='minimize', study_name=args.study_name)
            study.optimize(obj, n_trials=args.n_trials)
            logger.info(f"Best params: {study.best_params}")
            
        elif args.command == 'train':
            mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
            model = HeteroAgeHAB(num_cpgs=len(master_cpg_list), branch_info=branch_info, 
                                 mask_matrix=mask.to(device), unified_dim=args.hidden_dim, dropout=args.dropout).to(device)
            
            criterion = HybridAgeLoss(mae_weight=args.mae_weight, rank_weight=args.rank_weight, rank_margin=args.rank_margin)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            trainer = HeteroAgeTrainer(model, train_ds, val_ds, criterion, optimizer, device, args.output_dir, use_amp=args.use_amp)
            trainer.fit(args.epochs, args.patience, args.batch_size, args.num_workers)

    # ==========================
    # Inference Execution
    # ==========================
    elif args.command == 'predict':
        mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
        model = HeteroAgeHAB(num_cpgs=len(master_cpg_list), branch_info=branch_info, 
                             mask_matrix=mask.to(device), unified_dim=args.hidden_dim, dropout=args.dropout).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.eval()

        test_ds = TriModalDataset.load_from_directory(args.data_root, args.split, ref_cpg_list=master_cpg_list)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        results = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                b, c, s, a = [t.to(device) for t in batch]
                pred, breakdown = model(b, c, s, return_breakdown=True)
                
                pred_np, true_np = pred.cpu().numpy().flatten(), a.cpu().numpy().flatten()
                scores, weights = breakdown['branch_scores'].cpu().numpy(), breakdown['hallmark_weights'].cpu().numpy()
                
                for i in range(len(true_np)):
                    row = {'Predicted_Age': pred_np[i], 'True_Age': true_np[i]}
                    for h_idx, h_name in enumerate(breakdown['names']):
                        row[f'{h_name}_Score'] = scores[i, h_idx]
                        row[f'{h_name}_Weight'] = weights[i, h_idx]
                    results.append(row)

        pd.DataFrame(results).to_csv(os.path.join(args.output_dir, f"predictions_{args.split}.csv"), index=False)
        logger.info(f"Predictions saved to {args.output_dir}")

if __name__ == '__main__':
    main()