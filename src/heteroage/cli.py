import os
import argparse
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import logging
import optuna
import bitsandbytes as bnb
from tqdm import tqdm
from functools import partial
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

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

    # Command: Assemble
    assemble_parser = subparsers.add_parser('assemble', help="Align and assemble tri-modal data into tensor cache")
    assemble_parser.add_argument('--beta_path', type=str, required=True, help="Path to Beta matrix source")
    assemble_parser.add_argument('--chalm_path', type=str, required=True, help="Path to CHALM matrix source")
    assemble_parser.add_argument('--camda_path', type=str, required=True, help="Path to CAMDA matrix source")
    assemble_parser.add_argument('--split', type=str, default='Train', choices=['Train', 'Test'])

    # Command: Train
    train_parser = subparsers.add_parser('train', help="Training mode")
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--batch_size', type=int, default=128)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--patience', type=int, default=15)
    train_parser.add_argument('--weight_decay', type=float, default=1e-2)
    train_parser.add_argument('--hidden_dim', type=int, default=64, help="Unified latent dimension")
    train_parser.add_argument('--dropout', type=float, default=0.2)
    train_parser.add_argument('--modality_dropout', type=float, default=0.1, help="Probability of dropping an entire modality")
    train_parser.add_argument('--mae_weight', type=float, default=1.0)
    train_parser.add_argument('--rank_weight', type=float, default=1.5)
    train_parser.add_argument('--rank_margin', type=float, default=3.0)
    train_parser.add_argument('--aux_weight', type=float, default=0.3, help="Weight for Deep Supervision Auxiliary Loss") 
    train_parser.add_argument('--val_ratio', type=float, default=0.2)
    train_parser.add_argument('--modalities', nargs='+', default=['beta', 'chalm', 'camda'], choices=['beta', 'chalm', 'camda'])

    # Command: Tune
    tune_parser = subparsers.add_parser('tune', help="Hyperparameter tuning mode")
    tune_parser.add_argument('--n_trials', type=int, default=50)
    tune_parser.add_argument('--study_name', type=str, default="heteroage_optim")
    tune_parser.add_argument('--lr_range', type=float, nargs=2, default=[1e-5, 1e-3])
    tune_parser.add_argument('--weight_decay_range', type=float, nargs=2, default=[1e-4, 1e-1])
    tune_parser.add_argument('--unified_dim_choices', type=int, nargs='+', default=[32, 64])
    tune_parser.add_argument('--dropout_range', type=float, nargs=2, default=[0.15, 0.4])
    tune_parser.add_argument('--modality_dropout_range', type=float, nargs=2, default=[0.0, 0.3])
    tune_parser.add_argument('--batch_size_choices', type=int, nargs='+', default=[64, 128, 256])
    tune_parser.add_argument('--rank_weight_range', type=float, nargs=2, default=[1.0, 3.0])
    tune_parser.add_argument('--aux_weight_range', type=float, nargs=2, default=[0.1, 1.0], help="Search range for aux_weight")
    tune_parser.add_argument('--tune_epochs', type=int, default=20)
    tune_parser.add_argument('--val_ratio', type=float, default=0.2)
    tune_parser.add_argument('--modalities', nargs='+', default=['beta', 'chalm', 'camda'], choices=['beta', 'chalm', 'camda'])

    # Command: Predict
    pred_parser = subparsers.add_parser('predict', help="Inference mode")
    pred_parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    pred_parser.add_argument('--split', type=str, default='Test', choices=['Train', 'Test', 'Val'])
    pred_parser.add_argument('--val_ratio', type=float, default=0.2, help="Ratio used during training to recover Val set")
    pred_parser.add_argument('--batch_size', type=int, default=128) 
    pred_parser.add_argument('--hidden_dim', type=int, required=True)
    pred_parser.add_argument('--dropout', type=float, default=0.0)
    pred_parser.add_argument('--modalities', nargs='+', default=['beta', 'chalm', 'camda'], choices=['beta', 'chalm', 'camda'])
    pred_parser.add_argument('--shuffle', type=str, default=None, choices=['beta', 'chalm', 'camda'])
    
    return parser.parse_args()

def get_feature_list_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_cpgs = set()
    for cpg_list in data.values():
        all_cpgs.update(cpg_list)
    return sorted(list(all_cpgs))

def get_weighted_sampler(dataset):
    tissues = np.array(dataset.dataset.tissues)[dataset.indices] if isinstance(dataset, torch.utils.data.Subset) else np.array(dataset.tissues)
    unique_tissues, counts = np.unique(tissues, return_counts=True)
    weight_per_class = 1.0 / np.sqrt(counts)
    tissue_weight_map = dict(zip(unique_tissues, weight_per_class))
    samples_weight = np.array([tissue_weight_map[t] for t in tissues])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def objective(trial, args, device, master_cpg_list, hallmark_dict, train_ds, val_ds):
    lr = trial.suggest_float('lr', args.lr_range[0], args.lr_range[1], log=True)
    weight_decay = trial.suggest_float('weight_decay', args.weight_decay_range[0], args.weight_decay_range[1], log=True)
    unified_dim = trial.suggest_categorical('unified_dim', args.unified_dim_choices)
    dropout = trial.suggest_float('dropout', args.dropout_range[0], args.dropout_range[1])
    batch_size = trial.suggest_categorical('batch_size', args.batch_size_choices)
    rank_weight = trial.suggest_float('rank_weight', args.rank_weight_range[0], args.rank_weight_range[1])
    modality_dropout = trial.suggest_float('modality_dropout', args.modality_dropout_range[0], args.modality_dropout_range[1])
    aux_weight = trial.suggest_float('aux_weight', args.aux_weight_range[0], args.aux_weight_range[1])
    
    mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
    
    model = HeteroAgeHAB(
        num_cpgs=len(master_cpg_list), 
        branch_info=branch_info, 
        mask_matrix=mask.to(device), 
        unified_dim=unified_dim, 
        dropout=dropout,
        active_modalities=args.modalities,
        modality_dropout=modality_dropout
    ).to(device)
    
    train_sampler = get_weighted_sampler(train_ds)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = HybridAgeLoss(mae_weight=1.0, rank_weight=rank_weight, aux_weight=aux_weight)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(args.tune_epochs):
        model.train()
        for b, c, s, a, t in train_loader:
            b, c, s, a = b.to(device), c.to(device), s.to(device), a.to(device)
            optimizer.zero_grad()
            pred, _ = model(b, c, s)
            loss, _ = criterion(pred, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for b, c, s, a, t in val_loader:
                preds_tuple, _ = model(b.to(device), c.to(device), s.to(device))
                final_pred = preds_tuple[0]
                val_mae += torch.mean(torch.abs(final_pred.flatten() - a.to(device))).item()
        val_mae /= len(val_loader)
        
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    del model, optimizer, criterion
    torch.cuda.empty_cache()

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

    if args.command == 'assemble':
        logger.info(f"Assembling {args.split} data from specified modal paths...")
        TriModalDataset.load_from_directory(
            args.data_root, args.split, ref_cpg_list=master_cpg_list,
            source_paths={'beta': args.beta_path, 'chalm': args.chalm_path, 'camda': args.camda_path}
        )
        logger.info("Assembly complete. Cache generated in data_root.")

    elif args.command in ['train', 'tune']:
        full_train_ds = TriModalDataset.load_from_directory(args.data_root, 'Train', ref_cpg_list=master_cpg_list)
        
        val_size = int(len(full_train_ds) * args.val_ratio)
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        
        if args.command == 'tune':
            obj = partial(objective, args=args, device=device, master_cpg_list=master_cpg_list, 
                          hallmark_dict=hallmark_dict, train_ds=train_ds, val_ds=val_ds)
            db_path = os.path.join(args.output_dir, f"{args.study_name}.db")
            study = optuna.create_study(direction='minimize', study_name=args.study_name, storage=f"sqlite:///{db_path}", load_if_exists=True)
            study.optimize(obj, n_trials=args.n_trials)
            logger.info(f"Best params: {study.best_params}")
            logger.info(f"Best MAE (Validation): {study.best_value:.4f}")
            
        elif args.command == 'train':
            hp_suffix = (f"lr{args.lr}_wd{args.weight_decay}_"
                         f"dim{args.hidden_dim}_dp{args.dropout}_"
                         f"bs{args.batch_size}_rank{args.rank_weight}_"
                         f"moddp{args.modality_dropout}_aux{args.aux_weight}")
            args.output_dir = os.path.join(args.output_dir, hp_suffix)
            os.makedirs(args.output_dir, exist_ok=True)

            mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
            model = HeteroAgeHAB(num_cpgs=len(master_cpg_list), branch_info=branch_info, 
                                 mask_matrix=mask.to(device), unified_dim=args.hidden_dim, 
                                 dropout=args.dropout, 
                                 active_modalities=args.modalities,
                                 modality_dropout=args.modality_dropout).to(device)
            
            criterion = HybridAgeLoss(mae_weight=args.mae_weight, rank_weight=args.rank_weight, rank_margin=args.rank_margin, aux_weight=args.aux_weight)
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )

            train_sampler = get_weighted_sampler(train_ds)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            trainer = HeteroAgeTrainer(model, optimizer, criterion, train_loader, val_loader, scheduler=scheduler, device=device, output_dir=args.output_dir, use_amp=args.use_amp)
            trainer.fit(args.epochs, patience=25)

    elif args.command == 'predict':
        mask, branch_info = construct_biosparse_topology(hallmark_dict, master_cpg_list)
        model = HeteroAgeHAB(num_cpgs=len(master_cpg_list), branch_info=branch_info, 
                             mask_matrix=mask.to(device), unified_dim=args.hidden_dim, dropout=args.dropout, active_modalities=args.modalities).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.eval()

        if args.split == 'Val':
            logger.info(f"Recovering Validation split using seed {args.seed} and ratio {args.val_ratio}...")
            full_train_ds = TriModalDataset.load_from_directory(args.data_root, 'Train', ref_cpg_list=master_cpg_list)
            val_size = int(len(full_train_ds) * args.val_ratio)
            train_size = len(full_train_ds) - val_size
            _, test_ds = random_split(full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        else:
            test_ds = TriModalDataset.load_from_directory(args.data_root, args.split, ref_cpg_list=master_cpg_list)

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        results = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                b, c, s, a, t = batch
                b, c, s, a = b.to(device), c.to(device), s.to(device), a.to(device)

                if args.shuffle == 'beta':
                    b = b[torch.randperm(b.size(0))]
                elif args.shuffle == 'chalm':
                    c = c[torch.randperm(c.size(0))]
                elif args.shuffle == 'camda':
                    s = s[torch.randperm(s.size(0))]

                pred, breakdown = model(b, c, s, return_breakdown=True)
                
                pred_np, true_np = pred.cpu().numpy().flatten(), a.cpu().numpy().flatten()
                scores, weights = breakdown['branch_scores'].cpu().numpy(), breakdown['hallmark_weights'].cpu().numpy()
                
                for i in range(len(true_np)):
                    row = {'Tissue': t[i], 'Predicted_Age': pred_np[i], 'True_Age': true_np[i]}
                    for h_idx, h_name in enumerate(breakdown['names']):
                        row[f'{h_name}_Score'] = scores[i, h_idx]
                        row[f'{h_name}_Weight'] = weights[i, h_idx]
                    results.append(row)
        
        df_results = pd.DataFrame(results)

        detailed_csv_path = os.path.join(args.output_dir, f"detailed_predictions_{args.split}.csv")
        df_results.to_csv(detailed_csv_path, index=False)

        summary_df = df_results.groupby('Tissue').apply(lambda x: pd.Series({
            'MeanAE': mean_absolute_error(x['True_Age'], x['Predicted_Age']),
            'MedAE': median_absolute_error(x['True_Age'], x['Predicted_Age']),
            'MSE': mean_squared_error(x['True_Age'], x['Predicted_Age']),
            'Sample_Count': len(x)
        })).reset_index()
        
        summary_df = summary_df.sort_values(by='MeanAE').reset_index(drop=True)
        summary_csv_path = os.path.join(args.output_dir, f"tissue_metrics_summary_{args.split}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        
        overall_mean_ae = mean_absolute_error(df_results['True_Age'], df_results['Predicted_Age'])
        overall_med_ae = median_absolute_error(df_results['True_Age'], df_results['Predicted_Age'])
        overall_mse = mean_squared_error(df_results['True_Age'], df_results['Predicted_Age'])
        
        logger.info(f"=== Pan-Tissue Evaluation Results ===")
        logger.info(f"Overall MeanAE: {overall_mean_ae:.4f} Years")
        logger.info(f"Overall MedianAE: {overall_med_ae:.4f} Years")
        logger.info(f"Overall MSE: {overall_mse:.4f} Years^2")
        logger.info(f"Detailed predictions saved to {detailed_csv_path}")
        logger.info(f"Tissue-specific summary saved to {summary_csv_path}")
        print("\n", summary_df.to_string())

        pd.DataFrame(results).to_csv(os.path.join(args.output_dir, f"predictions_{args.split}.csv"), index=False)
        logger.info(f"Predictions saved to {args.output_dir}")

if __name__ == '__main__':
    main()