import os
import torch
import logging
import time
import numpy as np
import optuna
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

logger = logging.getLogger(__name__)

class HeteroAgeTrainer:
    """
    [Engine]: HeteroAge-HAB Training & Evaluation System (Deep Supervision Ready)
    """
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        scheduler=None,
        device='cuda',
        output_dir='./output',
        use_amp=True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        
        self.scaler = GradScaler() if use_amp else None
        os.makedirs(output_dir, exist_ok=True)
        
        self.best_val_mae = float('inf')
        self.start_epoch = 0
        ## Initialize dictionary
        self.history = {'train_loss': [], 'val_mae': []}

    def _run_epoch(self, epoch_idx, is_train=True):
        self.model.train() if is_train else self.model.eval()
        loader = self.train_loader if is_train else self.val_loader
        
        running_metrics = {"loss": 0.0, "mae": 0.0, "rank": 0.0, "aux": 0.0}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx}", ncols=110, leave=False, disable=not is_train)
        
        for batch in pbar:
            beta = batch[0].to(self.device, non_blocking=True)
            chalm = batch[1].to(self.device, non_blocking=True)
            camda = batch[2].to(self.device, non_blocking=True)
            age = batch[3].to(self.device, non_blocking=True)
            
            with torch.set_grad_enabled(is_train):
                with autocast(device_type='cuda', enabled=self.use_amp):
                    preds, _ = self.model(beta, chalm, camda)
                    loss, metrics = self.criterion(preds, age)

            if is_train:
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # Metric Accumulation
            running_metrics["loss"] += loss.item()
            running_metrics["mae"] += metrics.get('loss_mae', 0.0)
            running_metrics["rank"] += metrics.get('loss_rank', 0.0)
            running_metrics["aux"] += metrics.get('loss_aux', 0.0)
            
            if is_train:
                pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'MAE': f"{metrics.get('loss_mae', 0.0):.3f}", 'Aux': f"{metrics.get('loss_aux', 0.0):.3f}"})

        n = len(loader)
        return {k: v / n for k, v in running_metrics.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_targets = [], []
        
        for batch in loader:
            beta = batch[0].to(self.device, non_blocking=True)
            chalm = batch[1].to(self.device, non_blocking=True)
            camda = batch[2].to(self.device, non_blocking=True)
            age = batch[3].to(self.device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=self.use_amp):
                preds_tuple, _ = self.model(beta, chalm, camda)
            
            final_pred = preds_tuple[0]
            
            all_preds.append(final_pred.cpu().float().numpy())
            all_targets.append(age.cpu().float().numpy())
            
        y_pred = np.concatenate(all_preds).flatten()
        y_true = np.concatenate(all_targets).flatten()
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'pearson': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        }

    def fit(self, epochs, patience=10, trial=None):
        patience_counter = 0
        
        for epoch in range(self.start_epoch, epochs):
            start_t = time.time()
            
            # Training Phase
            train_results = self._run_epoch(epoch + 1, is_train=True)
            
            # Validation Phase
            val_results = self.evaluate(self.val_loader)
            curr_mae = val_results['mae']
            
            # Optuna Pruning
            if trial:
                trial.report(curr_mae, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Scheduler Update
            if self.scheduler:
                self.scheduler.step(curr_mae) if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()

            # Logging & Checkpointing
            duration = time.time() - start_t
            if not trial:
                logger.info(f"Epoch {epoch+1}/{epochs} | {duration:.1f}s | Train Loss: {train_results['loss']:.3f} (Aux: {train_results['aux']:.3f}) | Val MeanAE: {curr_mae:.4f} | Val MedAE: {val_results['median_ae']:.4f} | R2: {val_results['r2']:.4f}")
                
                self.history['train_loss'].append(train_results['loss'])
                self.history['val_mae'].append(curr_mae)

            if curr_mae < self.best_val_mae:
                self.best_val_mae = curr_mae
                patience_counter = 0
                if not trial:
                    self.save_checkpoint("best_model.pth", epoch + 1, curr_mae)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if not trial: logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        
        return self.best_val_mae

    def save_checkpoint(self, filename, epoch, val_mae):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_mae': val_mae,
        }, path)