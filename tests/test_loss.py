import torch
import pytest
from heteroage.engine.losses import RankConsistentLoss

def test_rank_loss_logic():
    """
    [Test]: Verifies the Pairwise Ranking Consistency logic.
    Ensures that ordinal violations are penalized while maintaining 
    a zero-loss state for correct ordering with sufficient margin.
    """
    margin = 2.0
    criterion = RankConsistentLoss(margin=margin)
    
    # ---------------------------------------------------------
    # Case 1: Ideal Ordering (Correct order + Sufficient margin)
    # Target: A=20, B=40 (B > A)
    # Pred:   A=20, B=40 (B - A = 20 > Margin)
    # ---------------------------------------------------------
    target_good = torch.tensor([20.0, 40.0], requires_grad=True)
    pred_good = torch.tensor([20.0, 40.0], requires_grad=True)
    
    loss_good = criterion(pred_good, target_good)
    assert loss_good.item() == 0, "Perfect ranking with margin should yield zero loss."
    
    # ---------------------------------------------------------
    # Case 2: Margin Violation (Correct order but too close)
    # Target: A=20, B=40 (B > A)
    # Pred:   A=20, B=21 (B - A = 1 < Margin)
    # ---------------------------------------------------------
    pred_close = torch.tensor([20.0, 21.0], requires_grad=True)
    loss_close = criterion(pred_close, target_good)
    assert loss_close.item() > 0, "Correct ordering but failing margin must be penalized."

    # ---------------------------------------------------------
    # Case 3: Ordinal Inversion (Major Error)
    # Target: A=20, B=40 (B > A)
    # Pred:   A=40, B=20 (B < A)
    # ---------------------------------------------------------
    pred_bad = torch.tensor([40.0, 20.0], requires_grad=True)
    loss_bad = criterion(pred_bad, target_good)
    assert loss_bad.item() > loss_close.item(), "Rank inversion should incur the highest penalty."

    # ---------------------------------------------------------
    # Case 4: Gradient Traceability
    # ---------------------------------------------------------
    loss_bad.backward()
    assert pred_bad.grad is not None, "Gradients must propagate back to predictions."

    print("RankConsistentLoss Logic Check: PASSED")

if __name__ == "__main__":
    test_rank_loss_logic()