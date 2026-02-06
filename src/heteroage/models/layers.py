import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BioSparseLinear(nn.Module):
    """
    [Layer]: Biologically-Constrained Sparse Linear Layer
    
    Enforces a sparsity pattern derived from biological priors. 
    Only connections defined in the mask are allowed to propagate signals.
    """
    def __init__(self, in_features, out_features, mask):
        super(BioSparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Raw weight parameter [Out, In]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Register mask: Transpose from [In, Out] to [Out, In] to match nn.Linear weights
        # Buffer ensures it's saved in checkpoint but not treated as a parameter
        self.register_buffer('mask', mask.t())
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Element-wise multiplication ensures zero gradients for masked connections
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)

    def extra_repr(self):
        active_params = self.mask.sum().item()
        total_params = self.mask.numel()
        return 'in_features={}, out_features={}, sparsity={:.2f}% (active={}/{})'.format(
            self.in_features, self.out_features, 
            100 * (1 - active_params / total_params),
            int(active_params), total_params
        )


class AttentionGate(nn.Module):
    """
    [Layer]: Context-Aware Hallmark Aggregation
    
    Learns importance weights (Alpha) for each aging hallmark to form 
     a unified aging representation.
    """
    def __init__(self, dim):
        super(AttentionGate, self).__init__()
        # Project hallmarks to a shared attention space
        self.attention_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: [Batch, Num_Hallmarks, Dim]
        Returns:
            weighted_sum: [Batch, Dim]
            weights: [Batch, Num_Hallmarks, 1]
        """
        # 1. Compute scores: [Batch, Num_Hallmarks, 1]
        scores = self.attention_net(x)
        
        # 2. Normalize weights per sample
        weights = F.softmax(scores, dim=1)
        
        # 3. Aggregate hallmarks into a single latent vector
        # x * weights: element-wise broading across the Dim axis
        weighted_sum = torch.sum(x * weights, dim=1) 
        
        return weighted_sum, weights