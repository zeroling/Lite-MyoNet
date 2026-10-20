"""Checkpoint attention activations without replaying BatchNorm layers."""
import os
import torch
from torch.utils.checkpoint import checkpoint

def attention_forward(module, features):
    if (module.training and torch.is_grad_enabled()
            and os.environ.get('LITE_MYONET_CHECKPOINT_ATTENTION', '1') == '1'):
        # Attention modules use LayerNorm. BatchNorm in the surrounding blocks
        # executes once; dropout RNG is preserved during recomputation.
        return checkpoint(module, features, use_reentrant=False, preserve_rng_state=True)
    return module(features)
