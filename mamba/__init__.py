"""
Mamba module - State Space Model implementations

This module provides:
- Mamba2: SSD (State Space Duality) formulation
- Mamba3: Trapezoidal discretization with complex SSM
- Jamba: Hybrid Mamba-Transformer architecture
"""

from .mamba2 import (
    Mamba2Config,
    Mamba2Model,
    Mamba2Block,
    Mamba2Mixer,
    create_mamba2,
)

from .mamba3_triton import (
    Mamba3Config,
    Mamba3Model,
    Mamba3Block,
    Mamba3Mixer,
    create_mamba3,
)

from .jamba import (
    JambaConfig,
    JambaModel,
    JambaMambaBlock,
    JambaAttentionBlock,
    create_jamba,
    get_block_pattern,
)

__all__ = [
    # Mamba2
    "Mamba2Config",
    "Mamba2Model",
    "Mamba2Block",
    "Mamba2Mixer",
    "create_mamba2",
    # Mamba3
    "Mamba3Config",
    "Mamba3Model",
    "Mamba3Block",
    "Mamba3Mixer",
    "create_mamba3",
    # Jamba
    "JambaConfig",
    "JambaModel",
    "JambaMambaBlock",
    "JambaAttentionBlock",
    "create_jamba",
    "get_block_pattern",
]
