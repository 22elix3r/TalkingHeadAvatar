#!/usr/bin/env python3
"""
Resume VHAP global refinement from OOM point with memory optimizations.

Key optimizations:
- batch_size reduced from 7 to 4 to lower peak VRAM usage
- async_func disabled to avoid stacking async callbacks and tensors
- Skips sequential tracking (begin_timestep=8556)
- Uses pre-tracked flame_params
"""

import sys
import os
from pathlib import Path
import yaml
import torch
from dataclasses import replace

# Add VHAP to path
sys.path.append(os.path.abspath("VHAP"))

from vhap.config.base import BaseTrackingConfig
from vhap.model.tracker import GlobalTracker


# =============================================================================
# Safe YAML Loader for Tyro-serialized configs (with Python object tags)
# =============================================================================

class _SafeTrackingConfigLoader(yaml.SafeLoader):
    """
    Safe YAML loader that supports a constrained subset of Python-tagged Tyro dumps.
    Only allows Path objects; everything else is converted to plain dicts/sequences.
    """
    pass


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_python_object(loader, suffix, node):
    # Convert tagged python objects into plain mappings/sequences/scalars.
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise yaml.constructor.ConstructorError(
        None,
        None,
        f"Unsupported YAML node for python/object:{suffix}",
        node.start_mark,
    )


def _construct_python_apply(loader, suffix, node):
    # Explicitly allow only pathlib constructors used in VHAP config dumps.
    if suffix in {"pathlib.PosixPath", "pathlib.WindowsPath", "pathlib.Path"}:
        args = loader.construct_sequence(node, deep=True)
        return Path(*args)
    raise yaml.constructor.ConstructorError(
        None,
        None,
        f"Unsupported YAML python/object/apply: {suffix}",
        node.start_mark,
    )


# Register constructors
_SafeTrackingConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)
_SafeTrackingConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:",
    _construct_python_object,
)
_SafeTrackingConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:",
    _construct_python_object,
)
_SafeTrackingConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:",
    _construct_python_apply,
)

def resume_with_reduced_batch():
    # Path to original config
    config_path = Path("output/sam_altman/vhap_track/2026-05-05_23-47-27/config.yml")
    tracked_params_path = Path("output/sam_altman/vhap_track/2026-05-05_23-47-27/tracked_flame_params_0.npz")
    
    print(f"\n{'='*70}")
    print(f"VHAP Global Refinement - Resume with Memory Optimization")
    print(f"{'='*70}")
    print(f"Config: {config_path}")
    print(f"Tracked params: {tracked_params_path}")
    
    # Verify files exist
    assert config_path.exists(), f"Config not found: {config_path}"
    assert tracked_params_path.exists(), f"Tracked params not found: {tracked_params_path}"
    
    # Load config using safe YAML loader (allows constrained Python objects)
    with open(config_path, "r") as f:
        cfg_dict = yaml.load(f, Loader=_SafeTrackingConfigLoader)
    
    print(f"✓ Loaded config with Python object tags (safe)")
    
    # Now we have cfg_dict as a dict with Path objects.
    print(f"\nCurrent config:")
    print(f"  batch_size: {cfg_dict.get('batch_size', 'N/A')}")
    print(f"  async_func: {cfg_dict.get('async_func', 'N/A')}")
    print(f"  begin_timestep: {cfg_dict.get('begin_timestep', 'N/A')}")
    
    # Update for resumption
    cfg_dict['begin_timestep'] = 8556  # Skip sequential tracking
    cfg_dict['batch_size'] = 4         # Reduce batch size to avoid OOM (was 7)
    cfg_dict['async_func'] = False     # Disable async to avoid stacking tensors
    
    print(f"\nModified config for resumption:")
    print(f"  batch_size: 4 (was 7, targets lower peak VRAM)")
    print(f"  async_func: False (avoids stacking async callbacks)")
    print(f"  begin_timestep: 8556 (skips sequential, starts refinement)")
    
    # Set the path to pre-tracked flame params
    if isinstance(cfg_dict['model'], dict):
        cfg_dict['model']['flame_params_path'] = str(tracked_params_path)
    else:
        cfg_dict['model'].flame_params_path = tracked_params_path
    
    # Ensure we use the correct device
    cfg_dict['device'] = "cuda"
    
    print(f"\nInitializing GlobalTracker...")
    print(f"  Device: cuda")
    print(f"  Flame params: {tracked_params_path}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠ WARNING: CUDA not available, will use CPU")
    
    # Initialize tracker with modified config
    tracker = GlobalTracker(cfg_dict)
    
    print(f"\n{'='*70}")
    print(f"Starting Global Refinement Optimization (Epoch 1, continued)...")
    print(f"{'='*70}\n")
    
    # Run optimization
    tracker.optimize()
    
    print(f"\n{'='*70}")
    print(f"Global Refinement completed successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    resume_with_reduced_batch()
