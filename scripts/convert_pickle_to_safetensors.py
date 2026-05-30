#!/usr/bin/env python3
"""
Convert FLAME model .pkl files (loaded via pickle.load) to safetensors format.

Usage:
    python scripts/convert_pickle_to_safetensors.py <input.pkl> <output.safetensors>

The script loads the pickle file, converts all numpy/scipy arrays to torch tensors,
and saves them using safetensors.torch.save_file for secure deserialization.
"""

import sys
import pickle
import numpy as np
import torch
from safetensors.torch import save_file


def convert_array(arr):
    """Convert numpy arrays and scipy sparse matrices to torch tensors."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    if hasattr(arr, "todense"):
        return torch.from_numpy(np.array(arr.todense()))
    if isinstance(arr, (list, tuple)):
        return torch.tensor(arr)
    raise TypeError(f"Unsupported type in pickle: {type(arr)}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output.safetensors>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading pickle file: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    if not isinstance(data, dict):
        print(f"Error: Expected a dict in pickle, got {type(data)}")
        sys.exit(1)

    print(f"Converting {len(data)} keys to tensors...")
    tensor_data = {}
    for key, value in data.items():
        tensor_data[key] = convert_array(value)
        print(f"  {key}: {tensor_data[key].shape if hasattr(tensor_data[key], 'shape') else 'scalar'}")

    print(f"Saving to safetensors: {output_path}")
    save_file(tensor_data, output_path)
    print("Done.")


if __name__ == "__main__":
    main()