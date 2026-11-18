from __future__ import annotations
from itertools import product

def generate_ablation_settings():
    settings = []
    for use_transformer, use_positional_encoding, use_spatial_attention in product([False, True], repeat=3):
        if not use_transformer and (use_positional_encoding or use_spatial_attention):
            continue
        settings.append({
            "use_transformer": use_transformer,
            "use_positional_encoding": use_positional_encoding,
            "use_spatial_attention": use_spatial_attention,
        })
    return settings
