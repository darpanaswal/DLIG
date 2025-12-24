import re

def resolve_layer_path(model, layer_path: str):
    """
    Resolves a layer path like 'model.layers[0].mlp' to the actual submodule.
    """
    parts = layer_path.split('.')
    curr = model

    for part in parts:
        # Match something like layers[0]
        match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$", part)
        if match:
            attr_name, idx = match.groups()
            if not hasattr(curr, attr_name):
                raise AttributeError(f"Model has no attribute '{attr_name}' in path '{layer_path}'")
            curr = getattr(curr, attr_name)[int(idx)]
        elif hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            raise AttributeError(f"Model has no attribute '{part}' in path '{layer_path}'")

    return curr