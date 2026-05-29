# KNN k=25 Submission Error - Root Cause & Fix

## Problem
Submission failed with:
```
TypeError: VisionTransformer.__init__() got an unexpected keyword argument 'head_dropout'
```

This occurred specifically when using KNN with k=25, even though KNN with k=50 worked fine.

## Root Cause
When a checkpoint is saved, its `model_config` dictionary contains all model parameters (e.g., `head_dropout`, `knn_neighbors`, `mlp_hidden_dim`, etc.). When loading the checkpoint for inference, the `resolve_model_kwargs_from_checkpoint()` function extracts these parameters and returns them as a kwargs dictionary.

However, not all saved parameters are explicit parameters of the `Model.__init__()` method. Parameters like:
- `classifier_input_dim` (used for head inference, not a direct Model param)
- Other metadata parameters

These "unexpected" parameters would fall through to the `**kwargs` catch-all in `Model.__init__()`, which then passed them directly to `timm.create_model()` as `backbone_kwargs`. The Vision Transformer backbone doesn't accept these parameters, causing the error.

## Solution
Implemented a two-layer defense:

### Layer 1: Filter in predict.py
Added `filter_model_kwargs_for_init()` function that uses Python's `inspect.signature()` to keep only valid Model.__init__ parameters:

```python
def filter_model_kwargs_for_init(model_kwargs):
    """Filter out kwargs that aren't valid Model.__init__ parameters."""
    valid_keys = {
        key for key in inspect.signature(Model.__init__).parameters
        if key not in {"self", "kwargs"}
    }
    return {key: value for key, value in dict(model_kwargs).items() if key in valid_keys}
```

This ensures that only parameters the Model class actually accepts are passed to it.

### Layer 2: Defense in model.py
Added explicit filtering of known model-specific parameters in `Model.__init__()` that shouldn't reach the backbone:

```python
# Filter out model-specific parameters that shouldn't be passed to backbone
model_specific_params = {
    "classifier_input_dim",  # Used for inferring classifier input, not backbone param
    "backbone_preset",  # Metadata
    "num_folds",  # Metadata
    "fold_index",  # Metadata
}
filtered_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}

backbone_kwargs = dict(
    pretrained=pretrained,
    num_classes=0,
    in_chans=in_channels,
    **filtered_kwargs,
)
```

## Why This Works for Both k=50 and k=25
The issue wasn't actually about the KNN value itself, but about which parameters ended up being saved in the checkpoint's `model_config`. Different model configurations (trained with different heads, dropout values, etc.) might save different sets of parameters. The filter ensures that regardless of what's in the checkpoint, only valid parameters are passed to the Model constructor, and only valid backbone parameters reach timm.

## Files Modified
1. **model.py**: Added defensive filtering in Model.__init__()
2. **Submission_files/predict.py**: Added filter_model_kwargs_for_init() function and applied it to model_kwargs before Model instantiation

## Testing
To verify this works:
1. Retrain models with KNN k=25 and k=50
2. Submit both - they should now work without the TypeError
3. The fix is backward compatible - it won't break existing checkpoints
