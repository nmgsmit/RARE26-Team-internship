# KNN Submission Error Fix - Applied to main and clean-baseline

## Issue Summary
Submissions were failing with:
```
TypeError: VisionTransformer.__init__() got an unexpected keyword argument 'head_dropout'
```

This occurred when submitting models with KNN heads (k=25, k=50, etc.) and potentially other checkpoint configurations.

## Root Cause
When a model checkpoint is saved, its `model_config` dictionary contains all training parameters:
- Head-specific params: `head_dropout`, `head_hidden_dim`, `mlp_hidden_dim`, etc.
- Metadata: `backbone_preset`, `num_folds`, `fold_index`
- Other inference-specific params: `classifier_input_dim`

During submission inference, `resolve_model_kwargs_from_checkpoint()` extracts these parameters. However, **not all saved parameters are explicit parameters in Model.__init__()**.

Parameters that aren't explicitly defined fall through to `**kwargs`, which then get passed directly to `timm.create_model()` as `backbone_kwargs`. The Vision Transformer backbone rejects these unknown parameters, causing the submission to fail.

## Branches Affected
- **main**: Full-featured model with MLPs, dropouts, and flexible head configurations
- **clean-baseline**: Simpler KNN/SVM-only model

Both branches had the vulnerability but with slightly different parameter sets.

## Solution Implemented

### Two-Layer Defense

#### Layer 1: Aggressive Filtering at Submission Entry Point
Added `filter_model_kwargs_for_init()` function using Python's `inspect.signature()`:

```python
def filter_model_kwargs_for_init(model_kwargs):
    """Filter out kwargs that aren't valid Model.__init__ parameters."""
    valid_keys = {
        key for key in inspect.signature(Model.__init__).parameters
        if key not in {"self", "kwargs"}
    }
    return {key: value for key, value in dict(model_kwargs).items() if key in valid_keys}
```

This is applied **immediately** after checkpoint resolution, removing any parameters not in the Model's signature.

#### Layer 2: Defensive Filtering in Model Class
Added explicit filtering in `Model.__init__()` to prevent known model-specific parameters from reaching the backbone:

**main branch:**
```python
model_specific_params = {
    "classifier_input_dim",   # Used during head inference, not a Model param
    "backbone_preset",        # Metadata
    "num_folds",              # Metadata
    "fold_index",             # Metadata
}
filtered_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}
```

**clean-baseline branch:**
```python
model_specific_params = {
    "classifier_input_dim",   # Used during head inference, not a Model param
    "head_hidden_dim",        # Not used in this branch's simpler head
    "head_dropout",           # Not used in this branch's simpler head
    "mlp_hidden_layers",      # Not used in this branch
    "mlp_hidden_dim",         # Not used in this branch
    "mlp_dropout",            # Not used in this branch
    "backbone_preset",        # Metadata
    "num_folds",              # Metadata
    "fold_index",             # Metadata
}
filtered_kwargs = {k: v for k, v in kwargs.items() if k not in model_specific_params}
```

## Changes Made

### main branch (commit 603dafb)
- **model.py**: Added defensive filtering in Model.__init__()
- **Submission_files/predict.py**: Added filter_model_kwargs_for_init() and applied it after checkpoint resolution
- **KNN_FIX_SUMMARY.md**: Root cause documentation

### clean-baseline branch (commit 5b95891)
- **model.py**: Added defensive filtering in Model.__init__() (with clean-baseline-specific params)
- **Submission_files/predict.py**: Added filter_model_kwargs_for_init() and applied it in _build_single_model()

## Why This Works

1. **Single Model Checkpoints**: Parameters are filtered before Model.__init__(), so even if a checkpoint has unexpected parameters, they won't reach the backbone.

2. **Ensemble Checkpoints**: Each fold's parameters go through the same filtering in _build_single_model(), so ensemble submissions with varied configurations all work.

3. **Backward Compatible**: The fix doesn't break existing checkpoints. It just removes extra parameters that weren't supposed to be there anyway.

4. **Future-Proof**: If model architecture changes add new parameters later, they'll be automatically included in the signature check without code changes.

## Testing Recommendations

1. **Test with existing checkpoints**: KNN k=5, k=25, k=50; SVM; linear heads
2. **Test with ensemble bundles**: Multi-fold checkpoints with varying configurations
3. **Test with edge cases**: Old checkpoints with deprecated parameters
4. **Verify predictions are identical**: The fix should not affect prediction values

## Files Modified Across Branches
```
main/
  model.py
  Submission_files/predict.py
  KNN_FIX_SUMMARY.md

clean-baseline/
  model.py
  Submission_files/predict.py
```

## Commits
- **main**: `603dafb` - Fix KNN submission error: filter model-specific kwargs
- **clean-baseline**: `5b95891` - Fix KNN submission error on clean-baseline: filter model-specific kwargs
