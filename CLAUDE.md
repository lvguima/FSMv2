# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FSM (Forecasting Stream Models)** is a research codebase for time series forecasting with a focus on **online learning** and **adaptive prediction**. The centerpiece is **Titan-Stream (M-Stream V9)**, a novel architecture implementing surprise-driven memory updates for non-stationary time series.

### Core Innovation: Memory As Context (MAC)

Titan-Stream uses a three-stream parallel architecture:
1. **Neural Long-term Memory** (Fast System): Dynamic D×D matrix capturing concept drift via gradient momentum
2. **Persistent Memory** (Slow System): Static learnable KV pairs storing fixed priors (periodicity, constraints)
3. **Core Forecaster** (Fusion Engine): Transformer encoder fusing all three streams

Key mechanism: **Surprise-driven updates** using proxy loss gradients with momentum: `M_t = (1-α_t)·M_{t-1} - η·S_t`

## Running Experiments

### Training (Offline)

```bash
# Basic training
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TitanStream \
  --data ETTh1 \
  --seq_len 96 --pred_len 24 \
  --d_model 256 --n_heads 4 \
  --train_epochs 10

# With high-order gradients (meta-learning)
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TitanStream \
  --data ETTm1_Online \
  --seq_len 512 --pred_len 96 \
  --chunk_len 128 \
  --use_high_order \
  --clip_grad 1.0 \
  --lambda_proxy 0.1
```

### Online Testing

```bash
python run.py \
  --task_name online_forecast \
  --is_training 0 \
  --model TitanStream \
  --data ETTm1_Online \
  --online_strategy proxy \
  --use_surprise_gate \
  --use_delayed_feedback \
  --delayed_batch_size 8 \
  --save_online_results
```

### Using Shell Scripts

```bash
# Complete training + online evaluation pipeline
bash scripts/online_forecast/titan_ettm1_online.sh
```

## Architecture Deep Dive

### Critical Design Patterns

#### 1. Training-Test Consistency (FIXED)

**Problem**: Training used static memory (`use_state=False`), but testing used dynamic memory, causing train-test mismatch.

**Solution**: Memory now continues across time chunks during training via `external_memory` parameter:

```python
# In exp_long_term_forecasting.py
for t, (t_start, t_end) in enumerate(time_slices):
    outputs = self.model(
        x_slice,
        use_state=False,  # Don't use buffer (avoids inplace ops)
        update_state=use_high_order,
        differentiable_update=use_high_order,
        external_memory=temp_memory,  # Pass memory externally
        external_momentum=temp_momentum
    )
    # [CRITICAL] Extract updated memory from stats and DETACH to prevent graph accumulation
    if stats.get('updated_memory') is not None:
        temp_memory = stats['updated_memory'].detach()  # Must detach!
    if stats.get('updated_momentum') is not None:
        temp_momentum = stats['updated_momentum'].detach()  # Must detach!
```

**Why `.detach()` is critical**: Without detaching, computation graphs accumulate across time slices, causing:
- Exponential slowdown (0.26s/iter → 3.7s/iter over epochs)
- Memory leak and eventual OOM
- Backward pass traversing entire history instead of single slice

#### 2. Differentiable Memory Updates

Two update modes:
- **Training (differentiable)**: `_update_memory_differentiable()` preserves gradients for meta-learning
- **Inference (in-place)**: `_update_online_memory()` uses `torch.no_grad()` for efficiency

Enable with `--use_high_order` flag.

#### 3. Memory Reset Timing

**Critical**: Reset memory at **batch level**, not epoch level:

```python
for epoch in range(epochs):
    for batch in train_loader:
        if is_titan_stream:
            self.model.reset_memory()  # Reset per batch
        # ... training loop
```

### H_mem Computation Semantics

The design uses `H_mem = M · Q` (memory as linear transformation matrix), not standard Linear Attention's `H = Q · M`. This allows memory to directly modulate input features, better capturing non-stationary dynamics.

See `FSMdesign.md` lines 35-39 for detailed explanation.

### Forecast Head Architecture (REDESIGNED)

**Problem**: Original design used mean pooling which discarded all temporal information:
```python
pooled = core_out.mean(dim=0)  # [B, D] - loses time structure
forecast = Linear(D, pred_len * c_out)
```

**Solution**: Attention Pooling + 2-layer MLP preserves temporal information:
```python
# Learnable query vector attends to all tokens
query = forecast_query.expand(-1, B, -1)  # [1, B, D]
pooled, _ = forecast_attn(query, core_out, core_out)  # [1, B, D]
pooled = pooled.squeeze(0)  # [B, D]

# 2-layer MLP for prediction
hidden = Linear(D, 2D) + GELU + Dropout
forecast = Linear(2D, pred_len * c_out)
```

**Why not Flatten + MLP?**: With seq_len=512, flattening creates 393K input dimensions → 201M parameters (99% of model). Attention pooling keeps parameters reasonable (740K, 35% of model) while learning which time steps matter.

## Key Parameters

### Titan-Stream Specific

- `--n_persistent 32`: Number of persistent memory tokens
- `--beta_momentum 0.9`: Momentum coefficient for memory updates
- `--lr_memory 0.01`: Learning rate for online memory updates
- `--gate_hidden 128`: Hidden size of forgetting gate MLP
- `--forecast_hidden_dim 0`: Forecast head hidden dim (0=auto: 2*d_model)
- `--chunk_len 128`: Time-dimension chunk length (0=disable)
- `--use_high_order`: Enable differentiable memory updates (meta-learning)
- `--clip_grad 1.0`: Gradient clipping max norm
- `--lambda_proxy 0.1`: Weight for proxy reconstruction loss
- `--persistent_lr_scale 0.0`: LR scale for persistent memory fine-tuning (0=freeze)

### Online Learning

- `--online_strategy proxy`: Strategy (proxy/naive_ft/static)
- `--use_surprise_gate`: Enable adaptive update gating
- `--surprise_thresh 3.0`: Surprise threshold (std)
- `--warmup_steps 50`: Warmup before gating
- `--use_delayed_feedback`: Enable delayed supervised updates
- `--delayed_batch_size 8`: Batch size for delayed updates
- `--delayed_horizon 96`: Delay horizon (usually pred_len)

## Common Issues & Solutions

### 1. Inplace Operation Error

**Error**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

**Cause**: Using `--use_high_order` with buffer inplace operations.

**Solution**: Already fixed. Memory is passed via `external_memory` parameter instead of modifying buffers.

### 2. Negative R² Scores

**Cause**: Training-test inconsistency (memory not continuing during training).

**Solution**: Already fixed. Use `--use_high_order` to enable proper memory continuation.

### 3. Gradient Explosion

**Cause**: Gradient norm threshold too high (was 1e3).

**Solution**: Already fixed. Threshold lowered to 10.0 in `_update_memory_differentiable()`.

### 4. OOM with High-Order Gradients

**Solutions**:
- Reduce `--batch_size` (32 → 16)
- Reduce `--chunk_len` (128 → 64)
- Reduce `--d_model` (256 → 128)

## File Structure

### Core Components

- `models/titan_stream.py`: Titan-Stream model implementation
- `models/MStream.py`: M-Stream baseline
- `exp/exp_long_term_forecasting.py`: Training pipeline with chunk-based memory continuation
- `exp/exp_online_forecast.py`: Online testing with delayed feedback
- `utils/online_utils.py`: Online metrics, surprise gate, delayed feedback buffer
- `run.py`: Main entry point with CLI arguments

### Experiment Classes Hierarchy

```
Exp_Basic (exp/exp_basic.py)
├── Exp_Long_Term_Forecast
│   └── Exp_Online_Forecast (inherits train, overrides test)
├── Exp_Short_Term_Forecast
├── Exp_Imputation
├── Exp_Anomaly_Detection
└── Exp_Classification
```

### Data Loaders

- `data_provider/data_loader.py`: Dataset classes
  - `Dataset_ETT_hour/minute`: Standard ETT splits (60/10/30)
  - `Dataset_ETT_minute_Online`: Online split (4/1/15 months)
  - `Dataset_Custom_Online`: Cold-start split (20/5/75)

### 50+ Model Zoo

Located in `models/`:
- Transformer-based: Autoformer, Informer, FEDformer, PatchTST, iTransformer
- Attention-free: DLinear, TimesNet, SegRNN
- State-space: Mamba, Koopa
- Online: MStream, TitanStream

## Recent Fixes (2025-12-11)

### P0 Core Fixes (Completed)

1. **Memory reset timing**: Moved from epoch-level to batch-level
2. **Differentiable updates**: Added `_update_memory_differentiable()` for meta-learning
3. **Training memory continuation**: Memory now continues across chunks via external parameters
4. **Inplace operation fix**: Memory passed as parameters instead of buffer modification
5. **🔴 Computation graph accumulation fix**: Added `.detach()` to `temp_memory` and `temp_momentum` to prevent graph accumulation across time slices, which caused exponential slowdown (Epoch 1: 0.26s/iter → Epoch 6: 3.7s/iter)

### P1 Performance Improvements (Completed)

6. **Gradient normalization**: Threshold lowered from 1e3 to 10.0
7. **Persistent memory tuning**: Added `freeze_persistent()`/`unfreeze_persistent()` methods
8. **H_mem semantics**: Clarified in design docs
9. **Forecast head redesign**: Replaced mean pooling with Attention Pooling + 2-layer MLP to preserve temporal information while avoiding parameter explosion

### Experiment Configuration Fixes (Completed)

10. **Checkpoint path consistency**: Unified `data` parameter between offline training and online testing (both use same dataset name)
11. **Training hyperparameters**: Increased patience (3→7), adjusted learning rate (1e-3→5e-4), switched to cosine scheduler
12. **Backbone parameter grouping**: Fixed parameter classification for TitanStream (added `core`, `input_proj`, `q/k/v_proj` to backbone keywords)

See `TITAN_STREAM_FIX_PLAN.md` for detailed fix documentation.

## Important Design Documents

- `FSMdesign.md`: Complete mathematical specification of Titan-Stream architecture
- `TITAN_STREAM_REFACTOR_PLAN.md`: Implementation roadmap and progress tracking
- `TITAN_STREAM_FIX_PLAN.md`: Detailed fix plan for training-test consistency issues

## Development Notes

### When Modifying Training Loop

- Always reset memory at batch level, not epoch level
- Use `external_memory`/`external_momentum` parameters to pass memory state
- Never use inplace operations on buffers when `use_high_order=True`
- **CRITICAL**: Always `.detach()` when extracting updated memory from `stats` dict to prevent graph accumulation
- Clean up intermediate variables after backward pass to prevent memory leaks:
  ```python
  del loss, loss_chunks, loss_pred_vals, loss_proxy_vals, loss_orth_vals
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
  ```

### When Adding New Models

- Inherit from `Exp_Long_Term_Forecast` for standard forecasting
- Override `test()` method for custom evaluation logic
- Return tuple `(pred, proxy_loss, stats)` from forward pass for Titan-Stream compatibility

### Numerical Stability

- Gradient clipping enabled via `--clip_grad`
- NaN safety checks in `titan_stream.py` forward pass
- Input normalization via `--use_norm 1`
- Momentum gradient normalization with max_grad_norm=10.0

## Expected Performance

After fixes:
- **R² metric**: Should be positive (target > 0.5), was negative before fixes
- **Training stability**: Smooth loss curves, no gradient explosion
- **Online adaptation**: Model correctly utilizes delayed feedback for updates

## Checkpoints & Results

- Training checkpoints: `./checkpoints/long_term_forecast_<model_id>_*/`
- Online results: `./results/online_forecast_<model_id>_*/`
- Test outputs: `./test_results/`
- Metrics log: `result_long_term_forecast.txt`
