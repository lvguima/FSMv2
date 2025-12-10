# M-Stream v4.0: Attention Memory 消融实验脚本
# 四个实验依次运行：标准配置、大容量记忆、激进监督、高速适应

# 切换到项目根目录（脚本在 scripts/online_forecast/ 目录下）
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
Set-Location $ProjectRoot

Write-Host "=========================================="
Write-Host "M-Stream v4.0 - Attention Memory 消融实验"
Write-Host "=========================================="
Write-Host "项目根目录: $ProjectRoot"
Write-Host ""

# ========== 实验 A: v4.0 标准配置 (Baseline) ==========
Write-Host "[实验 A/4] v4.0 标准配置: Rank=32, Weight=0.7, LR=0.001"
Write-Host "------------------------------------------"
python online_runner.py --model MStream --data ETTh1 --mode train_and_test --memory_type attention --memory_rank 32 --d_model 128 --train_epochs 20 --use_delayed_feedback 1 --delayed_supervised_weight 0.7 --lr_ttt 0.001 --des "v4_std_rank32_w0.7" --itr 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "实验 A 执行失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "实验 A 完成！" -ForegroundColor Green
Write-Host ""

# ========== 实验 B: 大容量记忆 (Large Memory) ==========
Write-Host "[实验 B/4] 大容量记忆: Rank=64 (增加一倍)"
Write-Host "------------------------------------------"
python online_runner.py --model MStream --data ETTh1 --mode train_and_test --memory_type attention --memory_rank 64 --d_model 128 --train_epochs 20 --use_delayed_feedback 1 --delayed_supervised_weight 0.7 --lr_ttt 0.001 --des "v4_large_rank64" --itr 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "实验 B 执行失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "实验 B 完成！" -ForegroundColor Green
Write-Host ""

# ========== 实验 C: 激进监督更新 (Aggressive Supervision) ==========
Write-Host "[实验 C/4] 激进监督更新: Weight=0.9 (高度依赖真实标签)"
Write-Host "------------------------------------------"
python online_runner.py --model MStream --data ETTh1 --mode train_and_test --memory_type attention --memory_rank 32 --d_model 128 --train_epochs 20 --use_delayed_feedback 1 --delayed_supervised_weight 0.9 --lr_ttt 0.001 --des "v4_agg_w0.9" --itr 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "实验 C 执行失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "实验 C 完成！" -ForegroundColor Green
Write-Host ""

# ========== 实验 D: 高速适应 (High Learning Rate) ==========
Write-Host "[实验 D/4] 高速适应: LR=0.005 (5倍学习率)"
Write-Host "------------------------------------------"
python online_runner.py --model MStream --data ETTh1 --mode train_and_test --memory_type attention --memory_rank 32 --d_model 128 --train_epochs 20 --use_delayed_feedback 1 --delayed_supervised_weight 0.7 --lr_ttt 0.005 --des "v4_fast_lr0.005" --itr 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "实验 D 执行失败，退出码: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "实验 D 完成！" -ForegroundColor Green
Write-Host ""

# ========== 实验总结 ==========
Write-Host "=========================================="
Write-Host "所有消融实验完成！" -ForegroundColor Green
Write-Host "=========================================="
Write-Host ""
Write-Host "实验配置总结:"
Write-Host "  A. 标准配置: Rank=32, Weight=0.7, LR=0.001"
Write-Host "  B. 大容量记忆: Rank=64, Weight=0.7, LR=0.001"
Write-Host "  C. 激进监督: Rank=32, Weight=0.9, LR=0.001"
Write-Host "  D. 高速适应: Rank=32, Weight=0.7, LR=0.005"
Write-Host ""

