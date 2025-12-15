#!/bin/bash
set -e

echo "=========================================="
echo "NSA + Optimizer Smoke Tests"
echo "=========================================="
echo ""

# Activate environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Detect number of GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected $NUM_GPUS CUDA GPU(s)"

# Set up torchrun command prefix
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --standalone"
    echo "Using torchrun with $NUM_GPUS GPUs"
elif [ "$NUM_GPUS" -eq 1 ]; then
    TRAIN_CMD="python"
    echo "Using single GPU"
else
    TRAIN_CMD="python"
    echo "No GPU detected, using CPU"
fi

# Set wandb to offline mode for smoke tests
export WANDB_MODE=offline
echo "WANDB_MODE set to offline for smoke tests"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    shift
    local test_cmd="$@"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
    echo "=========================================="
    echo "Test $TOTAL_TESTS: $test_name"
    echo "=========================================="

    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to print summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo "Total:  $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        return 1
    fi
}

echo "Step 1: Unit Tests"
echo "=========================================="
run_test "PyTest Unit Tests" "python -m pytest -v --tb=short -m 'not slow and not gpu'" || true

echo ""
echo "Step 2: Training Smoke Tests"
echo "=========================================="

# Common training parameters for all tests
MODEL_SIZE="100M"  # Small model for fast smoke tests
CONTEXT_LENGTH=512
NUM_TRAIN_STEPS=2
LOG_INTERVAL=1
BATCH_SIZE=1
GRAD_ACCUM_STEPS=2
DTYPE="bfloat16"

###############################################################################
# AdamW8bit Tests
###############################################################################

echo ""
echo "=========================================="
echo "AdamW8bit Optimizer Tests"
echo "=========================================="

# Test 1: AdamW8bit with 'A'/dense (Attention with dense attention)
run_test "AdamW8bit + A/dense" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --attn_type dense \
      --optimizer_type adamw8bit \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw8bit_a_dense" || true

# Test 2: AdamW8bit with 'A'/sparse (Attention with NSA sparse attention)
run_test "AdamW8bit + A/sparse" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --attn_type native_sparse_attention \
      --nsa_block_size 64 \
      --nsa_window_size 64 \
      --nsa_num_selected_blocks 4 \
      --optimizer_type adamw8bit \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw8bit_a_sparse" || true

# Test 3: AdamW8bit with 'D' (DeltaNet blocks)
run_test "AdamW8bit + D" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --block_pattern D \
      --block_repeats 2 \
      --optimizer_type adamw8bit \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw8bit_d" || true

# Test 4: AdamW8bit with 'M' (Mamba blocks)
run_test "AdamW8bit + M" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --block_pattern M \
      --block_repeats 2 \
      --optimizer_type adamw8bit \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw8bit_m" || true

# Test 5: AdamW8bit with 'MDMA'/sparse (Mixed blocks: Mamba-DeltaNet-Mamba-Attention)
run_test "AdamW8bit + MDMA/sparse" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --block_pattern MDMA \
      --block_repeats 2 \
      --optimizer_type adamw8bit \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw8bit_mdma_sparse" || true

###############################################################################
# SOAP8bit Tests
###############################################################################

echo ""
echo "=========================================="
echo "SOAP8bit Optimizer Tests"
echo "=========================================="

# Test 6: SOAP8bit with 'MDMA'/sparse (Mixed blocks: Mamba-DeltaNet-Mamba-Attention)
run_test "SOAP8bit + MDMA/sparse" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --block_pattern MDMA \
      --block_repeats 2 \
      --optimizer_type soap8bit \
      --precondition_frequency 5 \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_soap8bit_mdma_sparse" || true

###############################################################################
# SOAP4bit Tests
###############################################################################

echo ""
echo "=========================================="
echo "SOAP4bit Optimizer Tests"
echo "=========================================="

# Test 7: SOAP4bit with 'MDMA'/sparse (Mixed blocks: Mamba-DeltaNet-Mamba-Attention)
run_test "SOAP4bit + MDMA/sparse" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --block_pattern MDMA \
      --block_repeats 2 \
      --optimizer_type soap4bit \
      --precondition_frequency 5 \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_soap4bit_mdma_sparse" || true

###############################################################################
# Baseline Tests (for reference)
###############################################################################

echo ""
echo "=========================================="
echo "Baseline Tests (AdamW)"
echo "=========================================="

# Test 8: AdamW baseline with dense attention (for comparison)
run_test "AdamW (baseline) + dense" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --attn_type dense \
      --optimizer_type adamw \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --output_dir ./outputs/smoke_test_adamw_baseline" || true

# Test 9: AdamW with gradient checkpointing (memory efficiency test)
run_test "AdamW + Gradient Checkpointing" \
    "$TRAIN_CMD train.py \
      --model_size $MODEL_SIZE \
      --attn_type dense \
      --optimizer_type adamw \
      --context_length $CONTEXT_LENGTH \
      --num_train_steps $NUM_TRAIN_STEPS \
      --log_interval $LOG_INTERVAL \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --dtype $DTYPE \
      --gradient_checkpointing \
      --output_dir ./outputs/smoke_test_adamw_gradckpt" || true

# Print final summary
print_summary

# Cleanup test outputs
echo ""
echo "Cleaning up test outputs..."
rm -rf ./outputs/smoke_test_*
echo "Cleanup complete!"

exit 0
