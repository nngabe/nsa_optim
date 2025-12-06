#!/bin/bash
set -e

echo "=========================================="
echo "NSA + Optimizer Smoke Tests"
echo "=========================================="
echo ""

# Activate environment
source venv/bin/activate

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

# Test 1: AdamW with bfloat16 (baseline)
run_test "AdamW + BFloat16 (Baseline)" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type dense \
      --optimizer_type adamw \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype bfloat16 \
      --output_dir ./outputs/smoke_test_adamw" || true

# Test 2: AdamW8bit with bfloat16
run_test "AdamW8bit + BFloat16" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type dense \
      --optimizer_type adamw_8bit \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype bfloat16 \
      --output_dir ./outputs/smoke_test_adamw8bit" || true

# Test 3: AdamW with NSA
run_test "AdamW + Float16" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type native_sparse_attention \
      --optimizer_type adamw \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype float16 \
      --output_dir ./outputs/smoke_test_nsa" || true

# Test 5: Gradient checkpointing
run_test "AdamW + Gradient Checkpointing" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type dense \
      --optimizer_type adamw \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype bfloat16 \
      --gradient_checkpointing \
      --output_dir ./outputs/smoke_test_gradckpt" || true

# Test 6: SOAP optimizer (if available)
echo ""
echo "Testing SOAP optimizer (may skip if not installed)..."
run_test "SOAP + BFloat16" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type dense \
      --optimizer_type soap \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype bfloat16 \
      --output_dir ./outputs/smoke_test_soap" || true

# Test 7: SOAP lowbit 
echo ""
echo "Testing SOAP optimizer (may skip if not installed)..."
run_test "SOAP + BFloat16" \
    "$TRAIN_CMD train.py \
      --model_size 0.6B \
      --attention_type dense \
      --optimizer_type soap \
      --context_length 32768 \
      --num_train_steps 2 \
      --log_interval 1 \
      --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --dtype bfloat16 \
      --output_dir ./outputs/smoke_test_soap_lowbit" || true

# Print final summary
print_summary

# Cleanup test outputs
echo ""
echo "Cleaning up test outputs..."
rm -rf ./outputs/smoke_test_*
echo "Cleanup complete!"

exit 0
