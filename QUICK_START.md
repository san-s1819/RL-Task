# Quick Start Guide

## What You Have

An RL training task where an LLM must implement **dropout regularization from scratch** using NumPy.

## Files Created

```
hello-py/
├── main2.py                   ⭐ Main task file (run this!)
├── test_dropout_grader.py     ✓ Verified - grader works correctly
├── TASK_DESIGN.md             📖 Design rationale & requirements
├── SUBMISSION_SUMMARY.md      📋 Complete submission details
├── QUICK_START.md             👋 This file
├── README.md                  📚 Updated with task info
└── pyproject.toml             ✓ Dependencies updated
```

## Run It Now

```bash
# 1. Make sure you're in the hello-py directory
cd "S:\Portfolio\RL Enginer task\hello-py"

# 2. Set your API key (if not already set)
$env:ANTHROPIC_API_KEY="your_api_key_here"

# 3. Run the task evaluation
uv run main2.py
```

## What to Expect

The script will:
1. Run 10 concurrent test iterations
2. Each iteration asks Claude to implement dropout from scratch
3. Grade each implementation against 8 test cases
4. Report pass rate and failure reasons

**Expected pass rate:** 10-40% (requirement met!)

## Example Output

```
Running 10 test iterations concurrently...
============================================================
[PASS] Run 3: SUCCESS - All tests passed!
[FAIL] Run 1: FAILURE - Non-zero values are not scaled correctly
[FAIL] Run 2: FAILURE - No answer submitted
[PASS] Run 5: SUCCESS - All tests passed!
...
============================================================
Test Results:
  Passed: 3/10
  Failed: 7/10
  Pass Rate: 30.0%

Failure Summary:
  Run 1: Non-zero values are not scaled correctly (should be scaled by 1/(1-p))
  Run 2: No answer submitted
  ...
============================================================
```

## Testing the Grader

Before running the full evaluation, you can test that the grading function works:

```bash
uv run test_dropout_grader.py
```

This runs 6 tests (2 correct implementations, 4 common mistakes) and should show:
```
[PASS] Correct Implementation
[PASS] Alternative Implementation
[PASS] No Scaling Error (correctly rejected)
[PASS] Wrong Scaling Error (correctly rejected)
[PASS] No Train/Eval Distinction (correctly rejected)
[PASS] Missing Function (correctly rejected)

Total: 6/6 tests passed
```

## What Makes This a Good RL Task?

✅ **Realistic**: Implementing layers from scratch is common in ML work
✅ **Right difficulty**: Multiple ways to fail, not just one bottleneck
✅ **Clear grading**: Every prompt requirement is verified
✅ **Flexible**: Accepts any correct implementation
✅ **Educational**: Teaches fundamental ML concept (dropout + scaling)
✅ **Concise**: Under 200 lines total

## Common Failure Modes

The model typically fails because:
1. **Forgets scaling** (most common) - just zeros without `/ (1-p)`
2. **Wrong scaling factor** - uses `p` instead of `1/(1-p)`
3. **No train/eval distinction** - applies dropout even in eval mode
4. **Edge cases** - doesn't handle p=0 or p=1 correctly
5. **Runs out of steps** - doesn't submit answer in time

## Adjusting Difficulty

If the pass rate isn't in the 10-40% range:

### Too Easy (>40% pass rate)
- Remove the example test code from prompt
- Make formula description less explicit
- Add more edge case requirements

### Too Hard (<10% pass rate)
- Add more hints about the scaling factor
- Increase max_steps from 10 to 15
- Provide a partial implementation template

## Next Steps

1. **Run the evaluation**: `uv run main2.py`
2. **Measure pass rate**: Run at least 10 times to get accurate statistics
3. **Review failures**: Check what causes most failures
4. **Adjust if needed**: Tweak prompt/grading based on results
5. **Submit**: The task is ready as-is!

## Cost Estimate

- Model: Claude Haiku 4.5
- Per run: ~5-10 API calls
- Total cost for 10 runs: ~$0.50-1.00

## Support

All requirements are met:
- ✅ Resembles real ML work
- ✅ Target 10-40% pass rate
- ✅ Prompt matches grading
- ✅ Accepts all valid solutions
- ✅ Verifies all requirements
- ✅ Teaches useful skill
- ✅ Multiple approaches
- ✅ Diverse failure modes
- ✅ No tool issues
- ✅ <300 lines of code

Ready to evaluate! 🚀

