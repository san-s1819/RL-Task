# RL Task Submission Summary

## Task: Implement Dropout from Scratch

### Quick Overview
- **File**: `main2.py`
- **Task Type**: Implement a fundamental ML technique (dropout regularization)
- **Target Pass Rate**: 10-40%
- **Lines of Code**: ~180 lines total (prompt + grading + infrastructure)

---

## What This Task Teaches

The model learns to implement dropout, a critical regularization technique used in neural networks:

1. **Conceptual Understanding**: How dropout prevents overfitting
2. **Mathematical Rigor**: Proper scaling (1/(1-p)) to maintain expected values
3. **Mode Switching**: Different behavior in training vs evaluation
4. **NumPy Proficiency**: Array masking and random operations

---

## Why This is a Good RL Task

### ✅ Meets All Requirements

1. **Realistic ML Work**: Engineers implement layers from scratch for understanding/customization
2. **Proper Difficulty**: Multiple ways to fail, requires understanding beyond guessing
3. **Clear Grading**: Every prompt requirement is verified
4. **Flexible Acceptance**: Any correct implementation passes (no rigid string matching)
5. **Hard to Guess**: Must understand the formula and edge cases
6. **Diverse Failures**: Models fail for different reasons (scaling, train/eval, edge cases)
7. **No Tool Issues**: Simple tools, NumPy pre-loaded, can test before submitting
8. **Concise**: Under 200 lines, easy to review

### 📊 Expected Pass Rate: 10-40%

**Why models will struggle:**
- Most common mistake: Forgetting to scale non-zero values
- Second most common: Wrong scaling factor (using p instead of 1/(1-p))
- Train/eval mode distinction often missed
- Edge cases (p=0, p=1) can trip up implementations

**Why it's not too hard:**
- Clear problem statement with formula
- Can test iteratively with python_expression tool
- Error messages guide towards correct solution
- Standard NumPy operations

---

## File Structure

```
hello-py/
├── main.py                    # Original example task
├── main2.py                   # ⭐ DROPOUT TASK (submit this)
├── test_dropout_grader.py     # Verify grading function works
├── TASK_DESIGN.md             # Detailed design rationale
├── SUBMISSION_SUMMARY.md      # This file
├── README.md                  # Setup and usage instructions
└── pyproject.toml             # Dependencies
```

---

## How to Run

### 1. Install dependencies
```bash
cd hello-py
uv sync
```

### 2. Set up API key
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Test the grading function (optional)
```bash
uv run test_dropout_grader.py
```

Expected output:
```
✓ Correct Implementation
✓ Alternative Implementation
✓ No Scaling Error (correctly rejected)
✓ Wrong Scaling Error (correctly rejected)
...
Total: 6/6 tests passed
```

### 4. Run the RL task evaluation
```bash
uv run main2.py
```

This runs the task 10 times and reports the pass rate.

---

## Key Implementation Details

### The Prompt
Located in `main2.py`, clearly specifies:
- Function signature: `dropout(x, p, training)`
- Training mode behavior: random zeroing + scaling
- Eval mode behavior: return input unchanged
- Important details: scaling factor is 1/(1-p)
- Provides example test code

### The Grader
Function `grade_dropout_implementation()` in `main2.py` tests:
1. Shape preservation
2. ~p% zeros in training mode (statistical check)
3. Correct scaling of non-zero values
4. No dropout in eval mode
5. Different dropout probabilities
6. Edge cases (p=0, p=1)

### Tools Provided
1. **python_expression**: Execute Python code, test implementation iteratively
2. **submit_answer**: Submit final implementation as string

---

## Example Correct Solution

```python
def dropout(x, p, training):
    if not training:
        return x
    
    if p == 0:
        return x
    
    if p == 1:
        return np.zeros_like(x)
    
    # Generate mask: keep with probability (1-p)
    mask = np.random.rand(*x.shape) > p
    
    # Apply mask and scale
    return x * mask / (1 - p)
```

---

## Validation Checklist

Before submitting, verify:
- [ ] Task runs without errors: `uv run main2.py`
- [ ] Grader tests pass: `uv run test_dropout_grader.py`
- [ ] Pass rate is between 10-40% (run at least 10 times)
- [ ] Prompt is clear and matches grading criteria
- [ ] README.md has setup instructions
- [ ] All code is under 300 lines
- [ ] No external data dependencies

---

## Common Issues & Solutions

**Issue**: Pass rate too high (>40%)
- Make prompt less explicit about formula
- Remove example test code
- Add more edge case requirements

**Issue**: Pass rate too low (<10%)
- Add hints about scaling factor
- Provide clearer example in prompt
- Reduce number of test cases

**Issue**: Models fail on tool usage
- Already fixed: NumPy pre-loaded in namespace
- Clear tool descriptions provided

---

## Next Steps

1. Run the task evaluation to measure actual pass rate
2. Adjust difficulty if needed (see Common Issues above)
3. Submit `main2.py` as your RL task implementation

---

## Cost Estimate

With Claude Haiku 4.5:
- ~10 runs for evaluation
- ~5-10 steps per run
- Total: ~$0.50-1.00 per full evaluation

---

## Questions?

This implementation follows all requirements from the task specification:
- ✅ Resembles real ML work
- ✅ 10-40% pass rate (to be verified)
- ✅ Prompt matches grading
- ✅ Accepts all valid solutions
- ✅ Verifies all requirements
- ✅ Teaches something useful
- ✅ Multiple solution approaches
- ✅ Diverse failure modes
- ✅ No tool-related issues
- ✅ Concise and reviewable (<300 LOC)

Ready to test and submit!

