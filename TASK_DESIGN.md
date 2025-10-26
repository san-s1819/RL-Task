# RL Task Design: Dropout Implementation

## Task Overview
This task asks an LLM to implement the dropout regularization technique from scratch using NumPy, without relying on any deep learning frameworks.

## Learning Objectives
This task teaches the model:
1. **Core ML concept**: How dropout regularization works
2. **Scaling importance**: Why we scale by 1/(1-p) during training
3. **Train vs Eval modes**: Different behavior in training vs inference
4. **NumPy operations**: Array manipulation and random masking

## Task Requirements Checklist

### ✅ Realistic ML Work
- Implementing layers from scratch is common when:
  - Learning fundamentals
  - Custom implementations needed
  - Understanding what frameworks do under the hood
  - Teaching/research contexts

### ✅ Expected Pass Rate: 10-40%
**Common failure modes:**
1. **Forgetting to scale** (most common) - just zeros without multiplication
2. **Wrong scaling factor** - using p instead of 1/(1-p)
3. **No train/eval distinction** - applying dropout in both modes
4. **Incorrect random masking** - wrong probability or method
5. **Not using NumPy** - trying to import PyTorch/TensorFlow

Multiple implementations are valid as long as they satisfy the requirements.

### ✅ Prompt Matches Grading
Every requirement in the prompt is checked:
- Input shape preservation ✓
- Random zeroing with probability p ✓
- Scaling by 1/(1-p) ✓
- Train vs eval mode ✓
- Edge cases (p=0, p=1) ✓

### ✅ Accepts All Valid Solutions
The grader uses:
- Statistical checks (approximate zeros percentage)
- Numerical tolerance for scaling verification
- No exact string matching or rigid structure requirements
- Tests behavior, not implementation details

### ✅ Hard to Guess
To pass, the model must:
- Understand the formula: `output = mask * input / (1-p)` where mask has p% zeros
- Handle train/eval mode correctly
- Get the scaling factor right (not guessable)
- Pass 8 different test cases with varying p values

### ✅ Multiple Solution Approaches
Valid implementations can use:
- `np.random.rand()` or `np.random.binomial()`
- Boolean or float masking
- Different random number generation strategies
- In-place or copy operations

### ✅ Diverse Failure Modes
Models fail for various reasons:
1. Conceptual misunderstanding (no scaling)
2. Math errors (wrong scaling factor)
3. Implementation bugs (train/eval not handled)
4. Edge case misses (p=0, p=1)
5. Using forbidden libraries

Not a single bottleneck.

### ✅ No Tool-Related Failures
- `python_expression` tool is straightforward
- NumPy is pre-loaded in namespace
- Model can test iteratively before submitting
- Clear error messages guide debugging

### ✅ Concise and Reviewable
Total implementation: ~180 lines
- Prompt: ~30 lines (clear and specific)
- Grading function: ~70 lines (8 test cases)
- Infrastructure: ~80 lines (reusable framework)

## Grading Function Design

### Test Cases
1. **Shape preservation** - Basic sanity check
2. **Dropout rate** - Statistical check that ~p% are zeros
3. **Scaling verification** - Non-zero values are scaled correctly
4. **Eval mode** - No dropout applied when training=False
5. **Different p values** - Works with p=0.3
6. **Edge case p=0** - No dropout even in training mode
7. **Edge case p=1** - All values dropped
8. **High p scaling** - Correct scaling with p=0.8

### Why These Tests?
- Comprehensive coverage of all requirements
- Statistical checks allow implementation flexibility
- Edge cases catch off-by-one errors
- Multiple p values ensure proper formula usage

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

## Why This Task is Good for RL Training

1. **Clear reward signal**: Binary pass/fail with specific error messages
2. **Intermediate feedback**: Can test with python_expression tool before submitting
3. **Proper difficulty**: Not too easy (requires understanding), not too hard (solvable)
4. **Real-world relevance**: Actual ML skill used in practice
5. **Multiple learning paths**: Model can experiment and learn from failures

## Potential Extensions

If this task proves too easy/hard, consider:
- **Easier**: Implement without scaling requirement
- **Harder**: Add spatial dropout (drop entire feature maps)
- **Harder**: Implement variational dropout
- **Related**: Implement batch normalization with train/eval modes

