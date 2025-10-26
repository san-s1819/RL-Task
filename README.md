RL Task: Implement Dropout from Scratch
===

This is an RL training task for LLMs that teaches them to implement the dropout regularization technique from scratch using NumPy.

## Task Description

The LLM must implement a `dropout` function that:
- Takes an input array, dropout probability `p`, and `training` mode flag
- In training mode: randomly zeros out elements with probability `p` and scales remaining values by `1/(1-p)`
- In evaluation mode: returns input unchanged
- Properly handles edge cases (p=0, p=1, different shapes)

This task teaches fundamental ML concepts:
- Dropout regularization mechanism
- Train vs eval mode differences
- Proper scaling to maintain expected values
- NumPy array operations

## Why This Task?

**Target Pass Rate:** 10-40%
- Tests understanding of a core ML regularization technique
- Requires knowledge of proper scaling (often forgotten)
- Multiple ways to fail: wrong scaling, forgetting train/eval mode, incorrect masking
- Realistic ML engineering skill used daily

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Install dependencies:
   ```
   uv sync
   ```

4. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Run the task evaluation (main2.py has the dropout task):
   ```
   uv run main2.py
   ```

## Grading Criteria

The implementation is tested on:
1. ✅ Shape preservation
2. ✅ Correct dropout rate (~p% zeros in training mode)
3. ✅ Proper scaling of non-zero values (by 1/(1-p))
4. ✅ No dropout in eval mode
5. ✅ Different dropout probabilities (p=0.3, p=0.8)
6. ✅ Edge cases (p=0, p=1)

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main2.py`:

```python
asyncio.run(main(concurrent=True))   # Fast parallel execution
asyncio.run(main(concurrent=False))  # Sequential execution
```

When running concurrently, results print as they complete for faster overall execution.

## Example Run

```
Running 10 test iterations concurrently...
============================================================
✓ Run 3: SUCCESS - All tests passed!
✗ Run 1: FAILURE - Non-zero values are not scaled correctly
✓ Run 5: SUCCESS - All tests passed!
...
============================================================
Test Results:
  Passed: 3/10
  Failed: 7/10
  Pass Rate: 30.0%
============================================================
```
