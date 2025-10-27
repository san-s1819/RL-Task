# Logging System Guide

## Overview

The dropout evaluation task now includes comprehensive logging to track LLM interactions, tool usage, test execution, and results. Each evaluation run creates a timestamped log file with detailed information.

## Log Files

### Location
All log files are created in the `logs/` directory (automatically created if it doesn't exist).

### Naming Convention
```
dropout_eval_YYYY-MM-DD_HH-MM-SS.log
```

Example: `dropout_eval_2025-10-26_14-30-15.log`

### Git Ignore
Log files are automatically ignored by git (configured in `.gitignore`).

## What Gets Logged

### 1. Evaluation Configuration
- Number of runs
- Execution mode (concurrent vs sequential)
- Model being used
- Log file path

### 2. Per-Run Information
- Run start marker
- Initial prompt (first 100 characters)
- Each API call to Claude
- Tool usage:
  - `python_expression`: Code being tested (first 200 chars) and output (first 300 chars)
  - `submit_answer`: Submission event
- Answer submission or timeout

### 3. Grading Details
- Start of grading
- Each test case execution (1-8):
  - Test name
  - Pass/Fail status
  - Error message if failed
- Test summary (X/8 tests passed)
- Final result (PASS/FAIL)

### 4. Aggregate Statistics
- Overall pass rate
- Failure summary (which runs failed and why)
- Test case statistics:
  - How many times each test was passed/failed
  - Identification of common failure points

## Log Levels

### Console Output (INFO and above)
- Normal operation events
- Configuration information
- Run progress
- Test case results (pass/fail)
- Final statistics
- **No DEBUG spam** - keeps console clean
- **ASCII-safe** - Unicode characters (like ✓) are automatically converted to avoid Windows encoding errors

### File Output (DEBUG and above)
Everything from console PLUS:
- **Full prompt text** (complete prompt sent to LLM)
- **Full submitted code** (complete dropout implementation)
- Detailed code snippets from python_expression tool
- Tool outputs (complete, not truncated)
- Internal state information
- HTTP request details from Anthropic API

### WARNING
- No answer submitted
- Early loop termination
- Timeout warnings

### ERROR
- Failed test cases
- Failed runs
- Grading errors
- Tool execution errors

## Example Log Output

```log
2025-10-26 14:30:15 [INFO] ============================================================
2025-10-26 14:30:15 [INFO] === Dropout Evaluation Started ===
2025-10-26 14:30:15 [INFO] Configuration: 10 runs, concurrent=True, model=claude-haiku-4-5
2025-10-26 14:30:15 [INFO] Log file: logs/dropout_eval_2025-10-26_14-30-15.log
2025-10-26 14:30:15 [INFO] ============================================================

2025-10-26 14:30:15 [INFO] --- Run 1/10 Started ---
2025-10-26 14:30:15 [INFO] Prompt (437 chars): "Implement a dropout function from scratch in NumPy...."
2025-10-26 14:30:16 [INFO] Step 1: Calling Claude API...
2025-10-26 14:30:16 [INFO] Tool 'python_expression': Testing code (215 chars)
2025-10-26 14:30:16 [DEBUG] Code: def dropout(x, p, training):...
2025-10-26 14:30:16 [DEBUG] Output: Shape: (100, 50)
Zeros: 48.2%

2025-10-26 14:30:18 [INFO] Step 2: Calling Claude API...
2025-10-26 14:30:18 [INFO] Tool 'submit_answer': LLM submitted final answer
2025-10-26 14:30:18 [INFO] Answer submitted successfully
2025-10-26 14:30:18 [INFO] Run 1: Grading submission...
2025-10-26 14:30:18 [INFO]   Test 1/8: Shape preservation - PASS
2025-10-26 14:30:18 [INFO]   Test 2/8: Dropout rate (p=0.5) - PASS
2025-10-26 14:30:18 [INFO]   Test 3/8: Scaling verification - FAIL
2025-10-26 14:30:18 [ERROR]    Non-zero values are not scaled correctly (should be scaled by 1/(1-p))
2025-10-26 14:30:18 [INFO] Run 1: Passed 2/8 tests
2025-10-26 14:30:18 [ERROR] Run 1: FAILED - Non-zero values are not scaled correctly

... (runs 2-10) ...

2025-10-26 14:35:42 [INFO] ============================================================
2025-10-26 14:35:42 [INFO] === Evaluation Complete ===
2025-10-26 14:35:42 [INFO] Results: 3/10 passed (30.0%)
2025-10-26 14:35:42 [INFO] Failure Summary:
2025-10-26 14:35:42 [INFO]   Run 1: Non-zero values are not scaled correctly
2025-10-26 14:35:42 [INFO]   Run 2: No answer submitted
2025-10-26 14:35:42 [INFO]   ...
2025-10-26 14:35:42 [INFO] Test Case Statistics:
2025-10-26 14:35:42 [INFO]   Test 1/8 (Shape preservation): 10/10 passed (100%)
2025-10-26 14:35:42 [INFO]   Test 2/8 (Dropout rate (p=0.5)): 9/10 passed (90%)
2025-10-26 14:35:42 [INFO]   Test 3/8 (Scaling verification): 4/10 passed (40%) [COMMON FAILURE]
2025-10-26 14:35:42 [INFO]   Test 4/8 (Eval mode): 8/10 passed (80%)
2025-10-26 14:35:42 [INFO]   Test 5/8 (Dropout rate (p=0.3)): 7/10 passed (70%)
2025-10-26 14:35:42 [INFO]   Test 6/8 (Edge case (p=0)): 9/10 passed (90%)
2025-10-26 14:35:42 [INFO]   Test 7/8 (Edge case (p=1)): 9/10 passed (90%)
2025-10-26 14:35:42 [INFO]   Test 8/8 (High dropout scaling (p=0.8)): 6/10 passed (60%)
2025-10-26 14:35:42 [INFO] ============================================================
```

## Console Output

In addition to the log file, important information is also printed to the console:

### Per-Run Output
```
[PASS] Run 3: SUCCESS - All tests passed!
  [+] Test 1/8: Shape preservation - PASS
  [+] Test 2/8: Dropout rate (p=0.5) - PASS
  [+] Test 3/8: Scaling verification - PASS
  [+] Test 4/8: Eval mode - PASS
  [+] Test 5/8: Dropout rate (p=0.3) - PASS
  [+] Test 6/8: Edge case (p=0) - PASS
  [+] Test 7/8: Edge case (p=1) - PASS
  [+] Test 8/8: High dropout scaling (p=0.8) - PASS

[FAIL] Run 1: FAILURE - Non-zero values are not scaled correctly
  [+] Test 1/8: Shape preservation - PASS
  [+] Test 2/8: Dropout rate (p=0.5) - PASS
  [X] Test 3/8: Scaling verification - FAIL
      Reason: Non-zero values are not scaled correctly (should be scaled by 1/(1-p))
```

### Summary Output
```
============================================================
Test Results:
  Passed: 3/10
  Failed: 7/10
  Pass Rate: 30.0%

Failure Summary:
  Run 1: Non-zero values are not scaled correctly
  Run 2: No answer submitted
  ...

Test Case Statistics:
  Test 1/8 (Shape preservation): 10/10 passed (100%)
  Test 2/8 (Dropout rate (p=0.5)): 9/10 passed (90%)
  Test 3/8 (Scaling verification): 4/10 passed (40%) [COMMON FAILURE]
  ...
============================================================
```

## Using the Logs

### Finding Issues
1. Look for `[ERROR]` entries to find failures
2. Check test case statistics to identify common failure points
3. Review tool outputs (DEBUG level) to see what the LLM was trying

### Debugging
1. Search for a specific run: `grep "Run 3" logfile.log`
2. Find all failures: `grep "\[ERROR\]" logfile.log`
3. See what tests failed most: Look at the Test Case Statistics section

### Analysis
- Compare multiple runs to see if failures are consistent
- Identify which test cases are hardest for the model
- Track improvement over time by comparing log files

## Configuration

### Changing Log Level
In `main2.py`, modify the `setup_logging()` function:

```python
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detail
    ...
)
```

### Disabling Console Output
To log only to file without console output:

```python
handlers=[
    logging.FileHandler(log_filename),
    # Remove or comment out the StreamHandler
    # logging.StreamHandler()
]
```

## Benefits

1. **Debugging**: Easily trace what went wrong in failed runs
2. **Analysis**: Understand which tests are most challenging
3. **Transparency**: Full visibility into LLM interactions
4. **Reproducibility**: Complete record of each evaluation
5. **Improvement**: Track changes in pass rates over time

