import asyncio
import json
import logging
import numpy as np
import os
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

# Load environment variables from .env file
load_dotenv()

MAX_TOKENS = 2000


def setup_logging() -> str:
    """
    Set up logging to both file and console.
    File gets detailed DEBUG logs, console gets only INFO and above.
    Returns the log filename.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_dir / f"dropout_eval_{timestamp}.log"
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - captures everything (DEBUG and above)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler - only INFO and above (no DEBUG spam)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return str(log_filename)


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        # Make NumPy available in the namespace
        namespace = {"np": np, "numpy": np}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


def format_test_results(test_results: list[dict]) -> str:
    """
    Format test results for display.
    
    Args:
        test_results: List of test result dictionaries with keys:
                     'test_num', 'name', 'passed', 'message'
    
    Returns:
        Formatted string showing each test status
    """
    lines = []
    for test in test_results:
        status = "PASS" if test["passed"] else "FAIL"
        symbol = "+" if test["passed"] else "X"
        line = f"  [{symbol}] Test {test['test_num']}/8: {test['name']} - {status}"
        if not test["passed"]:
            line += f"\n      Reason: {test['message']}"
        lines.append(line)
    return "\n".join(lines)


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    
    # Log initial prompt (full version to DEBUG/file, preview to INFO/console)
    logging.info(f"Prompt: {len(prompt)} characters")
    logging.debug(f"Full prompt:\n{prompt}")

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")
        
        logging.info(f"Step {step + 1}: Calling Claude API...")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        expression = tool_input["expression"]
                        expr_preview = expression[:200] + "..." if len(expression) > 200 else expression
                        logging.info(f"Tool 'python_expression': Testing code ({len(expression)} chars)")
                        logging.debug(f"Code: {expr_preview}")
                        
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in expression.split("\n"):
                                print(f"{line}")
                            print("```")
                        
                        result = handler(expression)
                        
                        # Log output
                        if result["result"]:
                            output_preview = result["result"][:300] + "..." if len(result["result"]) > 300 else result["result"]
                            logging.debug(f"Output: {output_preview}")
                        if result["error"]:
                            logging.error(f"Tool error: {result['error']}")
                        
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        answer = tool_input["answer"]
                        logging.info("Tool 'submit_answer': LLM submitted final answer")
                        logging.debug(f"Submitted code ({len(str(answer))} chars):\n{answer}")
                        result = handler(answer)
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                logging.info("Answer submitted successfully")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            logging.warning("No tool use in response, ending loop early")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    logging.warning(f"Reached maximum steps ({max_steps}) without submitting answer")
    return None


def grade_dropout_implementation(code: str) -> tuple[bool, str, list[dict]]:
    """
    Grade the dropout implementation by running comprehensive tests.
    
    Returns:
        (success, message, test_results) tuple
        - success: bool indicating if all tests passed
        - message: summary message
        - test_results: list of dicts with individual test results
    """
    test_results = []
    
    try:
        # Create namespace and execute the code
        # Make NumPy available in the namespace so submitted code can use it
        namespace = {"np": np, "numpy": np}
        exec(code, namespace, namespace)
        
        if "dropout" not in namespace:
            return False, "No 'dropout' function found in submission", []
        
        dropout = namespace["dropout"]
        np.random.seed(42)
        
        # Test 1: Basic shape preservation
        try:
            x = np.random.randn(100, 50)
            result = dropout(x, p=0.5, training=True)
            if result.shape != x.shape:
                msg = f"Shape mismatch: expected {x.shape}, got {result.shape}"
                test_results.append({"test_num": 1, "name": "Shape preservation", "passed": False, "message": msg})
                logging.info(f"  Test 1/8: Shape preservation - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 1, "name": "Shape preservation", "passed": True, "message": "OK"})
            logging.info("  Test 1/8: Shape preservation - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 1, "name": "Shape preservation", "passed": False, "message": msg})
            logging.error(f"  Test 1/8: Shape preservation - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 2: Training mode - check that some values are zeroed
        try:
            x = np.ones((1000, 100))
            result = dropout(x, p=0.5, training=True)
            zero_ratio = np.sum(result == 0) / result.size
            if zero_ratio < 0.3 or zero_ratio > 0.7:
                msg = f"Expected ~50% zeros in training mode, got {zero_ratio*100:.1f}%"
                test_results.append({"test_num": 2, "name": "Dropout rate (p=0.5)", "passed": False, "message": msg})
                logging.info(f"  Test 2/8: Dropout rate (p=0.5) - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 2, "name": "Dropout rate (p=0.5)", "passed": True, "message": "OK"})
            logging.info("  Test 2/8: Dropout rate (p=0.5) - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 2, "name": "Dropout rate (p=0.5)", "passed": False, "message": msg})
            logging.error(f"  Test 2/8: Dropout rate (p=0.5) - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 3: Training mode - check that non-zero values are scaled correctly
        try:
            non_zero_values = result[result != 0]
            if len(non_zero_values) > 0:
                expected_scaling = 1.0 / (1 - 0.5)
                actual_values = non_zero_values / expected_scaling
                if not np.allclose(actual_values, 1.0, atol=0.01):
                    msg = "Non-zero values are not scaled correctly (should be scaled by 1/(1-p))"
                    test_results.append({"test_num": 3, "name": "Scaling verification", "passed": False, "message": msg})
                    logging.info(f"  Test 3/8: Scaling verification - FAIL")
                    logging.error(f"    {msg}")
                    return False, msg, test_results
            test_results.append({"test_num": 3, "name": "Scaling verification", "passed": True, "message": "OK"})
            logging.info("  Test 3/8: Scaling verification - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 3, "name": "Scaling verification", "passed": False, "message": msg})
            logging.error(f"  Test 3/8: Scaling verification - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 4: Eval mode - no dropout should be applied
        try:
            x = np.ones((100, 50))
            result = dropout(x, p=0.5, training=False)
            if not np.allclose(result, x):
                msg = "In eval mode (training=False), output should equal input (no dropout)"
                test_results.append({"test_num": 4, "name": "Eval mode", "passed": False, "message": msg})
                logging.info(f"  Test 4/8: Eval mode - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 4, "name": "Eval mode", "passed": True, "message": "OK"})
            logging.info("  Test 4/8: Eval mode - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 4, "name": "Eval mode", "passed": False, "message": msg})
            logging.error(f"  Test 4/8: Eval mode - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 5: Test with different dropout rates
        try:
            x = np.ones((2000, 100))
            result = dropout(x, p=0.3, training=True)
            zero_ratio = np.sum(result == 0) / result.size
            if zero_ratio < 0.15 or zero_ratio > 0.45:
                msg = f"With p=0.3, expected ~30% zeros, got {zero_ratio*100:.1f}%"
                test_results.append({"test_num": 5, "name": "Dropout rate (p=0.3)", "passed": False, "message": msg})
                logging.info(f"  Test 5/8: Dropout rate (p=0.3) - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 5, "name": "Dropout rate (p=0.3)", "passed": True, "message": "OK"})
            logging.info("  Test 5/8: Dropout rate (p=0.3) - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 5, "name": "Dropout rate (p=0.3)", "passed": False, "message": msg})
            logging.error(f"  Test 5/8: Dropout rate (p=0.3) - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 6: p=0 means no dropout
        try:
            x = np.random.randn(100, 50)
            result = dropout(x, p=0.0, training=True)
            if not np.allclose(result, x):
                msg = "With p=0, no dropout should be applied even in training mode"
                test_results.append({"test_num": 6, "name": "Edge case (p=0)", "passed": False, "message": msg})
                logging.info(f"  Test 6/8: Edge case (p=0) - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 6, "name": "Edge case (p=0)", "passed": True, "message": "OK"})
            logging.info("  Test 6/8: Edge case (p=0) - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 6, "name": "Edge case (p=0)", "passed": False, "message": msg})
            logging.error(f"  Test 6/8: Edge case (p=0) - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 7: p=1 means all dropped
        try:
            x = np.random.randn(100, 50)
            result = dropout(x, p=1.0, training=True)
            if not np.allclose(result, 0):
                msg = "With p=1, all values should be dropped"
                test_results.append({"test_num": 7, "name": "Edge case (p=1)", "passed": False, "message": msg})
                logging.info(f"  Test 7/8: Edge case (p=1) - FAIL")
                logging.error(f"    {msg}")
                return False, msg, test_results
            test_results.append({"test_num": 7, "name": "Edge case (p=1)", "passed": True, "message": "OK"})
            logging.info("  Test 7/8: Edge case (p=1) - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 7, "name": "Edge case (p=1)", "passed": False, "message": msg})
            logging.error(f"  Test 7/8: Edge case (p=1) - FAIL: {msg}")
            return False, msg, test_results
        
        # Test 8: Verify scaling with p=0.8
        try:
            x = np.ones((5000, 100))
            result = dropout(x, p=0.8, training=True)
            non_zero_values = result[result != 0]
            if len(non_zero_values) > 0:
                expected_scaling = 1.0 / (1 - 0.8)
                # Non-zero values should be 1 * (1/(1-0.8)) = 5
                if not np.allclose(non_zero_values, expected_scaling, atol=0.01):
                    msg = f"With p=0.8, non-zero values should be scaled to {expected_scaling}, got mean {non_zero_values.mean():.2f}"
                    test_results.append({"test_num": 8, "name": "High dropout scaling (p=0.8)", "passed": False, "message": msg})
                    logging.info(f"  Test 8/8: High dropout scaling (p=0.8) - FAIL")
                    logging.error(f"    {msg}")
                    return False, msg, test_results
            test_results.append({"test_num": 8, "name": "High dropout scaling (p=0.8)", "passed": True, "message": "OK"})
            logging.info("  Test 8/8: High dropout scaling (p=0.8) - PASS")
        except Exception as e:
            msg = f"Error in test: {str(e)}"
            test_results.append({"test_num": 8, "name": "High dropout scaling (p=0.8)", "passed": False, "message": msg})
            logging.error(f"  Test 8/8: High dropout scaling (p=0.8) - FAIL: {msg}")
            return False, msg, test_results
        
        return True, "All tests passed!", test_results
        
    except Exception as e:
        msg = f"Error during grading: {str(e)}"
        logging.error(f"Grading error: {msg}")
        return False, msg, test_results


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grader: Callable[[Any], tuple[bool, str, list[dict]]],
    verbose: bool = False,
) -> tuple[int, bool, Any, str, list[dict]]:
    logging.info(f"--- Run {run_id}/{num_runs} Started ---")
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,
        verbose=verbose,
    )

    # Grade the result
    test_results = []
    if result is None:
        success = False
        message = "No answer submitted"
        logging.warning(f"Run {run_id}: No answer submitted")
    else:
        logging.info(f"Run {run_id}: Grading submission...")
        success, message, test_results = grader(result)
        
        # Log test summary
        passed_count = sum(1 for t in test_results if t["passed"])
        logging.info(f"Run {run_id}: Passed {passed_count}/{len(test_results)} tests")

    # Print result with test breakdown
    if success:
        print(f"[PASS] Run {run_id}: SUCCESS - {message}")
        if test_results:
            print(format_test_results(test_results))
        logging.info(f"Run {run_id}: PASSED")
    else:
        print(f"[FAIL] Run {run_id}: FAILURE - {message}")
        if test_results:
            print(format_test_results(test_results))
        logging.error(f"Run {run_id}: FAILED - {message}")

    return run_id, success, result, message, test_results


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression. Use this to test your implementation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results. NumPy is available as 'np'.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit your final dropout implementation as a complete Python function",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete dropout function implementation as a string",
                    }
                },
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Dropout implementation task prompt
    prompt = """Implement a dropout function from scratch in NumPy.

Your function should have the following signature:
def dropout(x, p, training):
    # Your implementation here
    pass

Requirements:
1. x: Input array (can be any shape)
2. p: Dropout probability (float between 0 and 1) - the probability of dropping a unit
3. training: Boolean indicating whether in training or evaluation mode

Behavior:
- In TRAINING mode (training=True):
  - Randomly set elements to zero with probability p
  - Scale the remaining (non-zero) elements by 1/(1-p) to maintain expected value
  
- In EVALUATION mode (training=False):
  - Return the input unchanged (no dropout applied)

Important notes:
- Use NumPy (available as 'np') for all operations
- Do NOT use any deep learning libraries (PyTorch, TensorFlow, etc.)
- Ensure the output has the same shape as the input
- The scaling factor is crucial: non-zero values should be multiplied by 1/(1-p)

You can use the python_expression tool to test your implementation. When you're confident it's correct, submit your complete function using submit_answer.

Example test you might run:
```python
import numpy as np
# Your dropout function here
def dropout(x, p, training):
    ...

# Test it
x = np.ones((100, 50))
result = dropout(x, p=0.5, training=True)
print(f"Shape: {result.shape}")
print(f"Zeros: {(result == 0).sum() / result.size * 100:.1f}%")
print(f"Non-zero mean: {result[result != 0].mean():.2f}")
```"""

    # Setup logging
    log_filename = setup_logging()
    
    # Run the test multiple times and track success rate
    num_runs = 3
    model = "claude-haiku-4-5"
    execution_mode = "concurrently" if concurrent else "sequentially"
    
    logging.info("=" * 60)
    logging.info("=== Dropout Evaluation Started ===")
    logging.info(f"Configuration: {num_runs} runs, concurrent={concurrent}, model={model}")
    logging.info(f"Log file: {log_filename}")
    logging.info("=" * 60)
    
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print(f"Log file: {log_filename}")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            grader=grade_dropout_implementation,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(1 for _, success, _, _, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    
    # Aggregate test case statistics
    test_case_stats = {}
    for run_id, success, result, message, test_results in results:
        for test in test_results:
            test_name = test["name"]
            if test_name not in test_case_stats:
                test_case_stats[test_name] = {"passed": 0, "total": 0, "test_num": test["test_num"]}
            test_case_stats[test_name]["total"] += 1
            if test["passed"]:
                test_case_stats[test_name]["passed"] += 1
    
    # Log and print results
    logging.info("=" * 60)
    logging.info("=== Evaluation Complete ===")
    logging.info(f"Results: {successes}/{num_runs} passed ({pass_rate:.1f}%)")
    
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    
    # Show failure reasons
    failures = [(i, msg) for i, success, _, msg, _ in results if not success]
    if failures:
        print(f"\nFailure Summary:")
        logging.info("Failure Summary:")
        for run_id, msg in failures:
            print(f"  Run {run_id}: {msg}")
            logging.info(f"  Run {run_id}: {msg}")
    
    # Show test case statistics
    if test_case_stats:
        print(f"\nTest Case Statistics:")
        logging.info("Test Case Statistics:")
        sorted_tests = sorted(test_case_stats.items(), key=lambda x: x[1]["test_num"])
        for test_name, stats in sorted_tests:
            pass_count = stats["passed"]
            total = stats["total"]
            test_pass_rate = (pass_count / total * 100) if total > 0 else 0
            status = " [COMMON FAILURE]" if pass_count < total * 0.5 else ""
            line = f"  Test {stats['test_num']}/8 ({test_name}): {pass_count}/{total} passed ({test_pass_rate:.0f}%){status}"
            print(line)
            logging.info(line)
    
    print(f"{'=' * 60}")
    logging.info("=" * 60)


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
