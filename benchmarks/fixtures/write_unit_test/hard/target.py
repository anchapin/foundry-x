def calculator(expression):
    """Evaluate a simple arithmetic expression like '2 + 3' or '10 - 4'.

    Supports: +, -, *, /
    Returns None if division by zero would occur.
    Raises ValueError for invalid expressions.
    """
    try:
        parts = expression.split()
        if len(parts) != 3:
            raise ValueError("Expression must be '<num> <op> <num>'")
        left, op, right = parts
        left = float(left)
        right = float(right)

        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            if right == 0:
                return None
            return left / right
        else:
            raise ValueError(f"Unknown operator: {op}")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        raise ValueError(f"Invalid expression: {expression}")
