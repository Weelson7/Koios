"""Test template for basic arithmetic operations"""
import random

def generate_test():
    """Generate random arithmetic test data"""
    operators = ['+', '-', '*', '/', '**', '%']
    a = random.randint(-100, 100)
    b = random.randint(1, 100)  # Avoid division by zero
    op = random.choice(operators)
    
    if op == '/' and b == 0:
        b = 1
    
    return {
        'expression': f"({a}) {op} ({b})",
        'operand_a': a,
        'operand_b': b,
        'operator': op
    }

def run_test(test_data):
    """Run arithmetic test"""
    try:
        from core.calculation_engine import CalculationEngine
        calc = CalculationEngine()
        
        result = calc.evaluate(test_data['expression'])
        
        return {
            'success': result is not None,
            'result': result,
            'details': f"Evaluated {test_data['expression']} = {result}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Failed to evaluate {test_data['expression']}"
        }