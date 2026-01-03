"""Test template for trigonometric functions"""
import random
import math

def generate_test():
    """Generate random trigonometric test data"""
    functions = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
    func = random.choice(functions)
    
    # Generate appropriate values for each function
    if func in ['asin', 'acos']:
        value = random.uniform(-1, 1)
    elif func == 'atan':
        value = random.uniform(-10, 10)
    else:
        value = random.uniform(-2*math.pi, 2*math.pi)
    
    return {
        'expression': f"{func}({value})",
        'function': func,
        'value': value
    }

def run_test(test_data):
    """Run trigonometric function test"""
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