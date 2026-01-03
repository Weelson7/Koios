"""Test template for calculus operations"""
import random

def generate_test():
    """Generate random calculus test data"""
    expressions = ['x^2', 'sin(x)', 'cos(x)', 'exp(x)', 'ln(x)', 'x^3 + 2*x^2 + x + 1']
    operations = ['differentiate', 'integrate', 'limit']
    
    expr = random.choice(expressions)
    op = random.choice(operations)
    variable = 'x'
    
    test_data = {
        'expression': expr,
        'operation': op,
        'variable': variable
    }
    
    if op == 'limit':
        test_data['point'] = random.choice([0, 1, -1, 'oo'])
    elif op == 'differentiate':
        test_data['order'] = random.randint(1, 3)
    
    return test_data

def run_test(test_data):
    """Run calculus operation test"""
    try:
        from core.calculus_engine import CalculusEngine
        calculus = CalculusEngine()
        
        op = test_data['operation']
        expr = test_data['expression']
        var = test_data['variable']
        
        if op == 'differentiate':
            order = test_data.get('order', 1)
            result = calculus.differentiate(expr, var, order)
        elif op == 'integrate':
            result = calculus.integrate(expr, var)
        elif op == 'limit':
            point = test_data.get('point', 0)
            result = calculus.limit(expr, var, point)
        else:
            result = None
        
        return {
            'success': result is not None,
            'result': result,
            'details': f"{op.title()} of {expr} with respect to {var}: {result}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Failed to compute {test_data['operation']} of {test_data['expression']}"
        }