"""Test template for matrix operations"""
import random

def generate_test():
    """Generate random matrix test data"""
    operations = ['determinant', 'inverse', 'eigenvalues', 'trace', 'multiply', 'add']
    op = random.choice(operations)
    
    # Generate matrix size (2x2 to 4x4)
    size = random.randint(2, 4)
    
    # Generate random matrix data
    matrix_data = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(random.uniform(-10, 10))
        matrix_data.append(row)
    
    test_data = {
        'operation': op,
        'matrix_data': matrix_data,
        'size': size
    }
    
    # For binary operations, generate second matrix
    if op in ['multiply', 'add']:
        matrix_data2 = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(random.uniform(-10, 10))
            matrix_data2.append(row)
        test_data['matrix_data2'] = matrix_data2
    
    return test_data

def run_test(test_data):
    """Run matrix operation test"""
    try:
        from core.matrix_operations import MatrixOperations
        matrix_ops = MatrixOperations()
        
        # Create matrix
        matrix = matrix_ops.create_matrix(test_data['matrix_data'])
        op = test_data['operation']
        
        if op == 'determinant':
            result = matrix_ops.matrix_determinant(matrix['matrix'])
        elif op == 'inverse':
            result = matrix_ops.matrix_inverse(matrix['matrix'])
        elif op == 'eigenvalues':
            result = matrix_ops.matrix_eigenvalues(matrix['matrix'])
        elif op == 'trace':
            # Calculate trace manually since no direct method exists
            if hasattr(matrix['matrix'], 'trace'):
                trace_val = matrix['matrix'].trace()
                result = {'success': True, 'trace': float(trace_val)}
            else:
                import numpy as np
                result = {'success': True, 'trace': float(np.trace(matrix['matrix']))}
        elif op == 'multiply' and 'matrix_data2' in test_data:
            matrix2 = matrix_ops.create_matrix(test_data['matrix_data2'])
            result = matrix_ops.matrix_multiplication(matrix['matrix'], matrix2['matrix'])
        elif op == 'add' and 'matrix_data2' in test_data:
            matrix2 = matrix_ops.create_matrix(test_data['matrix_data2'])
            result = matrix_ops.matrix_addition(matrix['matrix'], matrix2['matrix'])
        else:
            result = None
        
        success = result is not None and isinstance(result, dict) and result.get('success', False)
        
        return {
            'success': success,
            'result': result,
            'details': f"Matrix {op} on {test_data['size']}x{test_data['size']} matrix: {'SUCCESS' if success else 'FAILED'}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Failed to perform matrix {test_data['operation']}"
        }