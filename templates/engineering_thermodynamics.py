"""Test template for engineering thermodynamics"""
import random

def generate_test():
    """Generate random thermodynamics test data"""
    processes = ['isothermal', 'adiabatic', 'isobaric', 'isochoric']
    process = random.choice(processes)
    
    # Generate random thermodynamic state data
    state_data = {
        'pressure1': random.uniform(50000, 200000),  # Pa
        'volume1': random.uniform(0.0001, 0.01),  # m³
        'temperature1': random.uniform(250, 400),  # K
        'volume2': random.uniform(0.0001, 0.01),  # m³
        'temperature2': random.uniform(250, 400)  # K
    }
    
    return {
        'process': process,
        'state_data': state_data
    }

def run_test(test_data):
    """Run thermodynamics test"""
    try:
        from core.calculation_engine import CalculationEngine
        calc = CalculationEngine()
        
        result = calc.thermodynamics(test_data['process'], test_data['state_data'])
        
        success = result is not None and isinstance(result, dict) and result.get('success', False)
        
        return {
            'success': success,
            'result': result,
            'details': f"Thermodynamics {test_data['process']} process: {'SUCCESS' if success else 'FAILED'}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Failed to compute thermodynamics {test_data['process']} process"
        }