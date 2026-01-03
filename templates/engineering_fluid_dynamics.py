"""Test template for computational fluid dynamics"""
import random

def generate_test():
    """Generate random CFD test data"""
    # Generate random fluid properties
    fluid = {
        'density': random.uniform(0.5, 2.0),  # kg/m³
        'viscosity': random.uniform(1e-6, 1e-3)  # Pa⋅s
    }
    
    # Generate random geometry
    geometry = {
        'length': random.uniform(0.1, 2.0),  # m
        'width': random.uniform(0.01, 0.5)  # m
    }
    
    # Generate random flow conditions
    flow = {
        'inlet_velocity': random.uniform(0.1, 10.0),  # m/s
        'inlet_pressure': random.uniform(90000, 110000)  # Pa
    }
    
    return {
        'fluid': fluid,
        'geometry': geometry,
        'flow': flow,
        'grid_x': random.randint(10, 30),
        'grid_y': random.randint(5, 15)
    }

def run_test(test_data):
    """Run CFD test"""
    try:
        from core.physics_simulator import PhysicsSimulator
        physics = PhysicsSimulator()
        
        result = physics.computational_fluid_dynamics(test_data)
        
        success = result is not None and isinstance(result, dict) and result.get('success', False)
        
        return {
            'success': success,
            'result': result,
            'details': f"CFD simulation: {'SUCCESS' if success else 'FAILED'}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'details': f"Failed to run CFD simulation"
        }