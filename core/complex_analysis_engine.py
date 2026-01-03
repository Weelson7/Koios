import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import matplotlib.pyplot as plt
from matplotlib import cm

class ComplexAnalysisEngine:
    """Engine for complex number calculations and analysis"""
    
    def __init__(self):
        self.i = complex(0, 1)
        self.tol = 1e-10  # Numerical tolerance
    
    def parse_complex(self, expr: str) -> complex:
        """Parse string expression to complex number"""
        try:
            # Handle various notations
            expr = expr.replace('i', 'j').replace('I', 'j')
            return complex(expr)
        except:
            # Try SymPy parsing
            parsed = sp.sympify(expr)
            return complex(parsed)
    
    def complex_arithmetic(self, z1: complex, z2: complex, operation: str = None) -> Dict[str, complex]:
        """Perform all basic complex arithmetic operations"""
        if operation:
            # If specific operation is requested
            if operation == '+':
                return {'result': z1 + z2}
            elif operation == '-':
                return {'result': z1 - z2}
            elif operation == '*':
                return {'result': z1 * z2}
            elif operation == '/':
                return {'result': z1 / z2 if z2 != 0 else None}
            elif operation == '**':
                return {'result': z1 ** z2}
            else:
                return {'error': f'Unknown operation: {operation}'}
        
        # Return all operations if no specific operation requested
        return {
            'addition': z1 + z2,
            'subtraction': z1 - z2,
            'multiplication': z1 * z2,
            'division': z1 / z2 if z2 != 0 else None,
            'power': z1 ** z2,
            'conjugate_z1': np.conj(z1),
            'conjugate_z2': np.conj(z2),
            'modulus_z1': abs(z1),
            'modulus_z2': abs(z2),
            'argument_z1': np.angle(z1),
            'argument_z2': np.angle(z2)
        }
    
    def polar_form(self, z: complex) -> Dict[str, float]:
        """Convert complex number to polar form"""
        r = abs(z)
        theta = np.angle(z)
        return {
            'r': r,
            'theta': theta,
            'theta_degrees': np.degrees(theta),
            'polar_form': f"{r:.4f} * exp({theta:.4f}i)"
        }
    
    def exponential_form(self, z: complex) -> Dict[str, Any]:
        """Express complex number in exponential form"""
        r = abs(z)
        theta = np.angle(z)
        return {
            'exponential': f"{r:.4f} * e^({theta:.4f}i)",
            'euler': f"{r:.4f} * (cos({theta:.4f}) + i*sin({theta:.4f}))",
            'r': r,
            'theta': theta
        }
    
    def complex_roots(self, z: complex, n: int) -> List[complex]:
        """Find all n-th roots of complex number"""
        r = abs(z)
        theta = np.angle(z)
        
        roots = []
        for k in range(n):
            root_r = r ** (1/n)
            root_theta = (theta + 2*np.pi*k) / n
            root = root_r * np.exp(1j * root_theta)
            roots.append(root)
        
        return roots
    
    def complex_logarithm(self, z: complex, branch: int = 0) -> complex:
        """Complex logarithm with branch selection"""
        if z == 0:
            return float('-inf')
        
        r = abs(z)
        theta = np.angle(z)
        
        # Principal branch + 2πi*branch
        return np.log(r) + 1j * (theta + 2*np.pi*branch)
    
    def complex_power(self, z: complex, w: complex) -> complex:
        """Complex power z^w"""
        if z == 0:
            return 0 if w.real > 0 else float('inf')
        
        # z^w = exp(w * log(z))
        return np.exp(w * self.complex_logarithm(z))
    
    def analytic_functions(self, z: complex) -> Dict[str, complex]:
        """Evaluate common analytic functions"""
        return {
            'exp': np.exp(z),
            'sin': np.sin(z),
            'cos': np.cos(z),
            'tan': np.tan(z),
            'sinh': np.sinh(z),
            'cosh': np.cosh(z),
            'tanh': np.tanh(z),
            'log': self.complex_logarithm(z),
            'sqrt': np.sqrt(z),
            'arcsin': np.arcsin(z) if abs(z) <= 1 else None,
            'arccos': np.arccos(z) if abs(z) <= 1 else None,
            'arctan': np.arctan(z)
        }
    
    def derivative(self, f: Callable[[complex], complex], z: complex, 
                  h: float = 1e-8) -> complex:
        """Numerical complex derivative"""
        # Using central difference
        df_dx = (f(z + h) - f(z - h)) / (2*h)
        df_dy = (f(z + 1j*h) - f(z - 1j*h)) / (2j*h)
        
        return df_dx
    
    def cauchy_riemann_check(self, f: Callable[[complex], complex], 
                           z: complex, h: float = 1e-8) -> Dict[str, Any]:
        """Check if function satisfies Cauchy-Riemann equations"""
        # f(z) = u(x,y) + iv(x,y)
        x, y = z.real, z.imag
        
        # Partial derivatives
        u_x = (f(z + h).real - f(z - h).real) / (2*h)
        u_y = (f(z + 1j*h).real - f(z - 1j*h).real) / (2*h)
        v_x = (f(z + h).imag - f(z - h).imag) / (2*h)
        v_y = (f(z + 1j*h).imag - f(z - 1j*h).imag) / (2*h)
        
        # Cauchy-Riemann: u_x = v_y and u_y = -v_x
        cr1 = abs(u_x - v_y)
        cr2 = abs(u_y + v_x)
        
        return {
            'u_x': u_x,
            'u_y': u_y,
            'v_x': v_x,
            'v_y': v_y,
            'cr_satisfied': cr1 < 1e-6 and cr2 < 1e-6,
            'cr_error': max(cr1, cr2)
        }
    
    def contour_integral(self, f: Callable[[complex], complex], 
                        contour: Callable[[float], complex],
                        t_range: Tuple[float, float] = (0, 1),
                        n_points: int = 1000) -> complex:
        """Numerical contour integration"""
        t = np.linspace(t_range[0], t_range[1], n_points)
        dt = (t_range[1] - t_range[0]) / (n_points - 1)
        
        integral = 0
        for i in range(n_points - 1):
            z = contour(t[i])
            dz_dt = (contour(t[i+1]) - contour(t[i])) / dt
            integral += f(z) * dz_dt * dt
        
        return integral
    
    def residue(self, f: Callable[[complex], complex], pole: complex, 
               order: int = 1, radius: float = 0.01) -> complex:
        """Calculate residue at a pole"""
        if order == 1:
            # Simple pole: Res = lim (z-pole)*f(z)
            def g(z):
                return (z - pole) * f(z)
            
            # Evaluate near the pole
            return g(pole + radius)
        else:
            # Higher order pole
            # Res = 1/(n-1)! * d^(n-1)/dz^(n-1) [(z-pole)^n * f(z)]
            # Numerical approximation
            h = 1e-8
            return self.derivative(lambda z: (z - pole)**order * f(z), pole, h)
    
    def residue_theorem(self, f: Callable[[complex], complex], 
                       poles: List[Tuple[complex, int]],
                       contour: Callable[[float], complex]) -> complex:
        """Apply residue theorem for contour integration"""
        total_residue = 0
        
        for pole, order in poles:
            # Check if pole is inside contour (simplified)
            # In practice, would use winding number
            res = self.residue(f, pole, order)
            total_residue += res
        
        return 2 * np.pi * 1j * total_residue
    
    def conformal_map(self, w_func: Callable[[complex], complex], 
                     z_grid: np.ndarray) -> np.ndarray:
        """Apply conformal mapping w = f(z)"""
        w_grid = np.zeros_like(z_grid)
        
        for i in range(z_grid.shape[0]):
            for j in range(z_grid.shape[1]):
                w_grid[i, j] = w_func(z_grid[i, j])
        
        return w_grid
    
    def joukowski_transform(self, z: complex) -> complex:
        """Joukowski transformation for airfoil shapes"""
        return z + 1/z
    
    def mobius_transform(self, z: complex, a: complex, b: complex, 
                        c: complex, d: complex) -> complex:
        """Möbius transformation: w = (az + b)/(cz + d)"""
        if c*z + d == 0:
            return float('inf')
        return (a*z + b) / (c*z + d)
    
    def schwarz_christoffel(self, vertices: List[complex], 
                          angles: List[float]) -> Callable:
        """Schwarz-Christoffel mapping (simplified)"""
        # This is a simplified version
        # Full implementation would require numerical integration
        def mapping(z):
            # Placeholder for actual S-C mapping
            return z
        
        return mapping
    
    def complex_ode_solve(self, ode_func: Callable[[complex, complex], complex],
                         z0: complex, t_span: Tuple[float, float],
                         n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Solve complex ODE dz/dt = f(t, z)"""
        from scipy.integrate import solve_ivp
        
        def real_system(t, y):
            z = y[0] + 1j*y[1]
            dz = ode_func(t, z)
            return [dz.real, dz.imag]
        
        y0 = [z0.real, z0.imag]
        sol = solve_ivp(real_system, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], n_points))
        
        z_sol = sol.y[0] + 1j*sol.y[1]
        return sol.t, z_sol
    
    def visualize_complex_function(self, f: Callable[[complex], complex],
                                 x_range: Tuple[float, float] = (-2, 2),
                                 y_range: Tuple[float, float] = (-2, 2),
                                 n_points: int = 100):
        """Visualize complex function using domain coloring"""
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        
        # Evaluate function
        W = np.zeros_like(Z)
        for i in range(n_points):
            for j in range(n_points):
                W[i, j] = f(Z[i, j])
        
        # Create color map based on argument and modulus
        arg = np.angle(W)
        mod = np.abs(W)
        
        # Normalize modulus for visualization
        mod_norm = np.log(1 + mod)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Argument plot
        im1 = ax1.imshow(arg, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                        cmap='hsv', origin='lower')
        ax1.set_title('Argument of f(z)')
        ax1.set_xlabel('Re(z)')
        ax1.set_ylabel('Im(z)')
        plt.colorbar(im1, ax=ax1)
        
        # Modulus plot
        im2 = ax2.imshow(mod_norm, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                        cmap='viridis', origin='lower')
        ax2.set_title('log(1 + |f(z)|)')
        ax2.set_xlabel('Re(z)')
        ax2.set_ylabel('Im(z)')
        plt.colorbar(im2, ax=ax2)
        
        return fig
    
    def riemann_surface(self, f: Callable[[complex], complex],
                       branch_points: List[complex] = None):
        """Visualize Riemann surface (simplified)"""
        # This would create a 3D visualization of multi-valued functions
        # Implementation would be complex for general functions
        pass
    
    def laurent_series(self, f: Callable[[complex], complex], 
                      center: complex, order: int = 10) -> Dict[int, complex]:
        """Compute Laurent series coefficients numerically"""
        coefficients = {}
        radius = 0.5  # Radius for contour integration
        
        for n in range(-order, order + 1):
            # c_n = 1/(2πi) ∮ f(z)/(z-center)^(n+1) dz
            def integrand(z):
                return f(z) / (z - center)**(n + 1)
            
            # Circular contour
            def contour(t):
                return center + radius * np.exp(2j * np.pi * t)
            
            coeff = self.contour_integral(integrand, contour) / (2 * np.pi * 1j)
            coefficients[n] = coeff
        
        return coefficients

# Example applications
    def fft_analysis(self, signal: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
        """Fast Fourier Transform analysis for signal processing"""
        n = len(signal)
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)
        
        # Only positive frequencies
        pos_mask = frequencies >= 0
        frequencies = frequencies[pos_mask]
        fft_result = fft_result[pos_mask]
        
        # Magnitude and phase
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Find dominant frequency
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_freq = frequencies[dominant_idx]
        
        return {
            'frequencies': frequencies,
            'fft': fft_result,
            'magnitude': magnitude,
            'phase': phase,
            'dominant_frequency': dominant_freq,
            'power_spectrum': magnitude**2
        }
    
    def transfer_function_analysis(self, num: List[float], den: List[float]) -> Dict[str, Any]:
        """Analyze transfer function H(s) = num(s)/den(s) for control systems"""
        # Find poles and zeros
        zeros = np.roots(num)
        poles = np.roots(den)
        
        # Check stability (all poles in left half plane)
        stable = all(pole.real < 0 for pole in poles)
        
        # Calculate gain
        dc_gain = num[-1] / den[-1] if den[-1] != 0 else float('inf')
        
        return {
            'zeros': zeros,
            'poles': poles,
            'stable': stable,
            'dc_gain': dc_gain,
            'num_zeros': len(zeros),
            'num_poles': len(poles)
        }
    
    def electromagnetic_impedance(self, frequency: float, resistance: float,
                                inductance: float, capacitance: float) -> complex:
        """Calculate complex impedance for RLC circuit"""
        omega = 2 * np.pi * frequency
        Z_R = resistance
        Z_L = 1j * omega * inductance
        Z_C = 1 / (1j * omega * capacitance) if capacitance > 0 else 0
        
        return Z_R + Z_L + Z_C
    
    def laurent_series_coefficients(self, f: Callable[[complex], complex],
                                  center: complex, inner_radius: float,
                                  outer_radius: float, max_order: int = 10) -> Dict[int, complex]:
        """Calculate Laurent series coefficients around a point"""
        coefficients = {}
        
        # Positive powers (regular Taylor series part)
        for n in range(max_order + 1):
            # c_n = (1/2πi) ∮ f(z)/(z-center)^(n+1) dz
            def integrand(z):
                return f(z) / (z - center)**(n + 1)
            
            # Circular contour
            def contour(t):
                return center + outer_radius * np.exp(2j * np.pi * t)
            
            coeff = self.contour_integral(integrand, contour) / (2j * np.pi)
            coefficients[n] = coeff
        
        # Negative powers (principal part)
        for n in range(1, max_order + 1):
            # c_{-n} = (1/2πi) ∮ f(z)*(z-center)^(n-1) dz
            def integrand(z):
                return f(z) * (z - center)**(n - 1)
            
            def contour(t):
                return center + inner_radius * np.exp(2j * np.pi * t)
            
            coeff = self.contour_integral(integrand, contour) / (2j * np.pi)
            coefficients[-n] = coeff
        
        return coefficients

def example_complex_calculations():
    """Demonstrate complex analysis capabilities"""
    engine = ComplexAnalysisEngine()
    
    # Basic complex arithmetic
    z1 = 3 + 4j
    z2 = 1 - 2j
    
    arithmetic = engine.complex_arithmetic(z1, z2)
    print(f"Complex arithmetic for z1={z1}, z2={z2}:")
    for op, result in arithmetic.items():
        if result is not None:
            print(f"  {op}: {result}")
    
    # Complex roots
    z = -8
    cube_roots = engine.complex_roots(z, 3)
    print(f"\nCube roots of {z}: {cube_roots}")
    
    # Analytic functions
    z = 1 + 1j
    functions = engine.analytic_functions(z)
    print(f"\nAnalytic functions at z={z}:")
    for func, value in functions.items():
        if value is not None:
            print(f"  {func}(z) = {value}")
    
    # Cauchy-Riemann check
    f = lambda z: z**2  # Analytic function
    cr_check = engine.cauchy_riemann_check(f, 1+1j)
    print(f"\nCauchy-Riemann check for f(z)=z²:")
    print(f"  Satisfied: {cr_check['cr_satisfied']}")
    print(f"  Error: {cr_check['cr_error']}")
    
    # Contour integration
    # ∮ 1/z dz around unit circle = 2πi
    f = lambda z: 1/z
    contour = lambda t: np.exp(2j * np.pi * t)
    integral = engine.contour_integral(f, contour)
    print(f"\nContour integral of 1/z around unit circle: {integral}")
    print(f"Expected: {2*np.pi*1j}")

if __name__ == "__main__":
    example_complex_calculations()