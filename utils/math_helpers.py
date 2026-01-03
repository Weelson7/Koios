import numpy as np
import sympy as sp
from typing import List, Dict, Any, Union, Optional, Tuple
import math
import cmath

class MathHelpers:
    """
    Mathematical utility functions to support the core calculation engines
    """
    
    @staticmethod
    def is_numeric(value: Any) -> bool:
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def safe_eval(expression: str, variables: Dict[str, float] = None) -> Union[float, complex, None]:
        """Safely evaluate mathematical expression"""
        try:
            if variables is None:
                variables = {}
            
            # Create safe namespace
            safe_dict = {
                '__builtins__': {},
                'abs': abs,
                'max': max,
                'min': min,
                'round': round,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'exp': math.exp,
                'log': math.log,
                'log10': math.log10,
                'pi': math.pi,
                'e': math.e,
                **variables
            }
            
            return eval(expression, safe_dict)
        except:
            return None
    
    @staticmethod
    def format_number(value: Union[float, complex], precision: int = 6) -> str:
        """Format number for display"""
        if isinstance(value, complex):
            if abs(value.imag) < 1e-10:
                return f"{value.real:.{precision}g}"
            elif abs(value.real) < 1e-10:
                return f"{value.imag:.{precision}g}i"
            else:
                sign = "+" if value.imag >= 0 else "-"
                return f"{value.real:.{precision}g}{sign}{abs(value.imag):.{precision}g}i"
        else:
            as_str = f"{value:.{precision}g}"
            return MathHelpers.format_scientific(value, precision) if 'e' in as_str or 'E' in as_str else as_str

    @staticmethod
    def format_scientific(value: float, precision: int = 6) -> str:
        """Render scientific notation as mantissa*10^exponent with no 'e' notation."""
        if value == 0:
            return "0"
        mantissa_str, exp_str = f"{value:.{precision}e}".split('e')
        # Normalize exponent: remove leading zeros and plus sign
        exp = exp_str.lstrip('+0') or "0"
        if exp.startswith('-'):
            stripped = exp.lstrip('-').lstrip('0') or '0'
            exp = f"-{stripped}"
        return f"{mantissa_str}*10^{exp}"
    
    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Convert degrees to radians"""
        return math.radians(degrees)
    
    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """Convert radians to degrees"""
        return math.degrees(radians)
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        return math.factorial(n)
    
    @staticmethod
    def combination(n: int, r: int) -> int:
        """Calculate combination C(n,r)"""
        if r > n or r < 0:
            return 0
        return math.comb(n, r)
    
    @staticmethod
    def permutation(n: int, r: int) -> int:
        """Calculate permutation P(n,r)"""
        if r > n or r < 0:
            return 0
        return math.perm(n, r)
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        return math.gcd(a, b)
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Calculate least common multiple"""
        if a == 0 or b == 0:
            return 0
        gcd_val = math.gcd(a, b)
        # Avoid overflow by dividing before multiplying
        return abs(a // gcd_val * b)
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """Get prime factors of a number"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
    
    @staticmethod
    def quadratic_formula(a: float, b: float, c: float) -> Tuple[Union[float, complex], Union[float, complex]]:
        """Solve quadratic equation ax² + bx + c = 0"""
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero. This is not a quadratic equation.")
        
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            sqrt_discriminant = math.sqrt(discriminant)
            x1 = (-b + sqrt_discriminant) / (2*a)
            x2 = (-b - sqrt_discriminant) / (2*a)
        else:
            sqrt_discriminant = cmath.sqrt(discriminant)
            x1 = (-b + sqrt_discriminant) / (2*a)
            x2 = (-b - sqrt_discriminant) / (2*a)
        
        return x1, x2
    
    @staticmethod
    def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance between two 2D points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        """Calculate distance between two 3D points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    @staticmethod
    def midpoint_2d(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Calculate midpoint between two 2D points"""
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def slope(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate slope between two points"""
        if abs(x2 - x1) < 1e-10:
            return float('inf')
        return (y2 - y1) / (x2 - x1)
    
    @staticmethod
    def angle_between_vectors(v1: List[float], v2: List[float]) -> float:
        """Calculate angle between two vectors (in radians)"""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        cos_angle = np.dot(v1_np, v2_np) / (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        return math.acos(cos_angle)
    
    @staticmethod
    def vector_magnitude(vector: List[float]) -> float:
        """Calculate magnitude of a vector"""
        return math.sqrt(sum(x**2 for x in vector))
    
    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """Normalize a vector to unit length"""
        magnitude = MathHelpers.vector_magnitude(vector)
        if abs(magnitude) < 1e-10:
            raise ValueError("Cannot normalize zero vector")
        return [x / magnitude for x in vector]
    
    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """Calculate dot product of two vectors"""
        return sum(a * b for a, b in zip(v1, v2))
    
    @staticmethod
    def cross_product_3d(v1: List[float], v2: List[float]) -> List[float]:
        """Calculate cross product of two 3D vectors"""
        if len(v1) != 3 or len(v2) != 3:
            raise ValueError("Cross product requires 3D vectors")
        
        return [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]
    
    @staticmethod
    def polynomial_eval(coefficients: List[float], x: float) -> float:
        """Evaluate polynomial at given x using Horner's method"""
        result = 0
        for coeff in coefficients:
            result = result * x + coeff
        return result
    
    @staticmethod
    def polynomial_derivative(coefficients: List[float]) -> List[float]:
        """Calculate derivative of polynomial given coefficients"""
        if len(coefficients) <= 1:
            return [0]
        
        derivative_coeffs = []
        for i, coeff in enumerate(coefficients[:-1]):
            power = len(coefficients) - 1 - i
            derivative_coeffs.append(coeff * power)
        
        return derivative_coeffs
    
    @staticmethod
    def linear_interpolation(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
        """Linear interpolation between two points"""
        if abs(x2 - x1) < 1e-10:
            return y1
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    @staticmethod
    def simpson_rule(func: callable, a: float, b: float, n: int = 1000) -> float:
        """Numerical integration using Simpson's rule"""
        if n % 2 == 1:
            n += 1  # Ensure n is even
        
        h = (b - a) / n
        x = a
        sum_vals = func(a)
        
        for i in range(1, n):
            x += h
            if i % 2 == 0:
                sum_vals += 2 * func(x)
            else:
                sum_vals += 4 * func(x)
        
        sum_vals += func(b)
        return (h / 3) * sum_vals
    
    @staticmethod
    def trapezoidal_rule(func: callable, a: float, b: float, n: int = 1000) -> float:
        """Numerical integration using trapezoidal rule"""
        h = (b - a) / n
        sum_vals = func(a) + func(b)
        
        for i in range(1, n):
            x = a + i * h
            sum_vals += 2 * func(x)
        
        return (h / 2) * sum_vals
    
    @staticmethod
    def newton_raphson(func: callable, derivative: callable, x0: float, 
                      tolerance: float = 1e-10, max_iterations: int = 100) -> Union[float, None]:
        """Find root using Newton-Raphson method"""
        x = x0
        
        for _ in range(max_iterations):
            fx = func(x)
            if abs(fx) < tolerance:
                return x
            
            dfx = derivative(x)
            if abs(dfx) < 1e-15:  # Avoid division by zero
                return None
            
            x_new = x - fx / dfx
            
            if abs(x_new - x) < tolerance:
                return x_new
            
            x = x_new
        
        return None  # Failed to converge
    
    @staticmethod
    def bisection_method(func: callable, a: float, b: float, 
                        tolerance: float = 1e-10, max_iterations: int = 100) -> Union[float, None]:
        """Find root using bisection method"""
        fa = func(a)
        fb = func(b)
        
        # Check if root is exactly at boundaries
        if abs(fa) < tolerance:
            return a
        if abs(fb) < tolerance:
            return b
        
        # Check if root exists in interval
        if fa * fb > 0:
            return None  # No root in interval
        
        for _ in range(max_iterations):
            c = (a + b) / 2
            fc = func(c)
            
            if abs(fc) < tolerance or abs(b - a) < tolerance:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return (a + b) / 2  # Return midpoint if max iterations reached
    
    @staticmethod
    def statistical_mean(data: List[float]) -> float:
        """Calculate arithmetic mean"""
        if not data:
            raise ValueError("Cannot calculate mean of empty dataset")
        return sum(data) / len(data)
    
    @staticmethod
    def statistical_median(data: List[float]) -> float:
        """Calculate median"""
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            return sorted_data[n//2]
    
    @staticmethod
    def statistical_mode(data: List[float]) -> List[float]:
        """Calculate mode(s)"""
        from collections import Counter
        
        counter = Counter(data)
        max_count = max(counter.values())
        return [value for value, count in counter.items() if count == max_count]
    
    @staticmethod
    def statistical_variance(data: List[float], population: bool = False) -> float:
        """Calculate variance"""
        if not data:
            raise ValueError("Cannot calculate variance of empty dataset")
        if len(data) == 1 and not population:
            raise ValueError("Cannot calculate sample variance with single data point. Use population=True for population variance.")
        
        mean = MathHelpers.statistical_mean(data)
        n = len(data) if population else len(data) - 1
        
        return sum((x - mean)**2 for x in data) / n
    
    @staticmethod
    def statistical_std_dev(data: List[float], population: bool = False) -> float:
        """Calculate standard deviation"""
        return math.sqrt(MathHelpers.statistical_variance(data, population))
    
    @staticmethod
    def correlation_coefficient(x_data: List[float], y_data: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            raise ValueError("Data arrays must have same length and at least 2 elements")
        
        n = len(x_data)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x * y for x, y in zip(x_data, y_data))
        sum_x2 = sum(x**2 for x in x_data)
        sum_y2 = sum(y**2 for y in y_data)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        
        if abs(denominator) < 1e-10:
            return 0  # No correlation
        
        return numerator / denominator
    
    @staticmethod
    def linear_regression(x_data: List[float], y_data: List[float]) -> Tuple[float, float]:
        """Calculate linear regression coefficients (slope, intercept)"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            raise ValueError("Data arrays must have same length and at least 2 elements")
        
        n = len(x_data)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x * y for x, y in zip(x_data, y_data))
        sum_x2 = sum(x**2 for x in x_data)
        
        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x**2
        
        if abs(denominator) < 1e-10:
            return 0, MathHelpers.statistical_mean(y_data)  # Vertical line case
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    @staticmethod
    def round_to_significant_figures(value: float, sig_figs: int) -> float:
        """Round number to specified number of significant figures"""
        if abs(value) < 1e-10:
            return 0
        
        return round(value, -int(math.floor(math.log10(abs(value)))) + (sig_figs - 1))
    
    @staticmethod
    def convert_base(number: int, from_base: int, to_base: int) -> str:
        """Convert number between different bases"""
        if from_base == 10:
            decimal = number
        else:
            # Convert from given base to decimal
            decimal = int(str(number), from_base)
        
        if to_base == 10:
            return str(decimal)
        
        # Convert from decimal to target base
        if decimal == 0:  # Integer comparison is fine here
            return "0"
        
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        
        while decimal > 0:
            result = digits[decimal % to_base] + result
            decimal //= to_base
        
        return result

# Create global instance and standalone functions for backward compatibility
math_helpers = MathHelpers()

# Standalone functions for easy import
def is_numeric(value: Any) -> bool:
    """Check if value is numeric"""
    return MathHelpers.is_numeric(value)

def safe_eval(expression: str, variables: Dict[str, float] = None) -> Union[float, complex, None]:
    """Safely evaluate mathematical expression"""
    return MathHelpers.safe_eval(expression, variables)
