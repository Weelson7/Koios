import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from utils.exceptions import InvalidInputError, InvalidDimensionError, ConfigurationError

class TensorType(Enum):
    """Tensor variance types"""
    CONTRAVARIANT = "CONTRAVARIANT"
    COVARIANT = "COVARIANT"
    MIXED = "MIXED"

@dataclass
class Tensor:
    """General tensor representation"""
    components: np.ndarray
    indices: List[TensorType]  # Variance of each index
    coordinates: List[sp.Symbol]
    name: Optional[str] = None
    
    @property
    def rank(self) -> int:
        """Get tensor rank (number of indices)"""
        return len(self.indices)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        return self.components.shape

class TensorCalculusEngine:
    """Engine for tensor calculus operations"""
    
    def __init__(self):
        self._metric_tensor = None
        self.inverse_metric = None
        self.christoffel_symbols = None
        
    def metric_tensor(self, dimension: int) -> Dict[str, Any]:
        """Create a simple metric tensor of given dimension"""
        # Create identity metric (Euclidean space)
        metric = np.eye(dimension)
        coords = [sp.Symbol(f'x_{i}') for i in range(dimension)]
        
        self.set_metric(metric, coords)
        
        return {
            'success': True,
            'dimension': dimension,
            'metric': metric.tolist(),
            'coordinates': [str(c) for c in coords],
            'type': 'Euclidean'
        }
    
    def set_metric(self, metric: np.ndarray, coordinates: List[sp.Symbol]):
        """Set the metric tensor for the space"""
        self._metric_tensor = Tensor(
            components=metric,
            indices=[TensorType.COVARIANT, TensorType.COVARIANT],
            coordinates=coordinates,
            name="g"
        )
        
        # Calculate inverse metric
        if metric.dtype == object:  # Symbolic
            self.inverse_metric = Tensor(
                components=sp.Matrix(metric).inv().as_mutable(),
                indices=[TensorType.CONTRAVARIANT, TensorType.CONTRAVARIANT],
                coordinates=coordinates,
                name="g^"
            )
        else:  # Numerical
            self.inverse_metric = Tensor(
                components=np.linalg.inv(metric),
                indices=[TensorType.CONTRAVARIANT, TensorType.CONTRAVARIANT],
                coordinates=coordinates,
                name="g^"
            )
    
    def tensor_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        """Compute tensor product (outer product)"""
        # Check coordinate compatibility
        if tensor1.coordinates != tensor2.coordinates:
            raise InvalidInputError("coordinate systems", f"tensor1={tensor1.coordinates}, tensor2={tensor2.coordinates}", "same coordinate system")
        
        # Compute product
        if tensor1.components.dtype == object or tensor2.components.dtype == object:
            # Symbolic computation
            result = sp.tensorproduct(
                sp.Matrix(tensor1.components),
                sp.Matrix(tensor2.components)
            )
            components = np.array(result.tolist(), dtype=object)
        else:
            # Numerical computation
            components = np.tensordot(tensor1.components, tensor2.components, axes=0)
        
        # Combine indices
        indices = tensor1.indices + tensor2.indices
        
        return Tensor(
            components=components,
            indices=indices,
            coordinates=tensor1.coordinates,
            name=f"{tensor1.name}x{tensor2.name}" if tensor1.name and tensor2.name else None
        )
    
    def contract_indices(self, tensor: Tensor, index1: int, index2: int) -> Tensor:
        """Contract two indices of a tensor"""
        if not isinstance(index1, int):
            raise InvalidInputError("index1", str(index1), "integer")
        if not isinstance(index2, int):
            raise InvalidInputError("index2", str(index2), "integer")
        if index1 < 0 or index1 >= tensor.rank:
            raise InvalidInputError("index1", str(index1), f"value in range [0, {tensor.rank-1}]")
        if index2 < 0 or index2 >= tensor.rank:
            raise InvalidInputError("index2", str(index2), f"value in range [0, {tensor.rank-1}]")
        
        if tensor.indices[index1] == tensor.indices[index2]:
            raise InvalidInputError("indices", f"type1={tensor.indices[index1]}, type2={tensor.indices[index2]}", "different index types for contraction")
        
        # Perform contraction
        result = np.trace(tensor.components, axis1=index1, axis2=index2)
        
        # Update indices
        new_indices = [idx for i, idx in enumerate(tensor.indices) 
                      if i not in [index1, index2]]
        
        return Tensor(
            components=result,
            indices=new_indices,
            coordinates=tensor.coordinates,
            name=f"C({tensor.name})" if tensor.name else None
        )
    
    def raise_index(self, tensor: Tensor, index: int) -> Tensor:
        """Raise a covariant index using metric tensor"""
        if not self.inverse_metric:
            raise ConfigurationError("metric tensor", "not set", "metric tensor must be defined")
        
        if tensor.indices[index] != TensorType.COVARIANT:
            raise InvalidInputError("index", str(index), "covariant index to raise")
        
        # Contract with inverse metric
        components = np.tensordot(
            self.inverse_metric.components,
            tensor.components,
            axes=([1], [index])
        )
        
        # Move contracted index to correct position
        axes = list(range(len(components.shape)))
        axes.insert(index, axes.pop(0))
        components = np.transpose(components, axes)
        
        # Update index type
        new_indices = tensor.indices.copy()
        new_indices[index] = TensorType.CONTRAVARIANT
        
        return Tensor(
            components=components,
            indices=new_indices,
            coordinates=tensor.coordinates,
            name=tensor.name
        )
    
    def lower_index(self, tensor: Tensor, index: int) -> Tensor:
        """Lower a contravariant index using metric tensor"""
        if not self._metric_tensor:
            raise ConfigurationError("metric tensor", "not set", "metric tensor must be defined")
        
        if tensor.indices[index] != TensorType.CONTRAVARIANT:
            raise InvalidInputError("index", str(index), "contravariant index to lower")
        
        # Contract with metric
        components = np.tensordot(
            self._metric_tensor.components,
            tensor.components,
            axes=([1], [index])
        )
        
        # Move contracted index to correct position
        axes = list(range(len(components.shape)))
        axes.insert(index, axes.pop(0))
        components = np.transpose(components, axes)
        
        # Update index type
        new_indices = tensor.indices.copy()
        new_indices[index] = TensorType.COVARIANT
        
        return Tensor(
            components=components,
            indices=new_indices,
            coordinates=tensor.coordinates,
            name=tensor.name
        )
    
    def christoffel_symbols_first_kind(self) -> np.ndarray:
        """Calculate Christoffel symbols of the first kind Γ_ijk"""
        if not self._metric_tensor:
            raise ConfigurationError("metric tensor", "not set", "metric tensor must be defined")
        
        g = self._metric_tensor.components
        coords = self._metric_tensor.coordinates
        dim = len(coords)
        
        # Initialize Christoffel symbols
        if g.dtype == object:  # Symbolic
            gamma = np.zeros((dim, dim, dim), dtype=object)
            
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        # Γ_ijk = 1/2 (∂g_jk/∂x^i + ∂g_ik/∂x^j - ∂g_ij/∂x^k)
                        term1 = sp.diff(g[j, k], coords[i])
                        term2 = sp.diff(g[i, k], coords[j])
                        term3 = sp.diff(g[i, j], coords[k])
                        gamma[i, j, k] = sp.Rational(1, 2) * (term1 + term2 - term3)
        else:
            # Numerical approximation using finite differences
            gamma = np.zeros((dim, dim, dim))
            # Implementation would require numerical derivatives
            
        return gamma
    
    def christoffel_symbols_second_kind(self) -> np.ndarray:
        """Calculate Christoffel symbols of the second kind Γ^i_jk"""
        gamma_lower = self.christoffel_symbols_first_kind()
        g_inv = self.inverse_metric.components
        dim = len(self._metric_tensor.coordinates)
        
        if gamma_lower.dtype == object:  # Symbolic
            gamma_upper = np.zeros((dim, dim, dim), dtype=object)
            
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        # Γ^i_jk = g^il Γ_ljk
                        gamma_upper[i, j, k] = sum(
                            g_inv[i, l] * gamma_lower[l, j, k]
                            for l in range(dim)
                        )
        else:
            # Numerical computation
            gamma_upper = np.einsum('il,ljk->ijk', g_inv, gamma_lower)
        
        self.christoffel_symbols = gamma_upper
        return gamma_upper
    
    def riemann_tensor(self) -> np.ndarray:
        """Calculate Riemann curvature tensor R^i_jkl"""
        if self.christoffel_symbols is None:
            self.christoffel_symbols_second_kind()
        
        gamma = self.christoffel_symbols
        coords = self._metric_tensor.coordinates
        dim = len(coords)
        
        if gamma.dtype == object:  # Symbolic
            riemann = np.zeros((dim, dim, dim, dim), dtype=object)
            
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for l in range(dim):
                            # R^i_jkl = ∂Γ^i_jl/∂x^k - ∂Γ^i_jk/∂x^l 
                            #          + Γ^i_mk Γ^m_jl - Γ^i_ml Γ^m_jk
                            term1 = sp.diff(gamma[i, j, l], coords[k])
                            term2 = sp.diff(gamma[i, j, k], coords[l])
                            
                            term3 = sum(gamma[i, m, k] * gamma[m, j, l] 
                                      for m in range(dim))
                            term4 = sum(gamma[i, m, l] * gamma[m, j, k] 
                                      for m in range(dim))
                            
                            riemann[i, j, k, l] = term1 - term2 + term3 - term4
        else:
            # Numerical approximation
            riemann = np.zeros((dim, dim, dim, dim))
            
        return riemann
    
    def ricci_tensor(self) -> Tensor:
        """Calculate Ricci tensor R_ij = R^k_ikj"""
        riemann = self.riemann_tensor()
        dim = len(self._metric_tensor.coordinates)
        
        if riemann.dtype == object:  # Symbolic
            ricci = np.zeros((dim, dim), dtype=object)
            
            for i in range(dim):
                for j in range(dim):
                    ricci[i, j] = sum(riemann[k, i, k, j] for k in range(dim))
        else:
            # Numerical: R_ij = R^k_ikj
            ricci = np.einsum('kikj->ij', riemann)
        
        return Tensor(
            components=ricci,
            indices=[TensorType.COVARIANT, TensorType.COVARIANT],
            coordinates=self._metric_tensor.coordinates,
            name="R"
        )
    
    def ricci_scalar(self) -> Union[sp.Expr, float]:
        """Calculate Ricci scalar R = g^ij R_ij"""
        ricci = self.ricci_tensor()
        g_inv = self.inverse_metric.components
        
        if ricci.components.dtype == object:  # Symbolic
            scalar = sum(
                g_inv[i, j] * ricci.components[i, j]
                for i in range(len(self._metric_tensor.coordinates))
                for j in range(len(self._metric_tensor.coordinates))
            )
        else:
            scalar = np.einsum('ij,ij->', g_inv, ricci.components)
        
        return scalar
    
    def covariant_derivative_vector(self, vector: Tensor, direction: int) -> Tensor:
        """Calculate covariant derivative of a vector field"""
        if self.christoffel_symbols is None:
            self.christoffel_symbols_second_kind()
        
        v = vector.components
        gamma = self.christoffel_symbols
        coords = vector.coordinates
        dim = len(coords)
        
        # Check if vector is contravariant or covariant
        is_contravariant = vector.indices[0] == TensorType.CONTRAVARIANT
        
        if v.dtype == object:  # Symbolic
            nabla_v = np.zeros((dim, dim), dtype=object)
            
            for i in range(dim):
                for j in range(dim):
                    if is_contravariant:
                        # ∇_j V^i = ∂V^i/∂x^j + Γ^i_jk V^k
                        nabla_v[i, j] = sp.diff(v[i], coords[j])
                        nabla_v[i, j] += sum(gamma[i, j, k] * v[k] 
                                           for k in range(dim))
                    else:
                        # ∇_j V_i = ∂V_i/∂x^j - Γ^k_ji V_k
                        nabla_v[i, j] = sp.diff(v[i], coords[j])
                        nabla_v[i, j] -= sum(gamma[k, j, i] * v[k] 
                                           for k in range(dim))
        else:
            # Numerical approximation would go here
            nabla_v = np.zeros((dim, dim))
        
        return Tensor(
            components=nabla_v,
            indices=[vector.indices[0], TensorType.COVARIANT],
            coordinates=coords,
            name=f"∇{vector.name}" if vector.name else None
        )
    
    def lie_derivative_vector(self, vector: Tensor, flow_vector: Tensor) -> Tensor:
        """Calculate Lie derivative of a vector field along another vector field"""
        v = vector.components
        u = flow_vector.components
        coords = vector.coordinates
        dim = len(coords)
        
        if v.dtype == object:  # Symbolic
            lie_v = np.zeros(dim, dtype=object)
            
            for i in range(dim):
                # L_U V^i = U^j ∂V^i/∂x^j - V^j ∂U^i/∂x^j
                lie_v[i] = sum(u[j] * sp.diff(v[i], coords[j]) 
                             for j in range(dim))
                lie_v[i] -= sum(v[j] * sp.diff(u[i], coords[j]) 
                              for j in range(dim))
        else:
            # Numerical approximation
            lie_v = np.zeros(dim)
        
        return Tensor(
            components=lie_v,
            indices=vector.indices,
            coordinates=coords,
            name=f"L_{{{flow_vector.name}}}{vector.name}" if vector.name and flow_vector.name else None
        )
    
    def geodesic_equation(self, parameter: sp.Symbol = sp.Symbol('s')) -> List[sp.Eq]:
        """Generate geodesic equations d²x^i/ds² + Γ^i_jk (dx^j/ds)(dx^k/ds) = 0"""
        if self.christoffel_symbols is None:
            self.christoffel_symbols_second_kind()
        
        coords = self._metric_tensor.coordinates
        gamma = self.christoffel_symbols
        dim = len(coords)
        
        # Create functions for coordinates as functions of parameter
        x_funcs = [sp.Function(f'x_{i}')(parameter) for i in range(dim)]
        
        equations = []
        for i in range(dim):
            # Second derivative term
            eq = sp.diff(x_funcs[i], parameter, 2)
            
            # Christoffel symbol terms
            for j in range(dim):
                for k in range(dim):
                    if gamma[i, j, k] != 0:
                        eq += gamma[i, j, k] * sp.diff(x_funcs[j], parameter) * sp.diff(x_funcs[k], parameter)
            
            equations.append(sp.Eq(eq, 0))
        
        return equations
    
    def parallel_transport(self, vector: Tensor, curve: List[sp.Expr], 
                         parameter: sp.Symbol) -> List[sp.Eq]:
        """Generate parallel transport equations along a curve"""
        if self.christoffel_symbols is None:
            self.christoffel_symbols_second_kind()
        
        v = vector.components
        gamma = self.christoffel_symbols
        dim = len(self._metric_tensor.coordinates)
        
        # Create functions for vector components
        v_funcs = [sp.Function(f'V_{i}')(parameter) for i in range(dim)]
        
        equations = []
        for i in range(dim):
            # dV^i/dt + Γ^i_jk V^j dx^k/dt = 0
            eq = sp.diff(v_funcs[i], parameter)
            
            for j in range(dim):
                for k in range(dim):
                    if gamma[i, j, k] != 0:
                        eq += gamma[i, j, k] * v_funcs[j] * sp.diff(curve[k], parameter)
            
            equations.append(sp.Eq(eq, 0))
        
        return equations

# Example metrics for common spaces
class StandardMetrics:
    """Collection of standard metric tensors"""
    
    @staticmethod
    def euclidean_2d():
        """2D Euclidean metric in Cartesian coordinates"""
        x, y = sp.symbols('x y')
        metric = np.array([[1, 0], [0, 1]], dtype=object)
        return metric, [x, y]
    
    @staticmethod
    def euclidean_polar():
        """2D Euclidean metric in polar coordinates"""
        r, theta = sp.symbols('r theta')
        metric = np.array([[1, 0], [0, r**2]], dtype=object)
        return metric, [r, theta]
    
    @staticmethod
    def sphere_2d():
        """2-sphere metric"""
        theta, phi = sp.symbols('theta phi')
        metric = np.array([
            [1, 0],
            [0, sp.sin(theta)**2]
        ], dtype=object)
        return metric, [theta, phi]
    
    @staticmethod
    def schwarzschild():
        """Schwarzschild metric for black hole"""
        t, r, theta, phi = sp.symbols('t r theta phi', real=True, positive=True)
        M = sp.Symbol('M', positive=True)  # Mass parameter
        
        # Note: This metric is singular at r = 2M (event horizon) and r = 0
        # User should avoid evaluating at these points
        metric = np.zeros((4, 4), dtype=object)
        metric[0, 0] = -(1 - 2*M/r)
        metric[1, 1] = 1/(1 - 2*M/r)
        metric[2, 2] = r**2
        metric[3, 3] = r**2 * sp.sin(theta)**2
        
        return metric, [t, r, theta, phi]
    
    @staticmethod
    def minkowski():
        """Minkowski spacetime metric"""
        t, x, y, z = sp.symbols('t x y z')
        metric = np.diag([-1, 1, 1, 1], dtype=object)
        return metric, [t, x, y, z]