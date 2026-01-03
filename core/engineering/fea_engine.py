import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ElementType(Enum):
    """Supported finite element types"""
    ROD_1D = "ROD_1D"
    BEAM_1D = "BEAM_1D"
    TRIANGLE_2D = "TRIANGLE_2D"
    QUAD_2D = "QUAD_2D"
    TETRAHEDRON_3D = "TETRAHEDRON_3D"
    HEXAHEDRON_3D = "HEXAHEDRON_3D"

class AnalysisType(Enum):
    """Types of FEA analysis"""
    STATIC_LINEAR = "STATIC_LINEAR"
    STATIC_NONLINEAR = "STATIC_NONLINEAR"
    DYNAMIC = "DYNAMIC"
    THERMAL = "THERMAL"
    MODAL = "MODAL"
    BUCKLING = "BUCKLING"

class Material:
    """Material properties for FEA"""
    def __init__(self, name: str, youngs_modulus: float, poissons_ratio: float, 
                 density: float, thermal_conductivity: Optional[float] = None,
                 thermal_expansion: Optional[float] = None, 
                 yield_strength: Optional[float] = None):
        self.name = name
        self.youngs_modulus = youngs_modulus  # Pa
        self.poissons_ratio = poissons_ratio
        self.density = density  # kg/m³
        self.thermal_conductivity = thermal_conductivity  # W/(m·K)
        self.thermal_expansion = thermal_expansion  # 1/K
        self.yield_strength = yield_strength  # Pa
    
    @classmethod
    def steel(cls):
        """Standard structural steel properties"""
        return cls(
            name="Steel",
            youngs_modulus=200e9,
            poissons_ratio=0.3,
            density=7850,
            thermal_conductivity=50,
            thermal_expansion=12e-6,
            yield_strength=250e6
        )
    
    @classmethod
    def aluminum(cls):
        """Standard aluminum properties"""
        return cls(
            name="Aluminum",
            youngs_modulus=70e9,
            poissons_ratio=0.33,
            density=2700,
            thermal_conductivity=237,
            thermal_expansion=23e-6,
            yield_strength=270e6
        )
    
    @classmethod
    def concrete(cls):
        """Standard concrete properties"""
        return cls(
            name="Concrete",
            youngs_modulus=30e9,
            poissons_ratio=0.2,
            density=2400,
            thermal_conductivity=1.7,
            thermal_expansion=10e-6,
            yield_strength=30e6
        )
    
    @classmethod
    def titanium(cls):
        """Ti-6Al-4V titanium alloy properties"""
        return cls(
            name="Titanium",
            youngs_modulus=113.8e9,
            poissons_ratio=0.342,
            density=4430,
            thermal_conductivity=6.7,
            thermal_expansion=8.6e-6,
            yield_strength=880e6
        )
    
    @classmethod
    def copper(cls):
        """Copper C101 properties"""
        return cls(
            name="Copper",
            youngs_modulus=110e9,
            poissons_ratio=0.34,
            density=8960,
            thermal_conductivity=401,
            thermal_expansion=16.5e-6,
            yield_strength=69e6
        )
    
    @classmethod
    def inconel(cls):
        """Inconel 718 superalloy properties"""
        return cls(
            name="Inconel",
            youngs_modulus=204.9e9,
            poissons_ratio=0.294,
            density=8190,
            thermal_conductivity=11.4,
            thermal_expansion=13.0e-6,
            yield_strength=1034e6
        )
    
    @classmethod
    def carbon_fiber(cls):
        """Carbon fiber T300 properties"""
        return cls(
            name="Carbon Fiber",
            youngs_modulus=230e9,
            poissons_ratio=0.20,
            density=1760,
            thermal_conductivity=7.0,
            thermal_expansion=-0.5e-6,
            yield_strength=3530e6
        )

@dataclass
class Node:
    """FEA node definition"""
    id: int
    x: float
    y: float
    z: float = 0.0
    constraints: Dict[str, bool] = None  # DOF constraints
    loads: Dict[str, float] = None  # Applied loads
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}
        if self.loads is None:
            self.loads = {}

@dataclass
class Element:
    """FEA element definition"""
    id: int
    type: ElementType
    nodes: List[int]  # Node IDs
    material: Material
    properties: Dict[str, float] = None  # Additional properties (area, thickness, etc.)
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class Mesh:
    """FEA mesh container"""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.node_count = 0
        self.element_count = 0
    
    def add_node(self, x: float, y: float, z: float = 0.0) -> int:
        """Add a node to the mesh"""
        self.node_count += 1
        node_id = self.node_count
        self.nodes[node_id] = Node(node_id, x, y, z)
        return node_id
    
    def add_element(self, element_type: ElementType, node_ids: List[int], 
                   material: Material, properties: Dict[str, float] = None) -> int:
        """Add an element to the mesh"""
        self.element_count += 1
        element_id = self.element_count
        self.elements[element_id] = Element(
            element_id, element_type, node_ids, material, properties
        )
        return element_id
    
    def apply_constraint(self, node_id: int, dof: str, constrained: bool = True):
        """Apply constraint to node DOF"""
        if node_id in self.nodes:
            self.nodes[node_id].constraints[dof] = constrained
    
    def apply_load(self, node_id: int, dof: str, value: float):
        """Apply load to node"""
        if node_id in self.nodes:
            self.nodes[node_id].loads[dof] = value

class FEAEngine:
    """Main Finite Element Analysis engine"""
    
    def __init__(self):
        self.mesh: Optional[Mesh] = None
        self.global_stiffness: Optional[sp.csr_matrix] = None
        self.global_mass: Optional[sp.csr_matrix] = None
        self.global_force: Optional[np.ndarray] = None
        self.displacement: Optional[np.ndarray] = None
        self.stress: Optional[Dict[int, np.ndarray]] = None
        self.strain: Optional[Dict[int, np.ndarray]] = None
        
    def create_mesh(self) -> Mesh:
        """Create a new mesh"""
        self.mesh = Mesh()
        return self.mesh
    
    def generate_beam_mesh(self, length: float, num_elements: int, 
                          material: Material, area: float, 
                          moment_of_inertia: float) -> Mesh:
        """Generate 1D beam mesh"""
        self.mesh = Mesh()
        
        # Create nodes
        for i in range(num_elements + 1):
            x = i * length / num_elements
            self.mesh.add_node(x, 0, 0)
        
        # Create elements
        for i in range(num_elements):
            self.mesh.add_element(
                ElementType.BEAM_1D,
                [i + 1, i + 2],
                material,
                {"area": area, "I": moment_of_inertia}
            )
        
        return self.mesh
    
    def generate_plate_mesh(self, width: float, height: float,
                           nx: int, ny: int, thickness: float,
                           material: Material) -> Mesh:
        """Generate 2D plate mesh with quad elements"""
        self.mesh = Mesh()
        
        # Create nodes
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * width / nx
                y = j * height / ny
                self.mesh.add_node(x, y, 0)
        
        # Create elements
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i + 1
                n2 = n1 + 1
                n3 = n2 + nx + 1
                n4 = n1 + nx + 1
                
                self.mesh.add_element(
                    ElementType.QUAD_2D,
                    [n1, n2, n3, n4],
                    material,
                    {"thickness": thickness}
                )
        
        return self.mesh
    
    def assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix"""
        if not self.mesh:
            raise ValueError("No mesh defined")
        
        # Determine DOFs
        num_nodes = len(self.mesh.nodes)
        dofs_per_node = 3  # u, v, w displacements
        total_dofs = num_nodes * dofs_per_node
        
        # Initialize sparse matrix
        row_ind = []
        col_ind = []
        data = []
        
        # Assemble element stiffness matrices
        for elem_id, element in self.mesh.elements.items():
            k_elem = self._compute_element_stiffness(element)
            
            # Map to global DOFs
            global_dofs = []
            for node_id in element.nodes:
                for i in range(dofs_per_node):
                    global_dofs.append((node_id - 1) * dofs_per_node + i)
            
            # Add to global matrix
            for i in range(len(global_dofs)):
                for j in range(len(global_dofs)):
                    row_ind.append(global_dofs[i])
                    col_ind.append(global_dofs[j])
                    data.append(k_elem[i, j])
        
        self.global_stiffness = sp.csr_matrix(
            (data, (row_ind, col_ind)), 
            shape=(total_dofs, total_dofs)
        )
    
    def _compute_element_stiffness(self, element: Element) -> np.ndarray:
        """Compute element stiffness matrix"""
        if element.type == ElementType.ROD_1D:
            return self._rod_stiffness(element)
        elif element.type == ElementType.BEAM_1D:
            return self._beam_stiffness(element)
        elif element.type == ElementType.QUAD_2D:
            return self._quad_stiffness(element)
        else:
            raise NotImplementedError(f"Element type {element.type} not implemented")
    
    def _rod_stiffness(self, element: Element) -> np.ndarray:
        """1D rod element stiffness matrix"""
        n1, n2 = element.nodes
        node1 = self.mesh.nodes[n1]
        node2 = self.mesh.nodes[n2]
        
        L = np.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
        E = element.material.youngs_modulus
        A = element.properties.get("area", 1.0)
        
        k = E * A / L
        return np.array([[k, -k], [-k, k]])
    
    def _beam_stiffness(self, element: Element) -> np.ndarray:
        """1D beam element stiffness matrix (Euler-Bernoulli)"""
        n1, n2 = element.nodes
        node1 = self.mesh.nodes[n1]
        node2 = self.mesh.nodes[n2]
        
        L = np.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
        E = element.material.youngs_modulus
        I = element.properties.get("I", 1.0)
        
        # 6 DOF beam element (u1, v1, θ1, u2, v2, θ2)
        k = np.zeros((6, 6))
        
        # Axial stiffness
        A = element.properties.get("area", 1.0)
        k_axial = E * A / L
        k[0, 0] = k[3, 3] = k_axial
        k[0, 3] = k[3, 0] = -k_axial
        
        # Bending stiffness (Euler-Bernoulli beam element)
        k_bend = E * I / L**3
        # Transverse displacement coupling
        k[1, 1] = k[4, 4] = 12 * k_bend
        k[1, 4] = k[4, 1] = -12 * k_bend
        # Displacement-rotation coupling
        k[1, 2] = k[2, 1] = 6 * k_bend * L
        k[1, 5] = k[5, 1] = 6 * k_bend * L
        k[2, 4] = k[4, 2] = -6 * k_bend * L
        k[4, 5] = k[5, 4] = -6 * k_bend * L
        # Rotational stiffness
        k[2, 2] = k[5, 5] = 4 * k_bend * L**2
        k[2, 5] = k[5, 2] = 2 * k_bend * L**2
        
        return k
    
    def _quad_stiffness(self, element: Element) -> np.ndarray:
        """2D quad element stiffness matrix (simplified)"""
        # Simplified implementation for demonstration
        # Full implementation would use Gauss quadrature
        E = element.material.youngs_modulus
        nu = element.material.poissons_ratio
        t = element.properties.get("thickness", 1.0)
        
        # Constitutive matrix (plane stress)
        # D[0:2,0:2] terms for normal stress, D[2,2] for shear
        G = E / (2 * (1 + nu))  # Shear modulus
        D = np.array([
            [E / (1 - nu**2), E * nu / (1 - nu**2), 0],
            [E * nu / (1 - nu**2), E / (1 - nu**2), 0],
            [0, 0, G]
        ])
        
        # Simplified 8x8 stiffness matrix for 4-node quad
        # In practice, would integrate B^T * D * B over element
        k = np.eye(8) * E * t / 100  # Placeholder
        
        return k
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions to system"""
        if self.global_stiffness is None:
            raise ValueError("Stiffness matrix not assembled")
        
        dofs_per_node = 3
        constrained_dofs = []
        
        # Identify constrained DOFs
        for node_id, node in self.mesh.nodes.items():
            for i, dof in enumerate(['u', 'v', 'w']):
                if node.constraints.get(dof, False):
                    global_dof = (node_id - 1) * dofs_per_node + i
                    constrained_dofs.append(global_dof)
        
        # Modify stiffness matrix (penalty method)
        penalty = 1e20
        K = self.global_stiffness.tolil()
        
        for dof in constrained_dofs:
            K[dof, dof] *= penalty
        
        self.global_stiffness = K.tocsr()
    
    def assemble_force_vector(self):
        """Assemble global force vector"""
        if not self.mesh:
            raise ValueError("No mesh defined")
        
        dofs_per_node = 3
        total_dofs = len(self.mesh.nodes) * dofs_per_node
        self.global_force = np.zeros(total_dofs)
        
        # Apply nodal loads
        for node_id, node in self.mesh.nodes.items():
            for i, dof in enumerate(['Fx', 'Fy', 'Fz']):
                if dof in node.loads:
                    global_dof = (node_id - 1) * dofs_per_node + i
                    self.global_force[global_dof] = node.loads[dof]
    
    def solve_static(self):
        """Solve static FEA problem"""
        if self.global_stiffness is None or self.global_force is None:
            raise ValueError("System not properly assembled")
        
        # Solve K * u = F
        self.displacement = spsolve(self.global_stiffness, self.global_force)
        
        # Calculate stress and strain
        self._calculate_stress_strain()
    
    def _calculate_stress_strain(self):
        """Calculate element stress and strain"""
        self.stress = {}
        self.strain = {}
        
        dofs_per_node = 3
        
        for elem_id, element in self.mesh.elements.items():
            # Extract element displacements
            elem_disp = []
            for node_id in element.nodes:
                for i in range(dofs_per_node):
                    global_dof = (node_id - 1) * dofs_per_node + i
                    elem_disp.append(self.displacement[global_dof])
            
            elem_disp = np.array(elem_disp)
            
            # Calculate strain (simplified)
            if element.type == ElementType.ROD_1D:
                n1, n2 = element.nodes
                node1 = self.mesh.nodes[n1]
                node2 = self.mesh.nodes[n2]
                L = np.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
                
                # Axial strain
                strain = (elem_disp[3] - elem_disp[0]) / L
                stress = element.material.youngs_modulus * strain
                
                self.strain[elem_id] = np.array([strain])
                self.stress[elem_id] = np.array([stress])
    
    def solve_modal(self, num_modes: int = 10):
        """Solve for natural frequencies and mode shapes"""
        from scipy.sparse.linalg import eigsh
        
        if self.global_stiffness is None or self.global_mass is None:
            raise ValueError("Stiffness and mass matrices required")
        
        # Solve generalized eigenvalue problem: K*phi = omega^2*M*phi
        eigenvalues, eigenvectors = eigsh(
            self.global_stiffness, M=self.global_mass, 
            k=num_modes, which='SM'
        )
        
        # Natural frequencies (Hz)
        frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
        
        return frequencies, eigenvectors
    
    def visualize_mesh(self, show_loads: bool = True, show_constraints: bool = True):
        """Visualize FEA mesh"""
        if not self.mesh:
            raise ValueError("No mesh defined")
        
        fig = plt.figure(figsize=(10, 8))
        
        # Determine if 2D or 3D visualization needed
        is_3d = any(node.z != 0 for node in self.mesh.nodes.values())
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        
        # Plot elements
        for element in self.mesh.elements.values():
            if element.type in [ElementType.ROD_1D, ElementType.BEAM_1D]:
                nodes = [self.mesh.nodes[n] for n in element.nodes]
                x = [n.x for n in nodes]
                y = [n.y for n in nodes]
                z = [n.z for n in nodes] if is_3d else None
                
                if is_3d:
                    ax.plot(x, y, z, 'b-', linewidth=2)
                else:
                    ax.plot(x, y, 'b-', linewidth=2)
            
            elif element.type == ElementType.QUAD_2D:
                nodes = [self.mesh.nodes[n] for n in element.nodes]
                nodes.append(nodes[0])  # Close the quad
                x = [n.x for n in nodes]
                y = [n.y for n in nodes]
                
                ax.plot(x, y, 'b-', linewidth=1)
                ax.fill(x, y, alpha=0.3, color='lightblue')
        
        # Plot nodes
        for node in self.mesh.nodes.values():
            if is_3d:
                ax.scatter(node.x, node.y, node.z, c='red', s=50)
            else:
                ax.scatter(node.x, node.y, c='red', s=50)
            
            # Show constraints
            if show_constraints and node.constraints:
                marker = 'v' if 'u' in node.constraints else ''
                marker += '<' if 'v' in node.constraints else ''
                marker += '^' if 'w' in node.constraints else ''
                
                if marker and is_3d:
                    ax.text(node.x, node.y, node.z, marker, fontsize=12)
                elif marker:
                    ax.text(node.x, node.y, marker, fontsize=12)
            
            # Show loads
            if show_loads and node.loads:
                for dof, value in node.loads.items():
                    if dof == 'Fx' and value != 0:
                        dx = 0.1 * np.sign(value)
                        if is_3d:
                            ax.arrow(node.x, node.y, node.z, dx, 0, 0)
                        else:
                            ax.arrow(node.x, node.y, dx, 0, 
                                   head_width=0.05, head_length=0.02, 
                                   fc='green', ec='green')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if is_3d:
            ax.set_zlabel('Z')
        
        ax.set_title('FEA Mesh Visualization')
        ax.grid(True)
        
        return fig
    
    def visualize_results(self, result_type: str = 'displacement', 
                         scale_factor: float = 1.0):
        """Visualize FEA results"""
        if self.displacement is None:
            raise ValueError("No results to visualize")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Create deformed shape
        dofs_per_node = 3
        
        for element in self.mesh.elements.values():
            nodes_original = []
            nodes_deformed = []
            
            for node_id in element.nodes:
                node = self.mesh.nodes[node_id]
                nodes_original.append([node.x, node.y])
                
                # Get displacements
                u = self.displacement[(node_id - 1) * dofs_per_node]
                v = self.displacement[(node_id - 1) * dofs_per_node + 1]
                
                nodes_deformed.append([
                    node.x + scale_factor * u,
                    node.y + scale_factor * v
                ])
            
            # Plot original shape
            nodes_orig = np.array(nodes_original)
            nodes_def = np.array(nodes_deformed)
            
            if element.type == ElementType.QUAD_2D:
                # Close the shape
                nodes_orig = np.vstack([nodes_orig, nodes_orig[0]])
                nodes_def = np.vstack([nodes_def, nodes_def[0]])
            
            ax.plot(nodes_orig[:, 0], nodes_orig[:, 1], 'b--', 
                   alpha=0.5, label='Original' if element.id == 1 else "")
            ax.plot(nodes_def[:, 0], nodes_def[:, 1], 'r-', 
                   linewidth=2, label='Deformed' if element.id == 1 else "")
            
            # Color by stress/strain if available
            if result_type == 'stress' and self.stress:
                if element.id in self.stress:
                    stress_val = np.max(np.abs(self.stress[element.id]))
                    color_intensity = min(stress_val / 100e6, 1.0)  # Normalize
                    ax.fill(nodes_def[:, 0], nodes_def[:, 1], 
                           color=(color_intensity, 0, 1 - color_intensity), 
                           alpha=0.5)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'FEA Results - {result_type.capitalize()}')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        return fig

# Example usage functions
def cantilever_beam_example():
    """Example: Cantilever beam analysis"""
    fea = FEAEngine()
    
    # Create beam mesh
    length = 1.0  # 1 meter
    height = 0.1  # 100mm
    width = 0.05  # 50mm
    
    mesh = fea.generate_beam_mesh(
        length=length,
        num_elements=10,
        material=Material.steel(),
        area=height * width,
        moment_of_inertia=width * height**3 / 12
    )
    
    # Apply boundary conditions (fixed at x=0)
    mesh.apply_constraint(1, 'u', True)
    mesh.apply_constraint(1, 'v', True)
    mesh.apply_constraint(1, 'w', True)
    
    # Apply load at free end
    mesh.apply_load(11, 'Fy', -1000)  # 1kN downward
    
    # Solve
    fea.assemble_stiffness_matrix()
    fea.apply_boundary_conditions()
    fea.assemble_force_vector()
    fea.solve_static()
    
    return fea

def plate_with_hole_example():
    """Example: Plate with central hole under tension"""
    fea = FEAEngine()
    
    # Create plate mesh
    mesh = fea.generate_plate_mesh(
        width=2.0,
        height=1.0,
        nx=20,
        ny=10,
        thickness=0.01,
        material=Material.aluminum()
    )
    
    # Apply constraints (fixed left edge)
    for j in range(11):
        node_id = j * 21 + 1
        mesh.apply_constraint(node_id, 'u', True)
        mesh.apply_constraint(node_id, 'v', True)
    
    # Apply tension on right edge
    for j in range(11):
        node_id = j * 21 + 21
        mesh.apply_load(node_id, 'Fx', 1000)  # Distributed load
    
    # Solve
    fea.assemble_stiffness_matrix()
    fea.apply_boundary_conditions()
    fea.assemble_force_vector()
    fea.solve_static()
    
    return fea