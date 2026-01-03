# Koïos Architecture Documentation

```
Koïos/
├── app.py
├── run_koios.py
├── styles.css
├── Koïos_Logo.png
├── core/
│   ├── __init__.py
│   ├── advanced_integration_engine.py
│   ├── advanced_ode_solver.py
│   ├── calculation_engine.py
│   ├── calculus_engine.py
│   ├── complex_analysis_engine.py
│   ├── expression_parser.py
│   ├── matrix_operations.py
│   ├── numerical_methods_engine.py
│   ├── ode_solver.py
│   ├── optimization_algorithms_engine.py
│   ├── physics_simulator.py
│   ├── statistical_mechanics_engine.py
│   ├── tensor_calculus_engine.py
│   ├── transforms_series_engine.py
│   └── engineering/
│       ├── __init__.py
│       ├── advanced_material_science.py
│       ├── cfd_engine.py
│       ├── electromagnetics_engine.py
│       ├── fea_engine.py
│       └── material_science_engine.py
├── templates/
│   ├── __init__.py
│   ├── arithmetic_operations.py
│   ├── calculus_operations.py
│   ├── engineering_finite_element_analysis.py
│   ├── engineering_fluid_dynamics.py
│   ├── engineering_signal_processing.py
│   ├── engineering_thermodynamics.py
│   ├── matrix_operations.py
│   ├── physics_simulations.py
│   └── trigonometric_functions.py
├── ui/
│   ├── __init__.py
│   ├── calculator_panel.py
│   ├── calculus_panel.py
│   ├── complex_analysis_panel.py
│   ├── engineering_panel.py
│   ├── equation_solver_panel.py
│   ├── matrix_panel.py
│   ├── numerical_methods_panel.py
│   ├── optimization_panel.py
│   ├── physics_panel.py
│   ├── tensor_calculus_panel.py
│   └── visualization_panel.py
└── utils/
    ├── __init__.py
    ├── exceptions.py
    ├── math_helpers.py
    ├── result_builder.py
    ├── ui_helpers.py
    └── validators.py
```

## File Roles and API Documentation

---

### Root Level Files

#### app.py
**Role:** Main Streamlit application entry point and orchestrator

**API Endpoints/Functions:**
- `main()` - Primary application entry point
  - Configures Streamlit page settings
  - Loads custom CSS styling
  - Renders navigation sidebar with tool categories
  - Routes to appropriate UI panel based on user selection
  - Handles module import errors gracefully

**Key Features:**
- Dynamic tool navigation system with categories:
  - Basic Tools (Calculator, Matrix)
  - Advanced Mathematics (Calculus, Equations, Complex Analysis, Tensor, Numerical, Optimization)
  - Physics & Engineering (Physics Simulations, Engineering Simulations)
  - Visualization (Function Plotting)
- Error handling and logging
- Logo display and branding

---

#### run_koios.py
**Role:** Application launcher script

**API Endpoints/Functions:**
- `main()` - Launches the Koios application
  - Checks for Streamlit installation
  - Executes Streamlit server with app.py
  - Configures server on localhost:8501

**Key Features:**
- Dependency verification
- Subprocess management for Streamlit
- Graceful shutdown handling

---

#### styles.css
**Role:** Custom CSS styling for the application UI

**Key Features:**
- Modern gradient-based header styling
- Enhanced card and tab styling
- Input field enhancements with focus states
- Selectbox custom styling
- Responsive design elements
- Dark theme optimizations

---

### Core Module (`core/`)

#### calculation_engine.py
**Role:** Core mathematical expression evaluation and manipulation engine

**API Endpoints/Functions:**
- `evaluate(expression, variables)` - Simple expression evaluation
- `evaluate_expression(expression, variables)` - Comprehensive expression evaluation
  - Returns: `{'success': bool, 'symbolic_result': Expr, 'numeric_result': float, 'error': str}`
- `simplify_expression(expression)` - Algebraic simplification
  - Returns: `{'success': bool, 'original': Expr, 'simplified': Expr, 'error': str}`
- `expand_expression(expression)` - Algebraic expansion
  - Returns: `{'success': bool, 'original': Expr, 'expanded': Expr, 'error': str}`

**Key Features:**
- LRU caching for repeated expressions
- Supports symbolic and numeric evaluation with consistent result dictionaries
- Variable substitution
- SymPy integration
- Control systems step response uses inverse Laplace transform for unit step inputs
- Precision control (default: 15 digits)

---

#### calculus_engine.py
**Role:** Comprehensive calculus operations engine

**API Endpoints/Functions:**
- `compute_derivative(expression, variable, order)` - Calculate derivatives
  - Returns: `{'success': bool, 'derivative': Expr, 'simplified_derivative': Expr, 'error': str}`
- `compute_integral(expression, variable, definite, lower_limit, upper_limit)` - Calculate integrals
  - Returns: `{'success': bool, 'integral': Expr, 'simplified_integral': Expr, 'numeric_value': float, 'error': str}`
- `compute_limit(expression, variable, limit_point, direction)` - Calculate limits
  - Returns: `{'success': bool, 'limit': Expr, 'error': str}`
- `differentiate(expression, variable, order)` - Simple derivative wrapper
- `integrate(expression, variable, lower, upper)` - Integral wrapper with proper parameter mapping
  - Automatically determines definite vs indefinite based on limits
- `limit(expression, variable, point)` - Simple limit wrapper
- `compute_complex_analysis(expression, variable)` - Complex analysis with specific exception handling

**Key Features:**
- Multi-order derivatives
- Definite and indefinite integrals with proper wrapper methods
- Directional limits (left, right, bidirectional)
- Automatic simplification
- Numerical evaluation for definite integrals
- Improved error handling with specific exception types
- Residue computation handles higher-order poles using the general formula
 - Fractional calculus now returns Riemann-Liouville derivative/integral and Caputo derivative; harmonic analysis provides truncated Fourier series and Legendre projections

---

#### expression_parser.py
**Role:** Mathematical expression parsing with SymPy

**API Endpoints/Functions:**
- `parse(expression)` - Parse string expression to SymPy expression
- `parse_expression(expression)` - Alias for parse method
- `_preprocess_expression(expression)` - Internal preprocessing with European scientific notation support (1e-3 -> 1*10**(-3))

**Key Features:**
- Standard mathematical transformations with proper scientific notation handling
- Implicit multiplication support (2x -> 2*x)
- Predefined mathematical symbols (x, y, z, t, theta, etc.)
- Safety guardrails: length limits and blacklist validation prior to parsing
- Common mathematical constants (pi, e, i, infinity)
- Built-in function support (sin, cos, exp, log, sqrt, etc.)
- Power notation conversion (^ to **)
- Validates multivariate polynomial expressions correctly

---

#### matrix_operations.py
**Role:** Comprehensive matrix operations using NumPy and SymPy

**API Endpoints/Functions:**
- `create_matrix(data, symbolic)` - Create numeric or symbolic matrix
  - Returns: `{'success': bool, 'matrix': ndarray/Matrix, 'shape': tuple, 'type': str, 'error': str}`
- `matrix_addition(matrix1, matrix2)` - Add two matrices
- `matrix_multiplication(matrix1, matrix2)` - Multiply two matrices with type validation
  - Validates both matrices are same type (NumPy or SymPy)
- `matrix_determinant(matrix)` - Calculate determinant
- `matrix_inverse(matrix)` - Calculate inverse
  - Now raises SingularMatrixError for singular matrices instead of generic ValueError
- `matrix_transpose(matrix)` - Transpose matrix
- `matrix_eigenvalues(matrix)` - Compute eigenvalues
- `matrix_eigenvectors(matrix)` - Compute eigenvectors
- `matrix_rank(matrix)` - Calculate matrix rank
- `matrix_norm(matrix, norm_type)` - Calculate matrix norm with corrected SymPy iteration
- `solve_linear_system(A, b)` - Solve Ax=b
- `qr_decomposition(matrix)` - QR decomposition
- `svd_decomposition(matrix)` - Singular value decomposition
- `generate_random_matrix(rows, cols, min_val, max_val)` - Generate random matrix with validated bounds

**Key Features:**
- Dual support for numeric (NumPy) and symbolic (SymPy) matrices
- Type checking for matrix operations
- Comprehensive linear algebra operations
- Matrix decompositions
- Validated parameter ranges for random generation
- Error handling for singular matrices with machine epsilon threshold
- Improved exception propagation in wrapper methods (transpose, rank)
- Fixed SymPy matrix norm calculation to properly iterate over all elements
- Uses custom exceptions (SingularMatrixError, InvalidDimensionError) for better error handling

---

#### complex_analysis_engine.py
**Role:** Complex number calculations and analysis

**API Endpoints/Functions:**
- `parse_complex(expr)` - Parse string to complex number
  - Handles both standard and SymPy notation (i, j, I)
  - Raises specific ValueError/TypeError instead of bare exceptions
- `complex_arithmetic(z1, z2, operation)` - Perform arithmetic operations
  - Returns: All operations or specific operation result
  - Uses tolerance-based division by zero checking: `abs(z2) < tol`
- `polar_form(z)` - Convert to polar representation
  - Returns: `{'r': float, 'theta': float, 'theta_degrees': float, 'polar_form': str}`
- `exponential_form(z)` - Express in exponential form
- `complex_roots(z, n)` - Find all nth roots
- `complex_logarithm(z, branch)` - Calculate complex log with branch selection
- `complex_power(z, w)` - Calculate z^w for complex numbers
- `analytic_functions(z)` - Evaluate standard analytic functions (exp, sin, cos, log, etc.)
- `derivative(f, z, h)` - Numerical complex derivative
- `cauchy_riemann_check(f, z, h)` - Verify Cauchy-Riemann equations
- `contour_integral(f, contour, t_range, n_points)` - Numerical contour integration
- `residue(f, pole, order, radius)` - Calculate residue at pole
  - Fixed: Uses proper numerical limit evaluation via h_values → 0
- `residue_theorem(f, poles, contour)` - Apply residue theorem
  - Warning: Assumes all poles inside contour (TODO: implement winding number check)
- `conformal_map(w_func, z_grid)` - Apply conformal mapping
  - Optimized: Uses np.vectorize for efficiency (eliminates nested loops)
- `schwarz_christoffel(vertices, angles)` - Schwarz-Christoffel mapping
  - Fixed: Raises NotImplementedError instead of returning misleading identity function
- `complex_ode_solve(ode_func, z0, t_span, n_points)` - Solve complex ODE
  - Fixed: Returns Dict{'t', 'z', 'success'} instead of tuple
  - API consistent with other engine methods
- `laurent_series(f, center, order)` - Compute Laurent series coefficients (DEPRECATED)
  - Now delegates to laurent_series_coefficients() for consistency
  - Use laurent_series_coefficients() directly for production code
- `laurent_series_coefficients(f, center, inner_radius, outer_radius, max_order)` - Primary Laurent series implementation
  - Consolidated implementation with two-contour approach
  - Separates Taylor part (positive powers) and principal part (negative powers)
  - Input validation: inner_radius must be < outer_radius
  - Improved documentation with examples and parameters explanation
- `fft_analysis(signal, sampling_rate)` - FFT for signal processing
- `transfer_function_analysis(num, den)` - Transfer function analysis
- `electromagnetic_impedance(frequency, resistance, inductance, capacitance)` - RLC impedance

**Key Features:**
- Multiple complex representations
- Branch cut handling
- Analytic function evaluation
- Complex calculus operations
- Tolerance-based numerical stability checks (self.tol = 1e-10)
- Proper error handling with specific exceptions
- Vectorized operations for performance
- Consistent API across all methods

---

#### physics_simulator.py
**Role:** Extensible physics simulation framework

**API Endpoints/Functions:**
- `run_simulation(simulation_name, parameters)` - Execute physics simulation
- `simulate(simulation_name, parameters)` - Alias for run_simulation
- `get_available_simulations()` - List all available simulations
- `register_simulation(name, simulation_func)` - Add custom simulation
- `projectile_motion(parameters)` - Ballistic trajectory simulation
  - Returns: `{'success': bool, 'time': list, 'x_position': list, 'y_position': list, ..., 'actual_points_generated': int}`
  - Note: Output may contain fewer points than requested if trajectory ends before time_max

**Available Simulations:**
- `projectile_motion` - Ballistic trajectory simulation
- `simple_harmonic_motion` - SHM oscillations
- `damped_oscillator` - Damped harmonic motion
- `pendulum` - Pendulum dynamics
- `circuit_rc`, `circuit_rl`, `circuit_rlc` - Electrical circuits
- `electromagnetic_wave` - EM wave propagation
- `doppler_effect` - Doppler frequency shift
- `wave_interference` - Wave superposition
- `heat_conduction` - Thermal diffusion
- `orbital_motion` - Two-body orbital mechanics
- `three_body_orbital`, `four_body_orbital` - N-body problems
- `fluid_flow` - Basic fluid dynamics
- `electromagnetic_field` - EM field calculations
- `photoelectric_effect` - Quantum photoelectric effect
- `relativity_time_dilation` - Special relativity
- `nuclear_decay` - Radioactive decay
- `particle_accelerator` - Particle acceleration
- `optics_lenses` - Geometrical optics

**Key Features:**
- Modular simulation architecture
- Time-series data output
- Physical constants included
- Extensible plugin system
- Comprehensive parameter validation with NaN/Inf checks and physical constraints
- Grid resolution validation for heat_conduction, computational_fluid_dynamics, and electromagnetic_field (minimum 2 points/dimensions, warning < 10)
- Physics-accurate implementations for damped oscillators (underdamped, critically damped, overdamped)
- Large angle warnings for pendulum simulations
- Improved projectile motion output tracking with actual_points_generated field for transparency

---

#### numerical_methods_engine.py
**Role:** Advanced numerical methods and algorithms

**API Endpoints/Functions:**
- `newton_method(func_str, x0, max_iter)` - Newton-Raphson root finding
  - Returns: `{'success': bool, 'root': float, 'iterations': int, 'convergence_history': list, 'final_residual': float, 'error': str}`
- `spectral_poisson_solver_2d(f, domain, boundary_conditions)` - 2D Poisson equation solver
  - Fixed: Added comprehensive documentation for zero-frequency mode handling
  - Explains compatibility condition for periodic BCs
- `fft_operations()` - Fast Fourier Transform methods
- `monte_carlo_integration()` - Monte Carlo numerical integration
  - Optimized: Attempts vectorized function evaluation first, falls back to element-wise
  - Supports uniform, quasi-random (Sobol), and importance sampling methods
- `adaptive_quadrature()` - Adaptive integration
- `sparse_linear_solver()` - Iterative sparse system solvers

**Key Features:**
- Root finding algorithms
- Numerical integration with vectorization support
- FFT and spectral methods
- Sparse matrix solvers
- Monte Carlo methods with multiple sampling strategies
- Convergence tracking

---

#### optimization_algorithms_engine.py
**Role:** Advanced optimization algorithms

**API Endpoints/Functions:**
- `lbfgs_b(f, grad_f, x0, bounds, m)` - Limited-memory BFGS with bounds
  - Returns: `OptimizationResult(x_opt, f_opt, iterations, converged, history)`
- `gradient_descent()` - Basic gradient descent
- `conjugate_gradient()` - Conjugate gradient method
- `nelder_mead()` - Nelder-Mead simplex
- `particle_swarm()` - Particle swarm optimization
- `genetic_algorithm()` - Genetic algorithm
- `simulated_annealing()` - Simulated annealing

**Optimization Types:**
- Unconstrained optimization
- Constrained optimization
- Multi-objective optimization
- Robust optimization
- Stochastic optimization

**Key Features:**
- Multiple optimization algorithms
- Constraint handling
- Pareto front calculation
- Convergence history tracking

---

#### tensor_calculus_engine.py
**Role:** Tensor calculus operations for differential geometry

**API Endpoints/Functions:**
- `metric_tensor(dimension)` - Create metric tensor
  - Returns: `{'success': bool, 'dimension': int, 'metric': list, 'coordinates': list, 'type': str}`
- `set_metric(metric, coordinates)` - Define metric tensor
- `tensor_product(tensor1, tensor2)` - Compute tensor product
- `contract_indices(tensor, index1, index2)` - Index contraction
- `raise_index(tensor, index)` - Raise covariant index
- `lower_index(tensor, index)` - Lower contravariant index
- `christoffel_symbols()` - Calculate Christoffel symbols
- `riemann_curvature()` - Compute Riemann curvature tensor
- `ricci_tensor()` - Calculate Ricci tensor
- `ricci_scalar()` - Compute Ricci scalar

**Key Features:**
- General coordinate systems
- Covariant and contravariant tensors
- Metric tensor operations
- Curvature calculations
- Geodesic equations

---

#### ode_solver.py
**Role:** Ordinary differential equation solver

**API Endpoints/Functions:**
- `parse_ode(ode_string, dependent_var, independent_var)` - Parse ODE from string
  - Returns: `{'success': bool, 'ode_expr': Expr, 'order': int, 'error': str}`
  - Now validates variable names to prevent conflicts and ensures valid identifiers
- `solve_symbolic_ode(ode_string, dependent_var, independent_var)` - Symbolic ODE solution
  - Returns: `{'success': bool, 'ode_expr': Expr, 'solution': Expr, 'error': str}`
- `euler_method(func, y0, x_span, num_points)` - Euler's method numerical solution
- `rk4_method(func, y0, x_span, num_points)` - Runge-Kutta 4th order
- `solve_ivp_wrapper()` - Wrapper for scipy.integrate.solve_ivp
  - Now returns consistent format: 1D array for single variable, 2D array for multiple variables
- `solve_system_odes(expressions, initial_conditions, x_span, variables, method)` - System of ODEs
  - Now validates input consistency (expressions count == variables count == initial conditions count)
  - Pre-parses all expressions once before solving (optimization)
  - Uses custom exceptions for better error handling

**Key Features:**
- Symbolic and numerical solutions
- Multiple numerical methods
- Initial value problems
- Order detection with improved regex parsing to prevent double replacement of derivative notations
- Derivative order calculation using SymPy's built-in `derivative_count` property for better reliability
- Improved variable validation to prevent name conflicts
- Optimized expression parsing for system of ODEs
- Consistent return value formatting across methods
- Custom exception usage (InvalidInputError, ConfigurationError, ExpressionParseError)

---

#### advanced_ode_solver.py
**Role:** Advanced ODE solving with multiple methods

**API Endpoints/Functions:**
- `solve_ode(problem, method, num_points, adaptive)` - General ODE solver
  - Returns: `ODESolution(t, y, method, success, message, error_estimate, stability_info)`
- `solve_system(equations, initial_conditions, t_span, method)` - System of ODEs
- `solve_bvp(equation, boundary_conditions, x_span, y_guess)` - Boundary value problems

**Available Methods:**
- Euler method
- Improved Euler (Heun's method)
- RK4 (4th order Runge-Kutta)
- RK45 (adaptive Runge-Kutta)
- Adams-Bashforth (multistep)
- Adams-Moulton (implicit multistep)
- BDF (Backward Differentiation Formula)
- Radau (implicit Runge-Kutta)
- DOP853 (8th order explicit)

**ODE Types Supported:**
- First-order ODEs
- Second-order ODEs
- Higher-order ODEs
- Systems of ODEs
- Partial differential equations
- Delay differential equations
- Stochastic differential equations

---

#### advanced_integration_engine.py
**Role:** Advanced integration techniques

**API Endpoints/Functions:**
- `integrate_expression(expression, variable, method)` - Advanced integration
  - Returns: `IntegrationResult(success, integral, method_used, steps, error)`

**Integration Methods:**
- Direct integration
- U-substitution
- Integration by parts
- Partial fractions
- Trigonometric substitution
- Special functions

**Key Features:**
- Step-by-step integration process
- Method selection
- Integration patterns database
- Fallback to symbolic integral

---

#### transforms_series_engine.py
**Role:** Comprehensive transforms and series analysis engine

**API Endpoints/Functions:**
- `laplace_transform(expression, t_var, s_var)` - Compute Laplace transform
  - Automatically substitutes time variable if single free symbol detected
  - Returns: `{'success': bool, 'original_expression': Expr, 'transform': Expr, 'error': str}`
- `inverse_laplace_transform(expression, s_var, t_var)` - Compute inverse Laplace transform
  - Safe variable substitution (preserves constants like I, E, pi)
  - Returns: `{'success': bool, 'original_expression': Expr, 'transform': Expr, 'error': str}`
- `fourier_transform(expression, t_var, omega_var)` - Compute Fourier transform
  - Automatically substitutes time variable if single free symbol detected
  - Returns: `{'success': bool, 'original_expression': Expr, 'transform': Expr, 'error': str}`
- `inverse_fourier_transform(expression, omega_var, t_var)` - Compute inverse Fourier transform
  - Safe variable substitution (preserves constants)
  - Returns: `{'success': bool, 'original_expression': Expr, 'transform': Expr, 'error': str}`
- `transfer_function_from_de(differential_equation, input_var, output_var, t_var, s_var)` - Extract transfer function from DE
  - Parses differential equation and extracts coefficients for y(t) and u(t) derivatives
  - Builds numerator B(s) and denominator A(s) polynomials
  - Computes H(s) = B(s)/A(s) with poles and zeros
  - Returns: `{'success': bool, 'differential_equation': Expr, 'transfer_function': Expr, 'poles': list, 'zeros': list, 'denominator_poly': Expr, 'numerator_poly': Expr, 'error': str}`
- `solve_de_with_laplace(differential_equation, initial_conditions, input_function, t_var, s_var)` - Solve DE using Laplace
  - Uses SymPy dsolve for ODE solving
  - **Zero-Input Response**: Solves with input=0 using given initial conditions
  - **Zero-State Response**: Solves with ICs=0 using given input function
  - **Initial Condition Handling**: Uses all available IC equations up to number of constants (no gating on derivative order)
  - **Transient/Steady Detection**: Analyzes poles via exponential real parts (Re{λ}<0 → decaying) and limits as t→∞
  - Returns: `{'success': bool, 'solution': Expr, 'transient': Expr, 'steady_state': Expr, 'zero_input': Expr, 'zero_state': Expr, 'error': str}`
- `solve_de_with_fourier(differential_equation, boundary_conditions, x_var, omega_var)` - Solve DE using Fourier
  - **Status**: Not implemented - returns error message
  - Intended for spatial domain problems with boundary conditions
  - Returns: `{'success': False, 'error': 'Fourier transform DE solving not yet fully implemented'}`
- `compute_fourier_series(expression, variable, period, n_terms)` - Compute Fourier series expansion
  - Computes trigonometric form (a0, an, bn coefficients)
  - Generates complex exponential form
  - Returns: `{'success': bool, 'original_expression': Expr, 'fourier_series': Expr, 'a0': Expr, 'an_coefficients': dict, 'bn_coefficients': dict, 'complex_form': Expr, 'error': str}`
- `decompose_solution(solution, t_var)` - Decompose into transient and steady-state
  - Uses pole analysis and limit behavior for classification
  - Returns: `{'success': bool, 'original_solution': Expr, 'transient': Expr, 'steady_state': Expr, 'error': str}`
- `compute_zero_input_zero_state(transfer_function, input_function, initial_conditions, s_var, t_var)` - Zero-input/zero-state decomposition
  - Standalone function for zero-input/zero-state analysis
  - Returns: `{'success': bool, 'zero_input': Expr, 'zero_state': Expr, 'total_response': Expr, 'error': str}`
- `analyze_frequency_response(transfer_function, s_var)` - Analyze frequency response
  - Computes magnitude |H(jω)| and phase ∠H(jω)
  - Extracts poles and zeros
  - Determines stability (all poles Re{λ}<0)
  - Returns: `{'success': bool, 'transfer_function': Expr, 'magnitude': Expr, 'phase': Expr, 'poles': list, 'zeros': list, 'stable': bool, 'error': str}`

**Transform Types:**
- Laplace transforms (forward and inverse) with safe variable substitution
- Fourier transforms (forward and inverse) with safe variable substitution
- Transfer function analysis via coefficient extraction from DEs
- Frequency response analysis with stability checking

**Differential Equation Solving:**
- **Laplace Transform Method**: 
  - Full solution with ICs
  - Zero-input response (input=0, uses ICs)
  - Zero-state response (ICs=0, uses input)
  - Transient/steady decomposition via pole analysis
  - Supports up to 2nd order ICs: y(0), y'(0), y''(0)
- **Fourier Transform Method**: Not yet implemented (returns error)

**Solution Decomposition:**
- **Transient Response**: Terms with decaying exponentials (Re{λ}<0) or approaching zero as t→∞
- **Steady-State Response**: Terms persisting as t→∞ (constants, bounded oscillations)
- **Zero-Input Response**: Natural response from initial conditions only (input=0)
- **Zero-State Response**: Forced response from input only (ICs=0)
- **Pole-Based Analysis**: Uses exponential coefficient real parts for transient classification

**Fourier Series:**
- Trigonometric form (sine and cosine)
- Complex exponential form
- Coefficient calculation (a0, an, bn)
- Periodic function expansion with arbitrary period

**Key Features:**
- Symbolic transform computation using SymPy
- Transfer function extraction via derivative coefficient parsing
- Pole-zero analysis and stability checking
- Zero-input/zero-state decomposition in solve_de_with_laplace
- Transient/steady detection via pole real parts and limit behavior
- Relaxed IC handling (uses all available ICs without derivative gating)
- Safe variable substitution (preserves mathematical constants)
- Fourier series with arbitrary period and term count
- Visualization support for series convergence

**Implementation Notes:**
- Variable substitution limited to single free symbol expressions to avoid parameter corruption
- Transfer function extraction handles general linear ODEs with y(t) and u(t) terms
- Pole stability checking: system stable if all poles have Re{λ}<0
- Fourier DE solving disabled in UI until implementation complete

---

#### statistical_mechanics_engine.py
**Role:** Statistical mechanics calculations and simulations

**API Endpoints/Functions:**
- `calculate_boltzmann_distribution(energies, temperature)` - Boltzmann distribution
  - Returns: `StatMechResult(success, method, results, error)`
- `canonical_partition_function(energy_function, states, temperature)` - Partition function
- `monte_carlo_simulation(energy_function, initial_state, temperature, n_steps)` - MC simulation
- `ising_model()` - Ising model simulation
- `molecular_dynamics()` - MD simulation

**Methods:**
- Boltzmann statistics
- Canonical ensemble
- Microcanonical ensemble
- Grand canonical ensemble
- Monte Carlo methods
- Molecular dynamics
- Metropolis algorithm
- Ising model

**Key Features:**
- Physical constants (Boltzmann, Avogadro, Planck)
- Thermodynamic property calculations
- Statistical ensemble simulations

---

### Engineering Submodule (`core/engineering/`)

#### fea_engine.py
**Role:** Finite Element Analysis engine

**Classes:**
- `Material` - Material property definitions with presets (Steel, Aluminum, Concrete, Titanium, Copper, Inconel, Carbon Fiber)
- `Node` - FEA node with coordinates, constraints, and loads
- `Element` - Finite element definition
- `FEAEngine` - Main FEA solver

**Element Types:**
- ROD_1D - 1D truss elements
- BEAM_1D - 1D beam elements
- TRIANGLE_2D - 2D triangular elements
- QUAD_2D - 2D quadrilateral elements
- TETRAHEDRON_3D - 3D tetrahedral elements
- HEXAHEDRON_3D - 3D hexahedral elements

**Analysis Types:**
- Static linear analysis
- Static nonlinear analysis
- Dynamic analysis
- Thermal analysis
- Modal analysis
- Buckling analysis

**Key Features:**
- Assembly of global stiffness matrix
- Boundary condition application
- Stress and strain calculations
- Displacement solutions
- Poisson's ratio validation (-1 < nu < 1) to prevent division by zero in constitutive matrix
- Warnings for Poisson's ratios outside typical material range [0, 0.5]

---

#### cfd_engine.py
**Role:** Computational Fluid Dynamics engine

**Classes:**
- `Fluid` - Fluid property definitions (Water, Air with temperature dependence)
- `CFDMesh` - Structured computational mesh
- `BoundaryCondition` - CFD boundary conditions
- `CFDEngine` - Main CFD solver

**Flow Types:**
- Laminar flow
- Turbulent flow
- Transitional flow
- Compressible flow
- Incompressible flow

**Boundary Types:**
- Wall boundaries
- Inlet (velocity/pressure)
- Outlet (pressure/velocity)
- Symmetry planes
- Periodic boundaries

**Solver Methods:**
- SIMPLE (Semi-Implicit Method for Pressure Linked Equations)
- PISO (Pressure-Implicit with Splitting of Operators)
- Coupled solver
- Fractional step method

**Key Features:**
- Navier-Stokes equation solving
- Pressure-velocity coupling
- Turbulence modeling
- Heat transfer coupling

---

#### electromagnetics_engine.py
**Role:** Electromagnetic field simulation

**Classes:**
- `EMaterial` - Electromagnetic material properties (Vacuum, Air, Copper, FR-4, Silicon)
- `Source` - EM source definitions
- `ElectromagneticsEngine` - Main EM solver

**Field Types:**
- Electric fields
- Magnetic fields
- Electromagnetic fields
- Electrostatic fields
- Magnetostatic fields

**Material Types:**
- Dielectric materials
- Conductors
- Magnetic materials
- Anisotropic materials

**Key Features:**
- 3D FDTD (Finite-Difference Time-Domain)
- 2D FDFD (Finite-Difference Frequency-Domain)
- Maxwell's equations solving
- Wave propagation
- Material interface handling

---

#### material_science_engine.py
**Role:** Material science calculations and crystal structures

**Classes:**
- `Material` - Comprehensive material properties
- Material presets: Aluminum 6061, Steel AISI 4140, Titanium Grade 5, Copper C101, Inconel 718

**Crystal Systems:**
- Cubic, Tetragonal, Orthorhombic, Hexagonal, Trigonal, Monoclinic, Triclinic

**Bravais Lattices:**
- Simple Cubic (SC), BCC, FCC
- Simple/Body-centered Tetragonal
- Various Orthorhombic structures
- Hexagonal, Rhombohedral
- Monoclinic, Triclinic

**Key Features:**
- Mechanical property database
- Physical property calculations
- Thermal properties
- Electrical resistivity
- Crystal structure definitions
- Lattice parameters

---

#### advanced_material_science.py
**Role:** Advanced material analysis

**API Endpoints/Functions:**
- `stress_strain_curve(material, max_strain, n_points, model)` - Generate stress-strain curves
  - Models: bilinear, Ramberg-Osgood, Johnson-Cook
- `fatigue_analysis()` - Fatigue life prediction
- `creep_analysis()` - Creep behavior modeling
- `fracture_mechanics()` - Fracture toughness calculations
- `failure_criterion_check()` - Apply failure criteria

**Failure Criteria:**
- Von Mises criterion
- Tresca criterion
- Mohr-Coulomb criterion
- Tsai-Wu (composites)
- Maximum stress criterion

**Material Types:**
- Metals
- Polymers
- Ceramics
- Composites
- Smart materials

**Key Features:**
- Material model presets (Steel 4140, Aluminum 6061-T6, Carbon Fiber/Epoxy)
- Fatigue life calculations
- Creep modeling
- Fracture mechanics

---

### Utilities Module (`utils/`)

#### exceptions.py
**Role:** Custom exception classes with user-friendly messages

**Exception Classes:**
- `KoiosError` - Base exception class
- `InvalidDimensionError` - Matrix dimension incompatibility
- `InvalidInputError` - Invalid user input
- `NumericalInstabilityError` - Numerical computation instability
- `SingularMatrixError` - Singular matrix operations
- `ConvergenceError` - Algorithm convergence failure
- `DomainError` - Input outside valid domain
- `ExpressionParseError` - Expression parsing failure
- `UndefinedOperationError` - Mathematically undefined operations
- `ConfigurationError` - Invalid configuration

**Key Features:**
- Formatted error messages with hints
- Context-aware error descriptions
- User-friendly suggestions
- Usage guide: All modules should import and use custom exceptions instead of generic ValueError/TypeError
- Provides specific exception types for different error categories (dimensions, input validation, convergence, etc.)

**Implementation Status:**
- matrix_operations.py: Now uses SingularMatrixError and InvalidDimensionError
- ode_solver.py: Now uses InvalidInputError, ConfigurationError, and ExpressionParseError for better error handling

---

#### result_builder.py
**Role:** Standard result structure builder for consistent API responses

**API Endpoints/Functions:**
- `ResultBuilder` - Fluent builder class for result dictionaries
  - Methods: `success(value)`, `result(value, type)`, `error(message)`, `iterations(count, converged)`, `add_warning(message)`, `metadata(key, value)`, `merge_metadata(dict)`, `build()`
  - Returns: Configured result dictionary with automatic computation time tracking
- `create_result(success, result, result_type, error, **kwargs)` - Convenience factory function
- `StandardResult` - TypedDict for result structure documentation

**Standard Result Structure:**
```python
{
    'success': bool,           # Always required - whether computation succeeded
    'result': Any,             # Main computational result
    'result_type': str,        # Description of result type (e.g., 'matrix', 'scalar', 'function')
    'iterations': int,         # For iterative methods (optional)
    'converged': bool,         # For iterative methods (optional)
    'error': str,              # Error message if failed (optional)
    'warnings': List[str],     # Non-fatal warnings (optional)
    'computation_time': float, # Time in seconds (automatically added)
    'metadata': dict           # Additional engine-specific data (optional)
}
```

**Key Features:**
- Fluent/builder pattern for easy result creation
- Automatic computation time tracking
- Automatic cleanup of empty lists and dicts
- Type-safe with TypedDict documentation
- Consistent API across all engines
- Reduces code duplication in result dictionary creation

**Usage Examples:**
```python
# Using ResultBuilder
result = ResultBuilder().success(True).result(42, 'scalar').iterations(10, True).build()

# Using create_result convenience function
result = create_result(success=True, result=42, result_type='scalar', iterations=10)

# Building with metadata
result = ResultBuilder().result(matrix, 'matrix').metadata('shape', (3, 3)).build()
```

---

#### validators.py
**Role:** Input validation utilities

**API Endpoints/Functions:**
- `validate_expression(expression)` - Validate mathematical expressions
  - Returns: `{'is_valid': bool, 'error_message': str, 'variables': list, 'functions': list, 'complexity_score': int, 'suggestions': list}`
- `validate_matrix_input(matrix_data)` - Validate matrix data
  - Returns: `{'is_valid': bool, 'error_message': str, 'rows': int, 'cols': int, 'is_square': bool, 'is_numeric': bool, 'suggestions': list}`
- `validate_numeric_input()` - Validate numeric inputs
- `validate_domain()` - Validate function domains
- `validate_convergence_params()` - Validate algorithm parameters
- `validate_ode_input(ode_expression)` - Validate ODE expressions with proper derivative order detection
- `validate_physics_parameters(simulation_type, parameters)` - Validate physics simulation parameters

**Key Features:**
- Parentheses balancing check
- Valid character checking (including ^ auto-conversion to **)
- Enhanced consecutive operator detection with explicit pattern separation:
  - Allows: ** (exponentiation), unary +/- after operators/parentheses/commas, unary signs in general
  - Disallows: true binary operator sequences like +*, +/, -*,  -/, *+, *-, /+, /-
  - Improved clarity between allowed unary patterns and forbidden consecutive binary operators
- SymPy parsing validation
- Complexity scoring
- Improved domain validation for sqrt and log functions
- Proper ODE order detection using derivative counting
- Comprehensive denominator finding using SymPy's denom()
- Documented implicit multiplication behavior
- Physics parameter validation with reasonable default constraints

---

#### math_helpers.py
**Role:** Mathematical utility functions

**API Endpoints/Functions:**
- `is_numeric(value)` - Check if value is numeric
- `safe_eval(expression, variables)` - Safely evaluate expressions using SymPy parser instead of eval()
- `format_number(value, precision)` - Format numbers for display
- `format_scientific(value, precision)` - Format scientific notation
- `degrees_to_radians(degrees)` - Angle conversion
- `radians_to_degrees(radians)` - Angle conversion
- `factorial(n)` - Calculate factorial
- `binomial_coefficient()` - Calculate binomial coefficients
- `gcd()`, `lcm()` - Greatest common divisor, least common multiple
- `slope(x1, y1, x2, y2)` - Calculate slope with proper handling of vertical lines

**Key Features:**
- Safe expression evaluation using SymPy parser (no eval() function)
- Number formatting utilities
- Angle conversions
- Common mathematical operations
- Complex number formatting
- Improved slope calculation distinguishing positive and negative infinity for vertical lines

---

#### ui_helpers.py
**Role:** UI utility functions for Streamlit

**API Endpoints/Functions:**
- `show_toast(message, kind)` - Display toast notifications
- `timed_spinner(label)` - Context manager with timing
- `run_task(label, func, *args, success_message, error_message, toast, **kwargs)` - Execute tasks with UI feedback
  - Returns: `(result, elapsed_seconds, error)`
- `copy_button(text, key, label)` - Render copy-to-clipboard button
- `load_css(css_file)` - Load and apply CSS file

**Key Features:**
- Toast notification fallbacks
- Timing and performance tracking
- Error handling with user feedback
- Clipboard integration
- CSS loading utilities

---

### UI Module (`ui/`)

#### calculator_panel.py
**Role:** Scientific calculator interface

**Functions:**
- `render_calculator_panel()` - Main calculator UI

**Key Features:**
- Basic arithmetic operations
- Advanced functions (trig, log, exp)
- Expression evaluation with variable substitution
- Improved variable parsing with empty value filtering
- Enhanced error handling for variable conversion
- History tracking
- Result formatting

---

#### matrix_panel.py
**Role:** Matrix calculator interface

**Functions:**
- `render_matrix_panel()` - Main matrix operations UI
- `load_example_matrix(example_name)` - Callback for loading example matrices
- `get_matrix_input_setup(method, matrix_label)` - Setup matrix input UI
- `get_matrix_from_setup(method, matrix_label)` - Generate matrix from setup with error handling
- `manual_matrix_entry_setup(matrix_label)` - Setup manual matrix entry UI
- `manual_matrix_entry_generate(matrix_label)` - Generate matrix with validation and missing value warnings
- `generate_random_matrix_setup(matrix_label)` - Setup random matrix generation
- `generate_random_matrix_generate(matrix_label)` - Generate random matrix
- `get_predefined_matrix_setup(matrix_label)` - Setup predefined matrix selection
- `get_predefined_matrix_generate(matrix_label)` - Generate predefined matrix with selection validation
- `perform_single_matrix_operation(matrix, operation)` - Execute single-matrix operations with type checking
- `perform_dual_matrix_operation(matrix_a, matrix_b, operation)` - Execute two-matrix operations with type checking
- `perform_matrix_norm(matrix, norm_type)` - Calculate matrix norms with type checking

**Constants:**
- `MATRIX_EXAMPLES` - Quick-load example matrices
- `PREDEFINED_MATRICES` - Predefined matrices (defined once at module level)

**Operations:**
- Matrix creation (manual/random/identity)
- Addition, subtraction, multiplication
- Determinant, inverse, transpose
- Eigenvalues and eigenvectors
- Matrix rank
- Linear system solving
- Decompositions (QR, SVD, LU)

**Key Features:**
- Callback-based matrix loading (no explicit st.rerun())
- Comprehensive input validation with user warnings
- Type checking on all operations (NumPy/SymPy validation)
- Consolidated predefined matrix definitions
- Detailed error messages for all failure modes

---

#### calculus_panel.py
**Role:** Calculus operations interface

**Functions:**
- `render_calculus_panel()` - Main calculus UI with 8 operation tabs
- `render_derivatives_section()` - Derivatives calculator
- `render_integrals_section()` - Integrals calculator
- `render_limits_section()` - Limits calculator
- `render_series_section()` - Series expansion
- `render_function_analysis_section()` - Function analysis
- `render_multivariable_calculus_section()` - Multivariable calculus
- `render_differential_equations_section()` - Differential equations
- `render_transforms_series_section()` - Transforms and series analysis (new comprehensive tool)
- `render_laplace_transform_tab()` - Laplace transform operations (forward/inverse)
- `render_fourier_transform_tab()` - Fourier transform operations (forward/inverse)
- `render_transfer_function_tab()` - Transfer function analysis with stability checking
- `render_de_solving_tab()` - DE solving with transforms (Laplace implemented, Fourier disabled)
- `render_fourier_series_tab()` - Fourier series expansion
- `visualize_fourier_series()` - Fourier series visualization

**Operations:**
- Differentiation (single and multi-variable)
- Integration (indefinite and definite)
- Limits (with direction)
- Series expansions
- Partial derivatives
- Laplace transforms (forward and inverse)
- Fourier transforms (forward and inverse)
- Transfer function computation from differential equations
- Differential equation solving using Laplace transform
- Solution decomposition:
  - Transient/steady-state (pole-based analysis)
  - Zero-input response (ICs only)
  - Zero-state response (input only)
- Fourier series with trigonometric and complex forms

**Key Features:**
- Interactive parameter controls
- Step-by-step solutions
- Real-time visualization
- Transfer function analysis with pole-zero plots
- System stability checking (pole real parts)
- Zero-input/zero-state decomposition in DE solving
- Improved transient/steady detection via pole analysis
- Relaxed IC handling (uses all available ICs)
- Fourier DE solving disabled with clear status message
- Fourier series convergence visualization
- Error analysis for series approximations
- LaTeX output with copy functionality

**Implementation Status:**
- ✅ Laplace transforms (fully implemented)
- ✅ Fourier transforms (fully implemented)
- ✅ Transfer functions from DEs (coefficient extraction working)
- ✅ DE solving with Laplace (zero-input/zero-state, transient/steady)
- ⚠️ DE solving with Fourier (disabled - not yet implemented)
- ✅ Fourier series (fully implemented)

---

#### equation_solver_panel.py
**Role:** Equation solving interface

**Functions:**
- `render_equation_solver_panel()` - Equation solver UI

**Solver Types:**
- Linear equations
- Polynomial equations
- Nonlinear equations
- Systems of equations
- Differential equations

---

#### physics_panel.py
**Role:** Physics simulation interface

**Functions:**
- `render_physics_panel()` - Main physics UI
- `render_projectile_motion()` - Projectile simulation
- `render_simple_harmonic_motion()` - SHM simulation
- `render_damped_oscillator()` - Damped motion
- `render_pendulum_simulation()` - Pendulum dynamics
- `render_rc_circuit()` - RC circuit analysis
- Additional simulation renderers for various physics scenarios

**Key Features:**
- Interactive parameter controls
- Real-time visualization
- Result plotting with Plotly
- Physical constant reference

---

#### engineering_panel.py
**Role:** Engineering simulation interface

**Functions:**
- `render_engineering_panel()` - Main engineering UI

**Modules:**
- Finite Element Analysis
- Computational Fluid Dynamics
- Electromagnetics
- Material Science
- Thermodynamics
- Signal Processing

---

#### complex_analysis_panel.py
**Role:** Complex number analysis interface

**Functions:**
- `render_complex_analysis_panel()` - Complex analysis UI

**Operations:**
- Complex arithmetic
- Polar/exponential forms
- Complex roots
- Complex logarithm
- Analytic functions
- Contour integration
- Residue calculations

---

#### tensor_calculus_panel.py
**Role:** Tensor calculus interface

**Functions:**
- `render_tensor_calculus_panel()` - Main tensor UI
- `render_metric_tensors()` - Metric tensor operations
- `render_christoffel_symbols()` - Christoffel symbol calculations
- `render_curvature_calculations()` - Curvature computations
- `render_standard_metrics()` - Standard metric definitions
- `render_geodesics()` - Geodesic equations

**Key Features:**
- Metric tensor input
- Curvature calculations
- Standard metrics (Minkowski, Schwarzschild, etc.)
- Geodesic equations

---

#### numerical_methods_panel.py
**Role:** Numerical methods interface

**Functions:**
- `render_numerical_methods_panel()` - Numerical methods UI

**Methods:**
- Root finding (Newton, Bisection, Secant)
- Numerical integration
- Interpolation
- Curve fitting
- FFT/spectral methods

---

#### optimization_panel.py
**Role:** Optimization algorithms interface

**Functions:**
- `render_optimization_panel()` - Optimization UI

**Algorithms:**
- Gradient descent variants
- BFGS and L-BFGS
- Nelder-Mead
- Genetic algorithms
- Particle swarm
- Simulated annealing

---

#### visualization_panel.py
**Role:** Function visualization interface

**Functions:**
- `render_visualization_panel()` - Main visualization UI
- `render_2d_function_plot()` - 2D plotting
- `render_3d_surface_plot()` - 3D surface plots
- `render_parametric_plot()` - Parametric curves
- `render_polar_plot()` - Polar plots
- `render_vector_field()` - Vector field visualization
- `render_contour_plot()` - Contour plots
- `render_animation_plot()` - Animated plots
- `render_function_analysis_viz()` - Function analysis

**Key Features:**
- Interactive Plotly visualizations
- Multiple plot types
- Customizable styling
- Export capabilities

---

### Templates Module (`templates/`)

All template files follow the same pattern:

**Functions:**
- `generate_test()` - Generate test cases
- `run_test(test_data)` - Execute tests

**Template Files:**
- `arithmetic_operations.py` - Basic arithmetic test templates
- `calculus_operations.py` - Calculus operation tests
- `engineering_finite_element_analysis.py` - FEA test cases
- `engineering_fluid_dynamics.py` - CFD test cases
- `engineering_signal_processing.py` - Signal processing tests
- `engineering_thermodynamics.py` - Thermodynamics tests
- `matrix_operations.py` - Matrix operation tests
- `physics_simulations.py` - Physics simulation tests
- `trigonometric_functions.py` - Trigonometric function tests

**Key Features:**
- Standardized test generation
- Test execution framework
- Expected results validation

---

## Architecture Summary

### Data Flow

1. **User Input** → Streamlit UI panels (`ui/`)
2. **Validation** → Input validators (`utils/validators.py`)
3. **Parsing** → Expression parser (`core/expression_parser.py`)
4. **Processing** → Core engines (`core/`)
5. **Results** → Formatted output via UI helpers (`utils/ui_helpers.py`)
6. **Visualization** → Plotly/Matplotlib via visualization panel

### Module Dependencies

```
app.py (entry point)
├── ui/ (presentation layer)
│   ├── Depends on: core/, utils/
│   └── Renders: Interactive Streamlit components
├── core/ (business logic layer)
│   ├── Calculation engines
│   ├── Physics simulators
│   ├── Engineering modules
│   └── Mathematical solvers
├── utils/ (utility layer)
│   ├── Validators
│   ├── Helpers
│   └── Exceptions
└── templates/ (testing layer)
    └── Test generation and execution
```

### Key Design Patterns

- **Singleton Pattern:** Core engines instantiated once per session
- **Factory Pattern:** Material and mesh creation
- **Strategy Pattern:** Multiple solver methods selectable at runtime
- **Observer Pattern:** Real-time visualization updates
- **Template Method:** Standardized test execution

### Technology Stack

- **Frontend:** Streamlit
- **Symbolic Math:** SymPy
- **Numerical Computation:** NumPy, SciPy
- **Visualization:** Plotly, Matplotlib
- **Matrix Operations:** NumPy, SymPy
- **Sparse Matrices:** SciPy sparse