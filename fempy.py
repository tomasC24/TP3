"""The file is organized into four tiny classes that each do one clear job:

1. Mesh             - holds the node coordinates.
2. FunctionSpace    - stores shape functions.
3. System           - builds element matrices/vectors and builds the linear system.
4. Solver           - solves the linear system with NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import override


# -----------------------------------------------------------------------------
# 1) Mesh - describes the geometry (nodes)
# -----------------------------------------------------------------------------
class Mesh(ABC):
    def __init__(self):
        pass


class Mesh1D(Mesh):
    """Create a 1-D mesh from a list/array of x-coordinates."""

    def __init__(self, coordinates):
        # Convert any sequence to a NumPy array
        self.nodal_coordinates = np.asarray(coordinates, dtype=float)

        # Check input validity (must be 1-D and sorted)
        if self.nodal_coordinates.ndim != 1:
            raise ValueError("Coordinates must be a 1-D sequence.")
        if not np.all(np.diff(self.nodal_coordinates) > 0):
            raise ValueError("Coordinates must be in ascending order.")


# -----------------------------------------------------------------------------
# 2) FunctionSpace - basis / shape functions
# -----------------------------------------------------------------------------
class FunctionSpace(ABC):
    """Declared abstract, does nothing, forces derived classes to implement these."""

    def __init__(self):
        pass

    @abstractmethod
    def local_to_global(self, e, k):
        pass

    @abstractmethod
    def global_to_local(self, e, n):
        pass

    @abstractmethod
    def localize(self, e, values):
        pass

    @abstractmethod
    def ref_N(self, k, eta):
        pass

    @abstractmethod
    def ref_dN_deta(self, k, eta):
        pass

    @abstractmethod
    def N(self, e, k, x):
        pass

    @abstractmethod
    def dN_dx(self, e, k, x):
        pass

    @override
    def J(self, e, eta):
        pass


class FunctionSpaceSeg1(FunctionSpace):
    """Linear (P1) 1D-finite-element basis on a given mesh."""

    # Defines a quadrature rule for P1 1D elements ----------------------------
    class QuadratureRule1PointSeg1:
        """1-point Gauss rule on [-1, 1]."""

        def __init__(self):
            self.position = np.array([0.0])
            self.weight = np.array([2.0])
            self.n_quad = 1

    # Defines a quadrature rule for P1 1D elements, just for testing ----------
    class QuadratureRule3PointSeg1:
        """3-point Gauss rule on [-1, 1]."""

        def __init__(self):
            self.position = np.array([-0.7745966692, 0.0, 0.7745966692])
            self.weight = np.array([0.5555555556, 0.8888888889, 0.5555555556])
            self.n_quad = 3

    def __init__(self, mesh):
        # Mesh and element data
        self.mesh = mesh

        # Nodal coordinates
        self.nodal_coordinates = mesh.nodal_coordinates
        self.n_nodes = len(self.nodal_coordinates)

        self.n_elements = self.n_nodes - 1
        self.n_element_nodes = 2

        # Consecutive nodes form an element: (0,1), (1,2), (2,3), ...
        self.elements = np.array([(i, i + 1) for i in range(self.n_elements)])
        self.element_sizes = np.array(
            [
                self.nodal_coordinates[i + 1] - self.nodal_coordinates[i]
                for i in range(self.n_elements)
            ]
        )

        # Attach quadrature rule
        self.quad_rule = self.QuadratureRule1PointSeg1()

    # Helper functions for assembling -----------------------------------------
    """Maps (element index, local node index) to global node index.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).

    Returns:
        int: Global node index corresponding to local index k in element e.
    """

    def local_to_global(self, e, k):
        return self.elements[e][k]

    """Maps (element index, global node index) to local node index.

    Args:
        e (int): Global element index.
        n (int): Global node index.

    Returns:
        int: Local node index (0 or 1 for a linear element) corresponding to global index n in element e.
    """

    def global_to_local(self, e, n):
        return n - e

    """Returns the local values of an array belonging to a given element.

    Args:
        e (int): Global element index.
        values (array): Global array of values.

    Returns:
        array: Array of local values at the given element nodes.
    """

    def localize(self, e, values):
        local_values = np.zeros(self.n_element_nodes)
        for node in self.elements[e]:
            local_node_index = self.global_to_local(e, node)
            local_values[local_node_index] = values[node]
        return local_values

    # Reference-element shape functions ---------------------------------------
    """Reference shape functions on [-1, 1]:
        N_0(eta) = 1/2 * (1 - eta),
        N_1(eta) = 1/2 * (1 + eta).

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: refN_k(eta).
    """

    @override
    def ref_N(self, k, eta):
        return ((1 - eta) * 0.5, (1 + eta) * 0.5)[k]

    """Derivatives of reference shape functions w.r.t. eta on [-1, 1]:
        dN_0/deta(eta) = -1/2,
        dN_1/deta(eta) = 1/2.

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: drefN_k/deta(eta).
    """

    @override
    def ref_dN_deta(self, k, eta):
        # Derivatives for the reference element are constant +/-1/2
        return (-0.5, 0.5)[k]

    # Element shape functions -------------------------------------------------
    """Return the value of the k-th (local index) 1-D linear shape function at global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: N_{e,k}(x).
    """

    @override
    def N(self, e, k, x):
        pass

    """Return the value of the k-th 1-D linear shape function derivative w.r.t. x at global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: dN_{e,k}/dx(x).
    """

    @override
    def dN_dx(self, e, k, x):
        pass

    # Jacobian ----------------------------------------------------------------
    """Return the value of the jacobian at local coordinate eta for element of global index e.

    Args:
        e (int): Global element index.
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: J_e(eta).
    """

    @override
    def J(self, e, eta):
        # x(eta) = x_a + 1/2 * h * (1 + eta), dx/deta = 1/2 * h
        return 0.5 * self.element_sizes[e]


class FunctionSpaceSeg2(FunctionSpace):
    """Quadratic (P2) 1D-finite-element basis on a given mesh."""

    # Defines a quadrature rule for P2 1D elements ----------------------------
    class QuadratureRule3PointSeg2:
        """3-point Gauss rule on [-1, 1]."""

        def __init__(self):
            self.position = np.array([-0.7745966692, 0.0, 0.7745966692])
            self.weight = np.array([0.5555555556, 0.8888888889, 0.5555555556])
            self.n_quad = 3

    def __init__(self, mesh):
        pass

        # Mesh and element data
        self.mesh = mesh

        # P1 nodal coordinates
        self.p1_nodal_coordinates = mesh.nodal_coordinates
        self.n_p1_nodes = len(self.p1_nodal_coordinates)

        # Second order nodes
        self.nodal_coordinates = mesh.nodal_coordinates
        for k in range(self.n_p1_nodes - 1):
            mid_node = 0.5 * (self.p1_nodal_coordinates[k] + self.p1_nodal_coordinates[k + 1])
            insert_pos = 2 * k + 1
            self.nodal_coordinates = np.insert(self.nodal_coordinates, insert_pos, mid_node)
        self.n_nodes = len(self.nodal_coordinates)

        self.n_elements = self.n_p1_nodes - 1
        self.n_element_nodes = 3

        # Ordering: (0,1,2), (2,3,4), (4,5,6), ...
        self.elements = np.array([(2 * i, 2 * i + 1, 2 * i + 2) for i in range(self.n_elements)])
        self.element_sizes = np.array(
            [
                self.p1_nodal_coordinates[i + 1] - self.p1_nodal_coordinates[i]
                for i in range(self.n_elements)
            ]
        )

        # Attach quadrature rule
        self.quad_rule = self.QuadratureRule3PointSeg2()

    # Helper functions for assembling -----------------------------------------
    """Maps (element index, local node index) to global node index.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).

    Returns:
        int: Global node index corresponding to local index k in element e.
    """

    def local_to_global(self, e, k):
        return self.elements[e][k]

    """Maps (element index, global node index) to local node index.

    Args:
        e (int): Global element index.
        n (int): Global node index.

    Returns:
        int: Local node index (0 or 1 for a linear element) corresponding to global index n in element e.
    """

    def global_to_local(self, e, n):
        return n - 2 * e

    """Returns the local values of an array belonging to a given element.

    Args:
        e (int): Global element index.
        values (array): Global array of values.

    Returns:
        array: Array of local values at the given element nodes.
    """

    def localize(self, e, values):
        local_values = np.zeros(self.n_element_nodes)
        for node in self.elements[e]:
            local_node_index = self.global_to_local(e, node)
            local_values[local_node_index] = values[node]
        return local_values

    # Reference-element shape functions ---------------------------------------
    """Reference shape functions on [-1, 1]:
        N_0(eta) = 1/2 eta * (eta - 1),
        N_1(eta) = 1 - eta^2,
        N_2(eta) = 1/2 eta * (eta + 1).

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: refN_k(eta).
    """

    @override
    def ref_N(self, k, eta):
        return (0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0))[k]

    """Derivatives of reference shape functions w.r.t. eta on [-1, 1]:
        dN_0/deta(eta) = -1/2,
        dN_1/deta(eta) = 1/2.

    Args:
        k (int): Local node index (0 or 1 for a linear element).
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: drefN_k/deta(eta).
    """

    @override
    def ref_dN_deta(self, k, eta):
        return (eta - 0.5, -2.0 * eta, eta + 0.5)[k]

    # Element shape functions -------------------------------------------------
    """Return the value of the k-th (local index) 1-D linear shape function at global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: N_{e,k}(x).
    """

    @override
    def N(self, e, k, x):
        pass

    """Return the value of the k-th 1-D linear shape function derivative w.r.t. x at global coordinate x for element of global index e.

    Args:
        e (int): Global element index.
        k (int): Local node index (0 or 1 for a linear element).
        x (float): Global coordinate.

    Returns:
        float: dN_{e,k}/dx(x).
    """

    @override
    def dN_dx(self, e, k, x):
        pass

    # Jacobian ----------------------------------------------------------------
    """Return the value of the jacobian at local coordinate eta for element of global index e.

    Args:
        e (int): Global element index.
        eta (float): Local coordinate in the reference element, -1 <= eta <= 1.

    Returns:
        float: J_e(eta).
    """

    @override
    def J(self, e, eta):
        ids = self.elements[e]
        x1, x2, x3 = self.nodal_coordinates[ids]

        return (eta - 0.5) * x1 + (-2 * eta) * x2 + (eta + 0.5) * x3


# -----------------------------------------------------------------------------
# 3) System - problem definition
# -----------------------------------------------------------------------------
class System(ABC):
    """Build the global matrix A and RHS vector b from V."""

    def __init__(self, V):
        pass

    @abstractmethod
    def assemble_capacity(self):
        pass

    @abstractmethod
    def assemble_stiffness(self):
        pass

    @abstractmethod
    def assemble_rhs(self):
        pass

    @abstractmethod
    def assemble_residual(self):
        pass

    @abstractmethod
    def assemble_element_capacity(self, e):
        pass

    @abstractmethod
    def assemble_element_stiffness(self, e):
        pass

    @abstractmethod
    def assemble_element_rhs(self, e):
        pass

    @abstractmethod
    def assemble_element_residual(self, e):
        pass

    @abstractmethod
    def apply_BCs(self):
        pass

    @abstractmethod
    def apply_initial_conditions(self):
        pass


class SteadyStateHeatTransferSystem(System):
    """Build the stiffness matrix K and RHS vector q from V."""

    def __init__(self, V, k, Q):
        self.V = V

        # Conductivity and heat source/sink
        self.k = k
        self.Q = Q

        # Vector of unknowns
        self.u = np.zeros(self.V.n_nodes)

        # Global stiffness and RHS vector
        # TODO: implement a skyline sparse matrix storage scheme
        self.K = np.zeros((self.V.n_nodes, self.V.n_nodes))
        self.q = np.zeros(self.V.n_nodes)

        # Boundary condition types and values (0 = Neumann, 1 = Dirichlet)
        self.BC_types = np.zeros(self.V.n_nodes)
        self.BC_values = np.zeros(self.V.n_nodes)

    @override
    def assemble_capacity(self):
        pass

    @override
    def assemble_stiffness(self):
        self.K[:, :] = 0

        for e in range(self.V.n_elements):
            self.assemble_element_stiffness(e)

        return

    @override
    def assemble_rhs(self):
        self.Q[:] = 0

        for e in range(self.V.n_elements):
            self.assemble_element_rhs(e)

        return

    @override
    def assemble_residual(self):
        pass

    @override
    def assemble_element_capacity(self, e):
        pass

    @override
    def assemble_element_stiffness(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local stiffness matrix
        local_K = np.zeros((n_element_nodes, n_element_nodes))
        # Local conductivity
        local_k = self.V.localize(e, self.k)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Conductivity
            k_g = 0.0
            for c in range(n_element_nodes):
                k_g += local_k[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                for b in range(n_element_nodes):
                    dNa = self.V.ref_dN_deta(a, pos)
                    dNb = self.V.ref_dN_deta(b, pos)

                    # Add to local stiffness
                    local_K[a, b] += k_g * (dNa / J) * (dNb / J) * W * J

        for a in range(n_element_nodes):
            for b in range(n_element_nodes):
                # Assemble to global stiffness
                i, j = self.V.local_to_global(e, a), self.V.local_to_global(e, b)
                self.K[i, j] += local_K[a, b]

        return

    @override
    def assemble_element_rhs(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local RHS vector
        local_q = np.zeros(n_element_nodes)
        # Local heat source/sink vector
        local_Q = self.V.localize(e, self.Q)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Heat source/sink
            Q_g = 0.0
            for c in range(n_element_nodes):
                Q_g += local_Q[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                Na = self.V.ref_N(a, pos)

                # Add to local RHS
                local_q[a] += Na * Q_g * W * J

        for a in range(n_element_nodes):
            # Assemble to global RHS
            i = self.V.local_to_global(e, a)
            self.q[i] += local_q[a]

        return

    @override
    def assemble_element_residual(self, e):
        pass

    @override
    def apply_BCs(self):
        # Dirichlet BCs
        dirichlet_nodes = np.where(self.BC_types == 1)[0]

        for i in dirichlet_nodes:
            u_bar = self.BC_values[i]

            # Substract the i-th column times u_bar
            self.q -= self.K[:, i] * u_bar

            # Zero the i-th row/column
            self.K[i, :] = 0.0
            self.K[:, i] = 0.0

            # And set the corresponding diagonal value to 1, RHS to u_bar
            self.K[i, i] = 1.0
            self.q[i] = u_bar

        # Neumann BCs
        neumann_nodes = np.where(self.BC_types == 0)[0]

        for i in neumann_nodes:
            self.q[i] += self.BC_values[i]

    @override
    def apply_initial_conditions(self):
        pass


# TODO: implement this
class TransientHeatTransferSystem(SteadyStateHeatTransferSystem):
    """Build the capacity matrix and residual vector from V."""

    def __init__(self, V, k, c, rho, Q):
        self.V = V

        # Conductivity, density, heat capacity, and heat source/sink
        self.k = k
        self.c = c
        self.rho = rho
        self.Q = Q

        # Vector of unknowns
        self.u = np.zeros(self.V.n_nodes)

        # Global residual and capacity
        self.res = np.zeros(self.V.n_nodes)
        self.capacity_matrix = np.zeros((self.V.n_nodes, self.V.n_nodes))
        # diagonal lumped capacity matrix
        self.capacity_lumped = np.zeros(self.V.n_nodes)

        # Boundary condition types and values (0 = Neumann, 1 = Dirichlet)
        self.BC_types = np.zeros(self.V.n_nodes)
        self.BC_values = np.zeros(self.V.n_nodes)

        self.A = np.zeros((self.V.n_nodes, self.V.n_nodes))
        self.b = np.zeros(self.V.n_nodes)

        self.q = np.zeros(self.V.n_nodes)

    @override
    def assemble_capacity(self):
        self.capacity_matrix[:, :] = 0
        self.capacity_lumped[:] = 0

        # calcula la matriz de capacidad consistente
        for e in range(self.V.n_elements):
            self.assemble_element_capacity(e)

        for col in range(self.V.n_nodes):
            sum = 0.0
            # suma en cada fila y asigna a la diagonal
            for row in range(self.V.n_nodes):
                sum += self.capacity_matrix[col][row]

            # de la matriz de capacidad lumpeada
            self.capacity_lumped[col] = sum

        return

    @override
    def assemble_stiffness(self):
        pass

    @override
    def assemble_rhs(self):
        self.Q[:] = 0

        for e in range(self.V.n_elements):
            self.assemble_element_rhs(e)

        return

    @override
    def assemble_residual(self):
        self.res[:] = 0

        for e in range(self.V.n_elements):
            self.assemble_element_residual(e)

        return

    @override
    def assemble_element_capacity(self, e):
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad

        local_capacity = np.zeros((n_element_nodes, n_element_nodes))
        local_rho = self.V.localize(e, self.rho)
        local_c = self.V.localize(e, self.c)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Heat capacity
            c_g = 0.0
            # Material density
            rho_g = 0.0
            for c in range(n_element_nodes):
                # interpola y obtiene el valor en los P.G.
                c_g += local_c[c] * self.V.ref_N(c, pos)
                rho_g += local_rho[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                for b in range(n_element_nodes):
                    Na = self.V.ref_N(a, pos)
                    Nb = self.V.ref_N(b, pos)

                    local_capacity[a][b] += Na * Nb * c_g * rho_g * W * J

        for a in range(n_element_nodes):
            for b in range(n_element_nodes):
                # Assemble to global capacity
                i, j = self.V.local_to_global(e, a), self.V.local_to_global(e, b)
                self.capacity_matrix[i, j] += local_capacity[a, b]

    @override
    def assemble_element_stiffness(self, e):
        pass

    @override
    def assemble_element_rhs(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local RHS vector
        local_q = np.zeros(n_element_nodes)
        # Local heat source/sink vector
        local_Q = self.V.localize(e, self.Q)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Heat source/sink
            Q_g = 0.0
            for c in range(n_element_nodes):
                Q_g += local_Q[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                Na = self.V.ref_N(a, pos)

                # Add to local RHS
                local_q[a] += Na * Q_g * W * J

        for a in range(n_element_nodes):
            # Assemble to global RHS
            i = self.V.local_to_global(e, a)
            self.q[i] += local_q[a]

        return

    @override
    def assemble_element_residual(self, e):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local residual vector
        local_res = np.zeros(n_element_nodes)
        # Local u
        local_u = self.V.localize(e, self.u)
        # Local conductivity
        local_k = self.V.localize(e, self.k)
        # Local heat source/sink vector
        local_Q = self.V.localize(e, self.Q)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Conductivity
            k_g = 0.0
            # Heat source/sink
            Q_g = 0.0
            for c in range(n_element_nodes):
                k_g += local_k[c] * self.V.ref_N(c, pos)
                Q_g += local_Q[c] * self.V.ref_N(c, pos)

            # grad(u) (scalar in 1D)
            grad_u_g = 0.0
            for b in range(n_element_nodes):
                dNb = self.V.ref_dN_deta(b, pos)
                grad_u_g += local_u[b] * (dNb / J)

            # integral del termino fuente/sumidero
            for a in range(n_element_nodes):
                Na = self.V.ref_N(a, pos)

                # Add to local RHS
                local_res[a] += Na * Q_g * W * J

            # integral del termino gradN_a · gradT
            for a in range(n_element_nodes):
                dNa = self.V.ref_dN_deta(a, pos)

                # Add to local stiffness
                local_res[a] -= k_g * (dNa / J) * grad_u_g * W * J

        for a in range(n_element_nodes):
            # Assemble to global RHS
            i = self.V.local_to_global(e, a)
            self.res[i] += local_res[a]

        return

    def assemble_a(self, dt):
        self.A[:, :] = 0
        for e in range(self.V.n_elements):
            self.assemble_element_a(e, dt)
        return

    def assemble_element_a(self, e, dt):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local stiffness matrix
        local_A = np.zeros((n_element_nodes, n_element_nodes))
        # Local conductivity
        local_k = self.V.localize(e, self.k)
        local_capacity = np.zeros((n_element_nodes, n_element_nodes))
        local_rho = self.V.localize(e, self.rho)
        local_c = self.V.localize(e, self.c)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Conductivity
            k_g = 0.0
            for c in range(n_element_nodes):
                k_g += local_k[c] * self.V.ref_N(c, pos)

            # Heat capacity
            c_g = 0.0
            # Material density
            rho_g = 0.0
            for c in range(n_element_nodes):
                # interpola y obtiene el valor en los P.G.
                c_g += local_c[c] * self.V.ref_N(c, pos)
                rho_g += local_rho[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                for b in range(n_element_nodes):
                    dNa = self.V.ref_dN_deta(a, pos)
                    dNb = self.V.ref_dN_deta(b, pos)
                    Na = self.V.ref_N(a, pos)
                    Nb = self.V.ref_N(b, pos)

                    # Add to local stiffness
                    local_A[a, b] += (k_g * (dNa / J) * (dNb / J) + (Na * Nb * c_g * rho_g)/dt )* W * J

        for a in range(n_element_nodes):
            for b in range(n_element_nodes):
                # Assemble to global stiffness
                i, j = self.V.local_to_global(e, a), self.V.local_to_global(e, b)
                self.A[i, j] += local_A[a, b]

        return

    @override
    def assemble_b(self, dt):
        # (C*T^(n-1) /dt)
        self.b[:] = 0

        for e in range(self.V.n_elements):
            self.assemble_element_b(e, dt)

        return

    @override
    def assemble_element_b(self, e, dt):
        # Compute
        n_element_nodes = self.V.n_element_nodes
        n_quad = self.V.quad_rule.n_quad
        # Local LeftSide vector
        local_b = np.zeros(n_element_nodes)
        # Local heat source/sink vector
        local_Q = self.V.localize(e, self.Q)
        local_rho = self.V.localize(e, self.rho)
        local_c = self.V.localize(e, self.c)
        # Local u
        local_u = self.V.localize(e, self.u)

        for g in range(n_quad):
            # Quadrature data
            pos = self.V.quad_rule.position[g]
            W = self.V.quad_rule.weight[g]
            # Jacobian
            J = self.V.J(e, pos)

            # Heat source/sink
            Q_g = 0.0
            # Heat capacity
            c_g = 0.0
            # Material density
            rho_g = 0.0
            # Previous temperature
            u_g = 0.0
            for c in range(n_element_nodes):
                Q_g += local_Q[c] * self.V.ref_N(c, pos)
                c_g += local_c[c] * self.V.ref_N(c, pos)
                rho_g += local_rho[c] * self.V.ref_N(c, pos)
                u_g += local_u[c] * self.V.ref_N(c, pos)

            for a in range(n_element_nodes):
                Na = self.V.ref_N(a, pos)

                # Add to local LS
                local_b[a] += ((Na * Q_g) + (rho_g * c_g * Na * u_g / dt)) * W * J

        for a in range(n_element_nodes):
            # Assemble to global LS
            i = self.V.local_to_global(e, a)
            self.b[i] += local_b[a]

        return

    def stable_dt(self):

        pass

    @override
    def apply_BCs(self):
        # Dirichlet BCs
        dirichlet_nodes = np.where(self.BC_types == 1)[0]

        for i in dirichlet_nodes:
            u_bar = self.BC_values[i]

            # Subtract the i-th column of A times u_bar from b
            self.b -= self.A[:, i] * u_bar

            # Zero the i-th row and column of A
            self.A[i, :] = 0.0
            self.A[:, i] = 0.0

            # Set the diagonal to 1
            self.A[i, i] = 1.0

            # Set b[i] to u_bar
            self.b[i] = u_bar

        # Neumann BCs
        neumann_nodes = np.where(self.BC_types == 0)[0]

        for i in neumann_nodes:
            self.b[i] += self.BC_values[i]


    @override
    def apply_initial_conditions(self, u):
        self.u = u


# -----------------------------------------------------------------------------
# 4) LinearSolver - wraps NumPy's linear solver
# -----------------------------------------------------------------------------
class Solver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def step(self, dt):
        pass


class SteadyStateSolver(Solver):
    def __init__(self, system):
        self.system = system

    @override
    def solve(self):
        # Assemble the stiffness matrix
        self.system.assemble_stiffness()
        # Assemble the RHS vector
        self.system.assemble_rhs()
        # Apply the BCs
        self.system.apply_BCs()

        # And solve the system of linear equations
        # TODO: implement a CG solver that utilizes the skyline storage scheme
        self.system.u = np.linalg.solve(self.system.K, self.system.q)

        return

    @override
    def step(self, dt):
        pass


# TODO: implement this
class ForwardEulerSolver(Solver):
    def __init__(self, system):
        self.system = system
        # Assemble the capacity matrix
        self.system.assemble_capacity()

    @override
    def solve(self):
        pass

    @override
    def step(self, dt):
        # Assemble the residual vector
        self.system.assemble_residual()

        res = self.system.res

        # nodos Neumann
        for i in range(self.system.V.n_nodes):
            if self.system.BC_types[i] == 0:
                res[i] += self.system.BC_values[i]

        # Matriz de masa lumpeada, calculada una sola vez al crear este objeto
        C_lumped = self.system.capacity_lumped

        for i in range(self.system.V.n_nodes):
            self.system.u[i] += (dt / C_lumped[i]) * res[i]

        for i in range(self.system.V.n_nodes):
            # nodos Dirichlet
            if self.system.BC_types[i] == 1:
                self.system.u[i] = self.system.BC_values[i]

        return


# TODO: implement this
class BackwardEulerSolver(Solver):
    def __init__(self,system):
        self.system = system

    @override
    def solve(self):
        pass

    @override
    def step(self, dt):
        self.system.assemble_a(dt)
        self.system.assemble_b(dt)
        # Apply the BCs
        self.system.apply_BCs()

        # And solve the system of linear equations
        # TODO: implement a CG solver that utilizes the skyline storage scheme

        self.system.u = np.linalg.solve(self.system.A, self.system.b)

        pass


def run_steady_state():
    # Definition of the geometry
    domain_length = 1.0
    n_nodes = 11

    # STEP 1: create geometry + basis
    x_coords = np.linspace(0.0, domain_length, n_nodes)  # np.linspace(start, stop, N)

    mesh = Mesh1D(x_coords)
    # modifiable (P1, P2, etc.)
    V = FunctionSpaceSeg2(mesh)

    # Print some info
    print("Elements and nodal coordinates")
    for elem in V.elements:
        print(elem)
        for node in elem:
            print(V.nodal_coordinates[node])

    # STEP 2: define the system
    # Uniform conductivity, modifiable
    k = np.zeros(V.n_nodes)
    k[:] = 1.0e-14
    # Uniform heat capacity
    c = np.zeros(V.n_nodes)
    c[:] = 1.0
    # Uniform density
    rho = np.zeros(V.n_nodes)
    rho[:] = 1.0
    # Uniform heat source/sink, modifiable
    Q = np.zeros(V.n_nodes)
    Q[:] = 1.0e-13

    # Construct the system, modifiable (transient, etc.)
    system = SteadyStateHeatTransferSystem(V, k, Q)

    # Boundary conditions, modifiable
    # Dirichlet BCs on the left
    system.BC_types[0] = 1
    # Dirichlet on the right
    system.BC_types[V.n_nodes - 1] = 1
    # Of values 0
    system.BC_values[0] = 0
    system.BC_values[V.n_nodes - 1] = 0

    # STEP 3: assemble and solve
    # Solver type is also modifiable
    # solver = SteadyStateSolver(system)
    # solver.solve()

    # Print some info
    print("Stiffness matrix K")
    print(system.K)
    print("RHS vector q")
    print(system.q)
    print("Solution vector u")
    print(system.u)

    # STEP 4: plot the solution
    plt.rcParams["font.size"] = 16
    plt.figure()

    # FEM solution
    plt.plot(V.nodal_coordinates, system.u, marker="o", linestyle="--", markersize=8, label="FEM")

    # Analytical solution
    analytical = [-5 * x * x + 8 * x for x in V.nodal_coordinates]
    plt.plot(V.nodal_coordinates, analytical, linestyle="-", linewidth=2, label="Analytical")

    plt.xlabel("$x$ [m]")
    plt.ylabel(r"Temperature $u$ [°C]")
    plt.title("1D Steady-State Temperature Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def run_transient_explicit():
    # Definition of the geometry
    domain_length = 1.0
    n_nodes = 11

    # STEP 1: create geometry + basis
    x_coords = np.linspace(0.0, domain_length, n_nodes)  # np.linspace(start, stop, N)

    mesh = Mesh1D(x_coords)
    # modifiable (P1, P2, etc.)
    V = FunctionSpaceSeg2(mesh)

    # Print some info
    print("Elements and nodal coordinates")
    for elem in V.elements:
        print(elem)
        for node in elem:
            print(V.nodal_coordinates[node])

    # STEP 2: define the system
    # Uniform conductivity, modifiable
    k = np.zeros(V.n_nodes)
    k[:] = 15
    # Uniform heat capacity
    c = np.zeros(V.n_nodes)
    c[:] = 500
    # Uniform density
    rho = np.zeros(V.n_nodes)
    rho[:] = 8000

    # Uniform heat source/sink, modifiable
    Q = np.zeros(V.n_nodes)
    Q[:] = 1.0e-13

    # Construct the system, modifiable (transient, etc.)
    system = TransientHeatTransferSystem(V, k, c, rho, Q)

    # Boundary conditions, modifiable
    # Dirichlet BCs on the left
    system.BC_types[0] = 1
    # Dirichlet on the right
    system.BC_types[V.n_nodes - 1] = 0
    # Of values 0
    system.BC_values[0] = 0
    system.BC_values[V.n_nodes - 1] = -2.0e-14

    # STEP 3: assemble and solve
    # Solver type is also modifiable
    solver = ForwardEulerSolver(system)
    # Initial conditions
    u_0 = np.zeros(V.n_nodes)
    u_0[:] = 1
    system.apply_initial_conditions(u_0)

    # Diffusivity
    alpha = k[0] / (c[0] * rho[0])
    # Delta x
    dx = domain_length / (n_nodes - 1)
    # dt = CFL * 0.5 * dx^2 / alpha para elementos lineales
    # dt = CFL * 0.083 * dx² / alpha para elementos cuadraticos
    dt = 0.9 * 0.083 * dx * dx / alpha
    # Implementar esto
    # dt = system.stable_dt()

    # Step in time
    t_max = 15000
    t = 0

    while t < t_max:
        print("Solution vector u, t =", t)
        print(system.u)
        solver.step(dt)

        t += dt

    # STEP 4: plot the solution
    plt.rcParams["font.size"] = 16
    plt.figure()

    # FEM solution
    plt.plot(V.nodal_coordinates, system.u, marker="o", linestyle="--", markersize=8, label="FEM")

    # Analytical solution
    analytical = np.zeros(V.n_nodes)
    for i in range(1000):
        analytical += (
            (4 / np.pi)
            * (np.sin((2 * i + 1) * np.pi * V.nodal_coordinates / domain_length) / (2 * i + 1))
            * np.exp(-(((2 * i + 1) * np.pi / domain_length) ** 2) * alpha * t)
        )

    plt.plot(V.nodal_coordinates, analytical, linestyle="-", linewidth=2, label="Analytical")

    plt.xlabel("$x$ [m]")
    plt.ylabel(r"Temperature $u$ [°C]")
    plt.title("1D Transient-State Temperature Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

def run_transient_implicit():
    # Definition of the geometry
    domain_length = 1.0
    n_nodes = 11

    # STEP 1: create geometry + basis
    x_coords = np.linspace(0.0, domain_length, n_nodes)  # np.linspace(start, stop, N)

    mesh = Mesh1D(x_coords)
    # modifiable (P1, P2, etc.)
    V = FunctionSpaceSeg2(mesh)

    # Print some info
    print("Elements and nodal coordinates")
    for elem in V.elements:
        print(elem)
        for node in elem:
            print(V.nodal_coordinates[node])

    # STEP 2: define the system
    # Uniform conductivity, modifiable
    k = np.zeros(V.n_nodes)
    k[:] = 15
    # Uniform heat capacity
    c = np.zeros(V.n_nodes)
    c[:] = 500
    # Uniform density
    rho = np.zeros(V.n_nodes)
    rho[:] = 8000

    # Uniform heat source/sink, modifiable
    Q = np.zeros(V.n_nodes)
    Q[:] = 0.0

    # Construct the system, modifiable (transient, etc.)
    system = TransientHeatTransferSystem(V, k, c, rho, Q)

    # Boundary conditions, modifiable
    # Dirichlet BCs on the left
    system.BC_types[0] = 0
    # Dirichlet on the right
    system.BC_types[V.n_nodes - 1] = 0
    # Of values 0
    system.BC_values[0] = -10
    system.BC_values[V.n_nodes - 1] = -10.0

    # STEP 3: assemble and solve
    # Solver type is also modifiable
    solver = BackwardEulerSolver(system)
    # Initial conditions
    u_0 = np.zeros(V.n_nodes)
    u_0[:] = 1
    system.apply_initial_conditions(u_0)

    # Diffusivity
    alpha = k[0] / (c[0] * rho[0])
    # Delta x
    dx = domain_length / (n_nodes - 1)
    # dt = CFL * 0.5 * dx^2 / alpha para elementos lineales
    # dt = CFL * 0.083 * dx² / alpha para elementos cuadraticos
    dt = 0.9 * 0.083 * dx * dx / alpha
    # Implementar esto
    # dt = system.stable_dt()

    # Step in time
    t_max = 15000
    t = 0

    while t < t_max:
        print("Solution vector u, t =", t)
        print(system.u)
        solver.step(dt)

        t += dt

    # STEP 4: plot the solution
    plt.rcParams["font.size"] = 16
    plt.figure()

    # FEM solution
    plt.plot(V.nodal_coordinates, system.u, marker="o", linestyle="--", markersize=8, label="FEM")

    # Analytical solution
    analytical = np.zeros(V.n_nodes)
    for i in range(1000):
        analytical += (
            (4 / np.pi)
            * (np.sin((2 * i + 1) * np.pi * V.nodal_coordinates / domain_length) / (2 * i + 1))
            * np.exp(-(((2 * i + 1) * np.pi / domain_length) ** 2) * alpha * t)
        )

    plt.plot(V.nodal_coordinates, analytical, linestyle="-", linewidth=2, label="Analytical")

    plt.xlabel("$x$ [m]")
    plt.ylabel(r"Temperature $u$ [°C]")
    plt.title("1D Transient-State Implicit Temperature Profile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

def main():
    #run_steady_state()
    run_transient_implicit()
    #run_transient_explicit()

    return


if __name__ == "__main__":
    main()
