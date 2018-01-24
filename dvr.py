"""Discrete variable representation functions

More info: TBW"""

import numpy as np
from scipy.sparse import kronsum, csr_matrix
from itertools import product


def calc_grid(dimension, basis_specs):
    """Diagonalizes the matrix of the position operator

    :dimension: dimension of the variational basis = number of grid points
    :basis_specs: array with a string which specifies the variational basis
            as first entry and an array with basis-characteristic parameters
            as the second
            entry; e.g.
            basis_specs = ['SINE', [spatial_dim, box_size, box_origin, mass]]
            basis_specs = ['HO', [spatial_dim, x_equilibrium, mass, omega]]

    :return: the eigenvalues of the position-operator matrix and
             the transformation matrix such that diag_x = U^T * x * U 
    """

    pos_op_mat = np.zeros((dimension, dimension))

    if basis_specs[0] == 'SINE':
        pos_op_mat[0, 1] = 0.5
        pos_op_mat[dimension - 1, dimension - 2] = 0.5
        for j in range(1, dimension - 1):
            pos_op_mat[j, j + 1] = 0.5
            pos_op_mat[j, j - 1] = 0.5
        evsys = np.linalg.eigh(pos_op_mat)
        evs = basis_specs[1][2] + basis_specs[1][1] / np.pi * np.arccos(
            evsys[0])
        return [np.sort(evs), evsys[1]]

    if basis_specs[0] == 'HO':
        for j in range(dimension):
            for k in range(dimension):
                if j == k:
                    pos_op_mat[j, k] = basis_specs[1][1]
                if j == k - 1:
                    pos_op_mat[j, k] = np.sqrt(
                        (j + 1) / (2 * basis_specs[1][2] * basis_specs[1][3]))
                if j == k + 1:
                    pos_op_mat[j, k] = np.sqrt(
                        j / (2 * basis_specs[1][2] * basis_specs[1][3]))

    return np.linalg.eigh(pos_op_mat)


def calc_Ekin(dim_basis, n_part, basis_specs, eigensys_coord):
    """Calculates the kinetic energy matrix.

        1) fill 1-particle matrix for 1 Cartesian dimension
        2) fill the N-particle, D-cart-dim matrix via a double Kronecker sum

    :dim_basis: dvr basis dimension '=' nbr of grid points (for ONE dimension)
    :n_part: nbr of particles
    :basis_specs: type and parameters of the variation basis (includes spatial dimensionality)
    :eigensys_coord: eigenvectors and transformation matrix from variational to discrete basis

    :return: Kinetic energy matrix
    """
    tmp = np.zeros((dim_basis, dim_basis))
    tmp2 = np.zeros((dim_basis, dim_basis))

    # analytic DVR of the kinetic-energy operator from
    # The Journal of Chemical Physics 96, 1982 (1992), Eq. A6a/b
    if basis_specs[0] == 'SINE':
        # infer the constant grid spacing (dx) and the box length (L) from
        # the eigenvalues; as the eigenvalues do not include the edges, 2*dx
        # has to be added;

        if sum(
                np.isclose(
                    np.diff(eigensys_coord[0]),
                    np.roll(np.diff(eigensys_coord[0]),
                            1))) != len(eigensys_coord[0]) - 1:
            print(
                'Grid points of the SINE basis are not equally spaced! Exiting...'
            )
            exit()

        dx = abs(eigensys_coord[0][1] - eigensys_coord[0][0])
        L = abs(eigensys_coord[0][-1] - eigensys_coord[0][0]) + 2 * dx
        for i in range(dim_basis):
            for ip in range(dim_basis):
                i_idx = i + 1
                ip_idx = ip + 1  # to array indices (i,ip) add 1
                if i == ip:
                    tmp[i, i] = (((2. * (dim_basis + 1)**2 + 1) / 3.) -
                                 np.sin(i_idx * (np.pi /
                                                 (dim_basis + 1)))**(-2))
                else:
                    tmp[i, ip] = ((-1)**(i_idx - ip_idx)) * (
                        np.sin(np.pi / (2. * (dim_basis + 1)) *
                               (i_idx - ip_idx))**
                        (-2) - np.sin(np.pi / (2. * (dim_basis + 1)) *
                                      (i_idx + ip_idx))**(-2))
                tmp[i, ip] *= np.pi**2 / (2. * L**2)

        tmp = tmp / (2. * basis_specs[1][3])
        # this factor has to be verified!

    if basis_specs[0] == 'HO':

        for alpha in range(dim_basis):
            for beta in range(dim_basis):
                for k in range(dim_basis):
                    tmp[alpha, beta] += basis_specs[1][3] * eigensys_coord[1][
                        k, alpha] * (k + 0.5) * eigensys_coord[1][k, beta]
                if alpha == beta:
                    tmp[alpha, beta] -= 0.5 * basis_specs[1][2] * basis_specs[
                        1][3]**2 * (
                            eigensys_coord[0][alpha] - basis_specs[1][1])**2

    mEkin = csr_matrix(0)
    # the two loops represent the Kronecker sums which generate the full matrix
    # for N particles, each in D Cartesian dimensions
    for particle in range(n_part):
        for cart_dim in range(basis_specs[1][0]):
            mEkin = kronsum(mEkin, tmp)
    return mEkin


def calc_potential(n_part, space_dims, pot_specs, eigensys_coord):
    """Calculates the potential energy matrix:

    2-body Gauss: V(r1,r2) = lec * exp(-beta * (r1-r2)**2)
    N-body HO: V(r1,...,rN) = 1/2 m omega^2 (r1^2 + ... + rN^2)

    :space_dims: Spatial dimensionality
    ::

    :return: Potential energy at (row_idx, col_idx)
    """

    # calculate dimension of the Hilbert space as a direct product of
    # spaces for each particle and spatial dimension
    dim_hilber = (len(eigensys_coord[0]))**(n_part * space_dims)

    # we populate only diagonal elements (to be changed, later)
    mpotential = np.zeros((dim_hilber))

    idx = 0
    grd1D = product(eigensys_coord[0], repeat=(space_dims * n_part))

    if pot_specs[0] == 'GAUSS':
        if n_part < 2:
            print(
                'Gaussian interaction implemented for N > 1 particles! exiting...'
            )
            exit()
        for coord in grd1D:
            mpotential[idx] = pot_specs[2] * np.exp(
                -pot_specs[1] * sum([(coord[n] - coord[n + space_dims])**2
                                     for n in range(space_dims)]))
            idx += 1

    if pot_specs[0] == 'HO':
        for coord in grd1D:
            #                       MASS / HBARC   OMEGA
            mpotential[idx] = 0.5 * pot_specs[2] * pot_specs[3]**2 * sum([(
                i - pot_specs[1])**2 for i in coord])

            idx += 1

    if pot_specs[0] == 'HOINT':
        if n_part < 2:
            print(
                'HO interaction (NOT trap) implemented for N > 1 particles! exiting...'
            )
            exit()
        for coord in grd1D:
<<<<<<< Updated upstream
            mpotential[idx] = (pot_specs[1] * sum(
=======
            #                        OMEGA**2
            mpotential[idx] = (0.5 * pot_specs[1]**1 * sum(
>>>>>>> Stashed changes
                np.array([[[(coord[i + x] - coord[j + x])**2
                            for i in range(n_part)] for j in range(n_part)]
                          for x in range(space_dims)]).flatten()))
            idx += 1

    return mpotential


def calc_mhamilton(n_part, dim_space, dim_bas, spec_bas, spec_pot):
    """ Function returns the Hamilton matrix; 

        :n_part: number of particles
        :dim_space: spatial (Cartesian) dimensions
        :dim_bas: variational-basis dim = number of segments each coordinate is divided into
        :spec_pot: parameters specifying the interaction potential
        :spec_bas: parameters specifying the basis

        :return: full Hamilton matrix in D(iscrete) V(ariable) R(epresentation)
    """
    # dimension of a single coordinate point; e.g., 2D 2Part: (x1,y1,x2,y2)
    # D spatial dimensions for each of the N particles;
    dim_grdpoint = n_part * dim_space
    # each component of a grid point takes dim_bas discrete values
    # e.g. x1 \in {x_1,...,x_dim_bas} where x_1 is an eigenvalue of the position matrix
    dim_h = dim_bas**dim_grdpoint

    # initialize empty matrices (might have to be "sparsed" for larger dim. problems)
    mpotential = np.zeros((dim_h, dim_h))
    mkinetic = np.zeros((dim_h, dim_h))
    mhamilton = np.zeros((dim_h, dim_h))

    # obtain eigensystem of the position operator in the basis of choice
    # eigenvalues '=' grid points ; transformation matrix necessary for Ekin
    # STATUS: one basis for all coordinates (future: xy->HO, z->SINE)
    coordOP_evSYS = calc_grid(dim_bas, spec_bas)

    # calculate potential and kinetic-energy matrices for a chosen basis
    # STATUS: for each additional basis, the matrices need to be specified in this function!
    mkinetic = calc_Ekin(dim_bas, n_part, spec_bas, coordOP_evSYS)
    # STATUS: the potential matrix is assumed to be diagonal (future: OPE+B => potential has non-zero offdiagonal elements)
    mpotential = np.diag(
        calc_potential(n_part, dim_space, spec_pot, coordOP_evSYS))

    mhamilton = (mkinetic + mpotential)
    return mhamilton