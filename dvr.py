"""Discrete variable representation functions

More info: TBW"""

import numpy as np


def calc_mkinetic(row_idx, col_idx, grid_point_dim, n_segments):
    """Calculates a single entry of the kinetic energy matrix.

    :row_idx: Row index
    :col_idx: Column index
    :n_segments: number of segments on a single spatial coordinate
    :grid_point_dim: Grid point dimension. Typically the product of spatial
                     dimensions and number of particles
    :return: Kinetic energy at (row_idx, col_idx)
    """
    tmp = 0.0

    # sum the contributions to matrix element from each spatial dimension
    for i in range(grid_point_dim):
        # diagonal elements
        if row_idx == col_idx:
            tmp += (((2. * n_segments**2 + 1) / 3.) -
                    np.sin(2 * np.pi / (2. * n_segments) * row_idx[i])**(-2))

        # off-diagonal elements in one dimension demand equality of all other
        # coordinates; all "fancy" version of the if clause, e.g., np.not_equal
        # np.delete, np.prod are an order of magnitude slower;
        if ((row_idx[i] != col_idx[i]) &
            (row_idx[(i + 1) % grid_point_dim] == col_idx[(
                i + 1) % grid_point_dim]) &
            (row_idx[(i + 2) % grid_point_dim] == col_idx[(
                i + 2) % grid_point_dim]) &
            (row_idx[(i + 3) % grid_point_dim] == col_idx[(
                i + 3) % grid_point_dim]) &
            (row_idx[(i + 4) % grid_point_dim] == col_idx[(
                i + 4) % grid_point_dim]) &
            (row_idx[(i + 5) % grid_point_dim] == col_idx[(
                i + 5) % grid_point_dim])):
            tmp += (-1)**(row_idx[i] - col_idx[i]) * (
                np.sin(np.pi / (2. * n_segments) * (row_idx[i] - col_idx[i]))**
                (-2) - np.sin(np.pi / (2. * n_segments) *
                              (row_idx[i] + col_idx[i]))**(-2))
    return tmp


def calc_mpotential(row_idx, col_idx, space_dims, grid_spacing, lec, beta):
    """Calculates a single entry of the potential energy matrix for 2-particle problems.

    FIXME: write down other assumptions >> HERE <<

    :row_idx: Row index
    :col_idx: Column index
    :space_dims: Spatial dimensionality
    :grid_spacing: Grid spacing
    :lec: Coupling strength
    :beta: Gaussian width

    :return: Potential energy at (row_idx, col_idx)
    """
    if row_idx == col_idx:
        return lec * np.exp(-beta * sum([(
            row_idx[n] - row_idx[n + space_dims] * grid_spacing)**2
                                         for n in range(space_dims)]))
    else:
        return 0.0
