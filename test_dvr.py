from pytest import approx

import dvr


def test_calc_mpotential_non_diag():
    assert dvr.calc_mpotential([1,2,3], [4,5,6], 3, 1, 1, 20) == 0.0


def test_calc_mpotential_diag():
    # trivial case should evaluate to lec
    assert dvr.calc_mpotential([1, 1, 1, 1],  # row_idx
                               [1, 1, 1, 1],  # col_idx
                               2,             # space_dims
                               1,             # grid spacing
                               1,             # lec
                               20             # beta (gaussian width)
                              ) == 1.0

    # TODO: write additonal test cases
    # result = dvr.calc_mpotential([1, 1, 1, 1],  # row_idx
    #                              [1, 1, 1, 1],  # col_idx
    #                              2,             # space_dims
    #                              1,             # grid spacing
    #                              1,             # lec
    #                              20             # beta (gaussian width)
    #                             ) == approx(2.3)
