from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit import Minuit
from nose.tools import (assert_almost_equal,
                        assert_less
                        )

import matplotlib as mpl

mpl.use('Agg')


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def f1_grad(x, y):
    dfdx = -2 * (1 - x)
    dfdy = 200 * (y - 1)
    return [dfdx, dfdy]


def test_mnprofile():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1), 'x')
    m.draw_mnprofile('x')


def test_mnprofile_with_grad():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, grad_fcn=f1_grad), 'x')
    m.draw_mnprofile('x')


def test_mncontour():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1))
    m.draw_mncontour('x', 'y')


def test_mncontour_with_grad():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, grad_fcn=f1_grad))
    m.draw_mncontour('x', 'y')


def test_drawcontour():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1))
    m.draw_contour('x', 'y')


def test_drawcontour_with_grad():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, grad_fcn=f1_grad))
    m.draw_contour('x', 'y')


def test_drawcontour_show_sigma():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1))
    m.draw_contour('x', 'y', show_sigma=True)


def test_drawcontour_show_sigma_with_grad():
    m = _run_minos(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, grad_fcn=f1_grad))
    m.draw_contour('x', 'y', show_sigma=True)


def _run_minos(m, variable=None):
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)
    m.minos(variable)
    return m
