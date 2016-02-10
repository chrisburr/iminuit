from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit import Minuit
from iminuit.frontends.html import HtmlFrontend
from iminuit.frontends.console import ConsoleFrontend


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def f1_grad(x, y):
    dfdx = -2 * (1 - x)
    dfdy = 200 * (y - 1)
    return [dfdx, dfdy]


def test_html():
    _test_minuit(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=HtmlFrontend()))


def test_html_gradient():
    _test_minuit(Minuit(f1, grad_fcn=f1_grad, x=0, y=0, pedantic=False, print_level=1, frontend=HtmlFrontend()))


def test_console():
    _test_minuit(Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=ConsoleFrontend()))


def test_console_gradient():
    _test_minuit(Minuit(f1, grad_fcn=f1_grad, x=0, y=0, pedantic=False, print_level=1, frontend=ConsoleFrontend()))


def _test_minuit(m):
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=HtmlFrontend())
    m.tol = 1e-4
    m.migrad()
    m.minos()
    m.print_matrix()
    m.print_initial_param()
    m.print_fmin()
    m.print_all_minos()
    m.latex_matrix()
