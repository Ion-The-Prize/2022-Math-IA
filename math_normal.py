import math
from math import isclose
import random
import numpy as np


def normal_float_str(f, num_decimal_places):
    format_string = "{:."+str(num_decimal_places)+"g}"
    return format_string.format(f).rstrip('0').rstrip('.')


def fmult(a,b):
    return a*b


def fdiv(a,b):
    return a/b


def fpow(a,b):
    return pow(a,b)


def fadd(a,b):
    return a+b


def fsubt(a,b):
    return a-b


def fstr(f, num_decimal_places = 6):
    format_string = "{:."+str(num_decimal_places)+"g}"
    return format_string.format(f).rstrip('0').rstrip('.')
    # return mp.nstr(f, num_decimal_places)


def fabs(f):
    return math.fabs(f)


def wrap_float(f):
    """
    All math-relevant floats are run through here to, perhaps, upgrade them to numpy.longdouble
    """
    return np.longdouble(f)


def unwrap_float(f):
    return float(f)


def fisclose(a,b, rel_tol=None, abs_tol=None):
    if rel_tol is not None and abs_tol is not None:
        return isclose(a,b,rel_tol=rel_tol, abs_tol=abs_tol)
    if rel_tol is not None:
        return isclose(a,b, rel_tol=rel_tol)
    elif abs_tol is not None:
        return isclose(a,b, abs_tol=abs_tol)
    else:
        return isclose(a,b)


def fisgreater(a,b):
    return a>b


def fcopysign(a,b):
    return math.copysign(a,b)


def frandom(min, max):
    return fadd(min, fmult(fsubt(max, min), random.random()))
