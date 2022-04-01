import math
import random
import sympy as sy

def normal_float_str(f, num_decimal_places):
    format_string = "{:."+str(num_decimal_places)+"g}"
    return format_string.format(f).rstrip('0').rstrip('.')


def fmult(a,b):
    return sy.Mul(a,b)


def fdiv(a,b):
    return sy.Rational(a,b)


def fpow(a,b):
    return sy.Pow(a,b)


def fadd(a,b):
    return sy.Add(a,b)


def fsubt(a,b):
    return sy.Add(a, sy.Mul(-1,b))


def fstr(f, num_decimal_places = 6):
    # If there are lots of zeros after the whole number, we'll print
    # them separately so we don't lose the tiny decimals
    integer_part=round(f)
    fraction_part=float(f - integer_part)

    # if the fraction part is tiny (but not zero), separate it out
    if fraction_part == 0:
        return str(integer_part)

    # if the fraction part is tiny tiny
    if integer_part != 0 and math.fabs(fraction_part) < 0.0001:
        if fraction_part > 0:
            # need to add + sign to output
            result = "({}+{:.6e})".format(integer_part, fraction_part)
        else:
            # minus sign is already a part of fraction_part
            result = "({}{:.6e})".format(integer_part, fraction_part)
    else:
        result = "{:.6g}".format(float(f))
    # format_string = "{:."+str(num_decimal_places)+"g}"
    # return format_string.format(f).rstrip('0').rstrip('.')
    # return mp.nstr(f, num_decimal_places)
    result = result.rstrip('0').rstrip('.')
    if len(result)==0:
        return "0"
    else:
        return result


def fabs(f):
    return sy.Abs(f)


def wrap_float(f):
    """
    All math-relevant floats are run through here to, perhaps, upgrade them to numpy.longdouble
    """
    # return np.longdouble(f)
    return sy.Float(f,17)


def unwrap_float(f):
    return float(f)


def fisclose(a,b, rel_tol=None, abs_tol=None):
    diff = fabs(fsubt(a,b))

    if rel_tol is not None:
        return not fisgreater(sy.simplify(diff), rel_tol)
    elif abs_tol is not None:
        return not fisgreater(sy.simplify(diff), abs_tol)
    else:
        return not fisgreater(sy.simplify(diff), 1e-9)

    # if rel_tol is not None and abs_tol is not None:
    #   return isclose(a,b,rel_tol=rel_tol, abs_tol=abs_tol)
    # elif rel_tol is not None:
    #     return isclose(a,b, rel_tol=rel_tol)
    # elif abs_tol is not None:
    #     return isclose(a,b, abs_tol=abs_tol)
    # else:
    #     return isclose(a,b)


def fisgreater(a,b):
    return sy.GreaterThan(a,b)
    # return a>b


def fcopysign(a,b):
    return sy.Mul(sy.sign(a), b)
    # return math.copysign(a,b)


def frandom(min, max):
    return fadd(min, fmult(fsubt(max, min), random.random()))

