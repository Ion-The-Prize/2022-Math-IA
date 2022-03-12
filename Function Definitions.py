# imports
import math

import barcode
from barcode import *
import random
import pandas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import longdouble
from math import isclose

import mplcursors

# constants
BUILD_BINOMIAL_RANGE = 10


def floatToString(f, format_string="{:.6f}"):
    return format_string.format(f).rstrip('0').rstrip('.')


def wrap_float(f):
    """
    All math-relevant floats are run through here to, perhaps, upgrade them to numpy.longdouble
    """
    return longdouble(f)


class CalculatedRoot:
    """A calculated root"""

    def __init__(self , x_value , y_value , steps_taken , starting_guess , guess_history = None, root_was_found = True, failure_reason = None):
        self.x_value = x_value
        self.y_value = y_value
        self.steps_taken = steps_taken
        self.guess_history = guess_history
        self.starting_guess = starting_guess
        self.root_was_found = root_was_found
        self.failure_reason = failure_reason
        self.associated_real_root = None
        self.x_error = None

    def __repr__(self):
        if self.root_was_found:
            return "x={:.3f} {} (#{:d} from {:.3f})".format(
                self.x_value ,
                "err={:g}".format(self.x_error) if self.x_error is not None else "" ,
                self.steps_taken ,
                self.starting_guess)
        else:
            return "FAIL ({}): (x,y)=({:.3f}, {:.3e})(#{:d} from {:.3f})".format(self.failure_reason, self.x_value , self.y_value , self.steps_taken , self.starting_guess)

    def was_found(self):
        return self.root_was_found

    def associate_with_real_root(self, real_root):
        self.associated_real_root = real_root
        self.x_error = self.associated_real_root - self.x_value


class Polynomial:
    """A polynomial"""

    def __init__(self , poly_coefficients_list , poly_roots = None):
        """

        :param poly_coefficients_list:
        :type poly_coefficients_list: list
        :param poly_roots:
        :type poly_roots: list
        """
        self.poly_coefficients_list = [wrap_float(c) for c in poly_coefficients_list]
        if poly_roots is not None:
            self.poly_roots = [wrap_float(r) for r in poly_roots]
        else:
            self.poly_roots = None
        self.poly_degree = len(poly_coefficients_list) - 1

    def __eq__(self, other):
        return self.poly_coefficients_list == other.poly_coefficients_list

    def __repr__(self):
        return self.poly_printer() + '::' + str(self.poly_coefficients_list)

    def poly_printer(self , desmos_format = False, coeff_format = None):
        """
        Given a Polynomial, returns that polynomial in math language form
        (e.g. given [-1, 5, 10, 10, -5, 1] and [1], returns x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 )

        :param desmos_format: whether the output will be copy-able into Desmos (default to standard math format)
        :type desmos_format: bool
        :return: string of math language polynomial
        """

        note=""
        if self.get_degree() == 2:
            b2_4ac = math.pow(self.poly_coefficients_list[1] , 2) - 4 * self.poly_coefficients_list[2] * self.poly_coefficients_list[0]
            if b2_4ac < 0:
                note = " [imaginary]"

        result = ""
        reverse_poly_coefficients_list = list(reversed(self.poly_coefficients_list))
        for i in range(len(reverse_poly_coefficients_list)):
            coefficient = reverse_poly_coefficients_list[i]
            power = self.poly_degree - i
            if isclose(coefficient, 0):
                pass
            else:
                if len(result) != 0:
                    if coefficient > 0:
                        result += " + "
                    else:
                        result += " - "
                else:
                    if coefficient < 0:
                        result += "-"
                if isclose(abs(coefficient), 1) and power != 0:
                    coefficient_string = ""
                else:
                    if coeff_format is not None:
                        coefficient_string = coeff_format.format(abs(coefficient))
                    else:
                        coefficient_string = floatToString(abs(coefficient))

                if power == 0:
                    result += "{}".format(coefficient_string)  # formats coeff as number
                elif power == 1:
                    result += "{}x".format(coefficient_string)
                else:
                    if not desmos_format:
                        result += "{}x^{:d}".format(coefficient_string , power)
                    else:
                        result += "{}x^{{{:d}}}".format(coefficient_string , power)
                        # To paste into desmos, the single curly braces are changed into triple curly braces
        return result + note

    def get_degree(self):
        """Returns int degree of the polynomial"""

        return self.poly_degree

    def multiply(self , polynomial_second = None):
        """
        Multiplies two polynomials' coefficients with each other.

        :param polynomial_second: coefficients of a second polynomial (default is [1] (function returns polynomial_first))
        :type polynomial_second: Polynomial
        :returns: new Polynomial (the multiplication product)
        """
        """
        Vars:
            result_degree: degree of result (coefficient is of x w/ degree of list pos)
            sec_poly_pos: iterator current coefficient of second polynomial (list pos is coefficient of x w/ same degree)
            first_poly_pos: iterator current coefficient of self (list pos is coefficient of x w/ same degree)
            result_roots: the roots of the result polynomial (roots of self + roots of polynomial_second)
            result_a: list result of (sec_poly_pos'th coefficient of polynomial_second) * (all of self)
            result_coefficients_list: running total of all result_a's
        """

        if polynomial_second is None:
            polynomial_second = Polynomial([1] , [])

        result_degree = self.poly_degree + polynomial_second.get_degree()
        result_coefficients_list = [0] * (result_degree + 1)  # read docstring
        # make result lists long enough to avoid IndexError: list assignment index out of range
        if self.poly_roots is not None and polynomial_second.poly_roots is not None:
            result_roots = self.poly_roots + polynomial_second.poly_roots
        else:
            result_roots = None

        for first_poly_pos in range(len(self.poly_coefficients_list)):
            # distribute current component of poly_a into poly_b
            result_a = [0] * (polynomial_second.get_degree() + 1)  # read docstring
            for sec_poly_pos in range(len(polynomial_second.poly_coefficients_list)):
                result_a[sec_poly_pos] = polynomial_second.poly_coefficients_list[sec_poly_pos] * self.poly_coefficients_list[first_poly_pos]
                # result_a = product of the sec_poly_pos'th coefficient of polynomial_second and every digit of polynomial_first
            for i in range(len(result_a)):
                # add result_a to the currect part of running total
                result_coefficients_list[first_poly_pos + i] += result_a[i]
        return Polynomial(result_coefficients_list , result_roots)

    def add(self , *other_polynomials):
        """Adds polynomials, returns Polynomial sum"""

        result_coefficients_list = self.poly_coefficients_list.copy()

        for polynomial in other_polynomials:
            for poly_pos in range(max(len(result_coefficients_list) , len(polynomial.poly_coefficients_list))):
                if len(result_coefficients_list) <= poly_pos:
                    result_coefficients_list += [wrap_float(0)]
                b = polynomial.poly_coefficients_list[poly_pos] if len(polynomial.poly_coefficients_list) > poly_pos else wrap_float(0)
                result_coefficients_list[poly_pos] += b
        return Polynomial(result_coefficients_list)

    def subtract(self , subtracting_polynomial = None):
        """Subtracts subtracting_polynomial from self"""
        if subtracting_polynomial is None:
            subtracting_polynomial = Polynomial([0])

        result_coefficients_list = []

        for poly_pos in range(max(len(self.poly_coefficients_list) , len(subtracting_polynomial.poly_coefficients_list))):
            a = self.poly_coefficients_list[poly_pos] if len(self.poly_coefficients_list) > poly_pos else wrap_float(0)
            b = subtracting_polynomial.poly_coefficients_list[poly_pos] if len(subtracting_polynomial.poly_coefficients_list) > poly_pos else 0
            result_coefficients_list += [a - b]
        while len(result_coefficients_list) > 1 and isclose(result_coefficients_list[-1], 0):
            del result_coefficients_list[-1]
        return Polynomial(result_coefficients_list)

    def divide(self , divisor = None):
        """
        Divides a self by another polynomial.

        :param divisor: coefficients of a second polynomial (the divisor) (default is [1] (function returns dividend_polynomial))
        :type divisor: Polynomial
        :return: tuple[Polynomial , Polynomial] Polynomial division quotient & remainder Polynomial
        """

        if divisor is None:
            divisor = Polynomial([1] , [])

        result = build_a_monomial(0.0 , 0)
        remainder_polynomial = Polynomial(self.poly_coefficients_list)
        while (remainder_polynomial.get_degree()) >= divisor.get_degree():
            leading_term_coeff = remainder_polynomial.poly_coefficients_list[-1]
            leading_term_degree = remainder_polynomial.get_degree()
            partial_quotient_coeff = float(leading_term_coeff / divisor.poly_coefficients_list[-1])  # worried about mixing int and float (despite doing it everywhere)
            partial_quotient_degree = leading_term_degree - divisor.get_degree()
            partial_quotient_poly = build_a_monomial(partial_quotient_coeff , partial_quotient_degree)
            result = result.add(partial_quotient_poly)

            poly_to_subtract = divisor.multiply(partial_quotient_poly)
            remainder_polynomial = remainder_polynomial.subtract(poly_to_subtract)
        return result , remainder_polynomial

    def evaluate(self , x ):
        """
        Given an x value, computes y value of self

        :param x: x value
        :type x: float
        Returns: float value of polynomial with coefficients [poly_coefficients] at x value
        """

        x = wrap_float(x)
        result = wrap_float(0.0)

        for i in range(len(self.poly_coefficients_list)):
            result += self.poly_coefficients_list[i] * (x ** i)
        return result

    def evaluate_array(self, x):
        y = np.empty(shape=(len(x), 1), dtype=float)
        for i in range(len(x)):
            y[i] = self.evaluate(x[i])

        return y

    def poly_primer(self):
        """
        Differentiates a polynomial

        :returns: Polynomial derivative of input polynomial
        """

        result_coefficients_list = []

        for i in range(1 , len(self.poly_coefficients_list)):
            result_coefficients_list.append(self.poly_coefficients_list[i] * i)
        return Polynomial(result_coefficients_list)

    def get_linear_root(self):
        """Returns x-intercept of a LINEAR EQUATION"""

        assert len(self.poly_coefficients_list) == 2
        if self.poly_coefficients_list[1] == 0:
            return None

        root = (-1.0 * self.poly_coefficients_list[0]) / self.poly_coefficients_list[1]
        return root

    def get_tangent_line(self , x):
        """
        Invokes evaluate & poly_primer to generate a tangent line

        :param x: x value
        :type x: float
        :returns: binomial Polynomial tangent line
        """

        slope = self.poly_primer().evaluate(x)
        y_intercept = slope * (0 - x) + self.evaluate(x)  # plug x = 0 into y = m(x - x1) + y1
        tangent_line = Polynomial([y_intercept , slope])  # representing tangent eq as my standard polynomial format
        x_intercept = tangent_line.get_linear_root()

        if x_intercept is None:
            return tangent_line  # (without roots)
        else:
            return Polynomial([y_intercept , slope] , [x_intercept])

    def get_relative_extrema(self):
        """
        Given a Polynomial with degree >1, returns tuple lists of x values of relative maxima and minima

        :return: tuple list of relative maxima , relative minima (e.g. max @ x = -4.0, min @ x = 4.0 is [-4.0] [4.0]
        :raises NotPoly: if a monomial or binomial is entered, returns None
        """

        first_derivative = self.poly_primer()
        first_derivative_zeros = []
        second_derivative = first_derivative.poly_primer()

        # Find first derivative zeroes somehow (maybe by finding GCF of coeff and pulling that out then looping??)

        rel_maxima = []
        rel_minima = []

        for i in range(self.get_degree()):
            if second_derivative.evaluate(first_derivative_zeros[i]) > 0:
                rel_maxima += [i]
            elif second_derivative.evaluate(first_derivative_zeros[i]) < 0:
                rel_minima += [i]
            elif not second_derivative.get_degree():
                raise NotPoly
        return rel_maxima , rel_minima

    def get_newton_root_from_point(self , starting_x , max_steps = 10 , epsilon = 1e-8 , debug = False , minimum_adjustment = None):
        """
        Performs Newton's method for finding roots at a given x-value

        :param starting_x: the x-value that Newton's method will be performed on
        :param max_steps: maximum steps before a result is considered failed
        :param epsilon: really small value that the y-value of a root approximation needs to be below for the root approximation to be considered an actual root
        :param debug: whether debug strings will be printed
        :param minimum_adjustment: If a value is given, this will be the minimum delta x-value for each step
        :type minimum_adjustment: float
        :return: CalculatedRoot
        """

        guess_history=[]
        epsilon = wrap_float(epsilon)
        current_guess = wrap_float(starting_x)
        current_value = self.evaluate(current_guess)
        guess_history.append(current_guess)
        step_number = 0
        while not isclose(current_value , 0 , abs_tol = epsilon):
            step_number += 1
            if step_number > max_steps:
                break
            previous_guess = current_guess
            previous_value = current_value

            new_guess_poly = self.get_tangent_line(current_guess)
            if new_guess_poly.poly_roots is None:
                return CalculatedRoot(current_guess , current_value , step_number , starting_x , guess_history = guess_history, root_was_found = False, failure_reason='zero derivative')
            new_guess = new_guess_poly.poly_roots[0]  # new_guess = x_intercept of tangent line

            # default isclose value is 1e-9 (a billionth)
            if isclose(current_guess, new_guess):
                if minimum_adjustment is not None:
                    # force a nudge in the right direction
                    new_guess = current_guess + math.copysign(minimum_adjustment , new_guess - current_guess)
                else:
                    print("Failed to make progress on finding root after {} steps. Search ended at x={:.5e} where y={:.5e}. Last update was {:.5e}".format(
                        step_number, current_guess, current_value, current_guess-new_guess))

                    return CalculatedRoot(current_guess, current_value, step_number, starting_x,
                                          guess_history = guess_history, root_was_found = False,
                                          failure_reason = 'no progress')
            current_guess = new_guess
            current_value = self.evaluate(current_guess)
            guess_history.append(current_guess)
            if debug or (self.get_degree() == 1 and step_number == 5):
                # If the degree is 1, it should have converged long before getting to step 5
                print("Updating guess for {} time: from ({:.5g} , {:.5g}) to ({:.5g} , {:.5g}) [delta_x={:e}] :: was notclose[{:e}]:: poly={} tangent={}"
                      .format(step_number, previous_guess, previous_value, current_guess, current_value, current_guess-previous_guess, epsilon,  self, new_guess_poly.poly_printer(coeff_format = "{}")))
        if isclose(current_value, 0, abs_tol = epsilon):
            if debug:
                print("Found root after {} steps at x={:.5e} where y={:.5e}".format(
                    step_number, current_guess, current_value))
            return CalculatedRoot(current_guess , current_value , step_number , starting_x, guess_history = guess_history)
        else:
            if debug:
                print("Failed to find root after {} steps. Search ended at x={:.5e} where y={:.5e}".format(
                    step_number, current_guess, current_value))
            return CalculatedRoot(current_guess , current_value , step_number , starting_x , guess_history = guess_history, root_was_found = False, failure_reason = 'too many steps')

    def get_roots(self , max_steps = 20 , epsilon = 1e-8 , starting_guess_count = None ,
                  random_starting_guesses = True , guess_range_min = -BUILD_BINOMIAL_RANGE - 1,
                  guess_range_max = BUILD_BINOMIAL_RANGE + 1 , sort_roots = False):
        """
        Uses Newton's method for finding roots of higher order polynomials.

        :param max_steps: the number of tangent lines used (default 20)
        :type max_steps: int
        :param starting_guess_count: the number of starting starting_guesses (default is degree of inputted polynomial)
        :type starting_guess_count: int
        :param epsilon: really small value, only roots with y values below epsilon will be counted as actual roots
        :type epsilon: float
        :param random_starting_guesses: if the starting guesses are randomly distributed within the root range or evenly spaced (default true)
        :type random_starting_guesses: bool
        :param guess_range_min: interval of guessing ("root range") minimum
        :type guess_range_min: float
        :param guess_range_max: interval of guessing ("root range") maximum
        :type guess_range_max: float
        :param sort_roots: whether the final roots string will be sorted from most negative to most positive (default False)
        :type sort_roots: bool
        :return: tuple[list[CalculatedRoot (successful roots)] , list[CalculatedRoot (failed roots)]]
        """

        if starting_guess_count is None:
            starting_guess_count = self.get_degree()

        # will be subbed in for the farthest relative extrema once that's built
        starting_guesses = []
        if not random_starting_guesses:
            guess_increment = (guess_range_max - guess_range_min) / starting_guess_count
            for i in range(starting_guess_count):
                starting_guesses.append(guess_range_min + (i * guess_increment))
        else:
            for i in range(starting_guess_count):
                starting_guesses.append(guess_range_min + (guess_range_max - guess_range_min) * random.random())  # generate a random float in the range

        poly_roots = []
        failed_roots = []
        for i in range(starting_guess_count):
            newton_result = self.get_newton_root_from_point(starting_guesses[i] , max_steps , epsilon)
            if newton_result.root_was_found:
                poly_roots += [newton_result]
            else:
                failed_roots += [newton_result]
        if sort_roots:
            poly_roots.sort()
        return poly_roots , failed_roots

    def get_roots_with_dividing(self , max_steps_per_root = 20 , max_steps_before_quitting = None, epsilon = 1e-10 , guess_range_min = -BUILD_BINOMIAL_RANGE - 1,
                  guess_range_max = BUILD_BINOMIAL_RANGE + 1, sort_roots = False , human_dividing = False):
        """
        Uses Newton's method for finding roots of higher order polynomials. After finding a root, it is factored out
            of the polynomial, resulting in all the actual roots with no extra repeats
            (will only factor out roots with y values below epsilon)

        :param max_steps_per_root: the number of tangent lines used (default 20)
        :type max_steps_per_root: int
        :param max_steps_before_quitting: the number of times it should loop without finding a root with y below epsilon
            before giving up, defaults to degree^2. If 0, will keep trying until a root below epsilon is found
        :type max_steps_before_quitting: int
        :param epsilon: really small value, only roots with y values below epsilon will be counted as actual roots. If 0, will go up until max_steps_per_root
        :type epsilon: float
        :param guess_range_min: interval of guessing ("root range") minimum
        :type guess_range_min: float
        :param guess_range_max: interval of guessing ("root range") maximum
        :type guess_range_max: float
        :param sort_roots: whether the final roots string will be sorted from most negative to most positive (default False)
        :type sort_roots: bool
        :param human_dividing: whether the factored out roots will be the actual root (True) or the calculated one (False)
        :return: triple[list[CalculatedRoot (successful roots)] , list[CalculatedRoot (failed roots)] , list[Polynomial (remainders)]]
        """

        if max_steps_before_quitting is None:
            max_steps_before_quitting = self.poly_degree ** 2

        real_roots_left = self.poly_roots
        poly_roots = []
        failed_roots = []
        remainders = []
        factored_poly = self
        loops = 0
        while factored_poly.poly_degree > 0:
            guess = guess_range_min + (guess_range_max - guess_range_min) * random.random()
            newton_result = factored_poly.get_newton_root_from_point(guess , max_steps_per_root , epsilon)
            if newton_result.root_was_found:
                found_root = newton_result.x_value
                if self.poly_roots is not None:
                    matching_real_root = None
                    for real_root in real_roots_left:
                        # if we haven't found a matching real root yet or if this real_root is closer than the one we already matched
                        if matching_real_root is None or abs(real_root - found_root) < abs(matching_real_root - found_root):
                            matching_real_root = real_root
                    # print("Found root {:.3f} is closest to real root {:.3f} [diff={:g}]".format(found_root, matching_real_root, (matching_real_root-found_root)))
                    newton_result.associate_with_real_root(matching_real_root)
                    real_roots_left.remove(matching_real_root)

                    factor = make_polynomial_from_coefficients(-1.0 * newton_result.associated_real_root , 1.0) if human_dividing else make_polynomial_from_coefficients(-1.0 * newton_result.x_value , 1.0)
                else:
                    factor = make_polynomial_from_coefficients(-1.0 * newton_result.x_value , 1.0)
                factored_poly , factor_remainder = factored_poly.divide(factor)  # remainder's only returned; never used
                # print("Factor         " , factor.poly_printer())
                # print("Remaining Poly:" , factored_poly.poly_printer())
                # print("Remainder:     " , factor_remainder.poly_printer())
                poly_roots.append(newton_result)
                remainders.append(factor_remainder)  # store remainder
                loops = 0
            else:
                failed_roots += [newton_result]
                loops += 1
            if loops >= max_steps_before_quitting != 0:
                print("{}Could not find factorable root within {:.1e} after trying {} times. {}".format(
                     "LINEAR!!" if factored_poly.get_degree() == 1 else "",
                     epsilon , loops , factored_poly.poly_printer(coeff_format="{}")))
                print("  Original poly: {}. All roots: {}".format(self, self.poly_roots))
                print("  Roots found so far: {}".format(" || ".join(str(r) for r in poly_roots)))
                print("  Failed root[0]: ", failed_roots[0])
                print("  Failed root[0] guess history: ",['{:.3e}'.format(g) for g in failed_roots[0].guess_history])
                print("==")

                break
        if sort_roots:
            poly_roots.sort()
        return len(poly_roots) == self.get_degree() , poly_roots , failed_roots , remainders

    def poly_power(self , power , pascal = 0):
        """
        Raises a given polynomial to a specified power

        :param power: the power that the polynomial will be raised to
        :type power: int
        :param pascal: whether or not every step should be printed (giving pascal's triangle) (default no) (!= 0 is yes)
        :type pascal: any
        :return: list coefficients for polynomial
        """

        assert(power >= 0)

        current_answer = Polynomial([1] , [])

        for i in range(power):
            current_answer = current_answer.multiply(self)
            if pascal != 0:
                print(i ,
                      current_answer.poly_coefficients_list)  # numbers rows of pascal's triangle as degree of poly from a binomial (so [1 2 1] is row 2)
        return current_answer

    def open_barcode_window(self, minimum, maximum, window_width, epsilon=1e-8):
        """

        :param minimum:
        :type minimum: float
        :param maximum:
        :type maximum: float
        :param window_width:
        :param epsilon:
        :type epsilon: float
        :return:
        """

        assert (maximum > minimum)
        poly_barcode = BarCode("{:s} | Roots @x = {:s}".format(self.poly_printer(), ",".join(
            ["{:.3f}".format(r) for r in sorted(self.poly_roots)]))
                               , minimum, maximum, window_width, 200, self.poly_degree + 1)
        poly_barcode.close_on_click()

        self.fill_barcode(poly_barcode, epsilon)
        return poly_barcode

    def fill_barcode(self, poly_barcode, epsilon=1e-8):
        if self.poly_roots is not None:
            for r in self.poly_roots:
                poly_barcode.assign_color_number_to_item(r)

        i = 0
        for x in poly_barcode.get_x_range():
            i += 1
            if i % 100 == 0:
                poly_barcode.draw()
            root = self.get_newton_root_from_point(starting_x=x, max_steps=128, epsilon=epsilon)
            print("Newton Result: ", root)
            if root.root_was_found:
                # Looking for what color the root bar should be
                closest_exact_root = self.get_closest_exact_root(root.x_value)
                poly_barcode.add_bar(x=x, color_item=closest_exact_root, y=root.steps_taken)
            else:
                poly_barcode.add_bar(x=x, color=GREY)

        # Draw the actual roots with White lines
        for r in range(len(self.poly_roots)):
            poly_barcode.add_bar(x=self.poly_roots[r], color=WHITE, y=None)
        poly_barcode.draw()
        print("Done filling barcode")

    def get_closest_exact_root(self, approximate_root):
        closest_exact_root = None
        closest_epsilon = None
        assert self.poly_roots is not None

        for exact_root in self.poly_roots:
            if isclose(exact_root, approximate_root):
                return exact_root

            current_epsilon = abs(approximate_root - exact_root)
            if closest_exact_root is None or closest_epsilon > current_epsilon:
                closest_exact_root = exact_root
                closest_epsilon = current_epsilon
        # print("Closest Epsilon: {:.2e} | Closest Root: {:.2f}".format(closest_epsilon , closest_exact_root))

        return closest_exact_root


def build_a_monomial(leading_coefficient , degree):
    """
    :type leading_coefficient: float
    :type degree: int
    """

    poly_coefficients_list = [0.0] * degree
    poly_coefficients_list.append(leading_coefficient)
    if degree > 0:
        mono_roots = [0]
    else:
        mono_roots = None
    return Polynomial(poly_coefficients_list , mono_roots)


def randomly_build_a_binomial(rand_lower_bound = -BUILD_BINOMIAL_RANGE , rand_upper_bound = BUILD_BINOMIAL_RANGE , only_int_roots = False):
    """
    Randomly generates a binomial with the format ax + b, where a and b are integers
    Note: sometimes, a or b will be 0, in which case it returns a monomial

    :param rand_lower_bound: the lower bound of the coefficients
    :param rand_upper_bound: the upper bound of the coefficients
    :param only_int_roots: self-explanatory; sets a to be 1 or -1
    :type only_int_roots: bool
    :returns: Polynomial of binomial (with root)
    """
    assert(rand_lower_bound < rand_upper_bound)
    
    # note: randrange does not include the upper bound
    if only_int_roots:
        root = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        return Polynomial([root , 1] , [root])
    else:
        position = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        numerator = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        denominator = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        while denominator == 0:
            denominator = random.randrange(rand_lower_bound , rand_upper_bound + 1)

        root = position + (numerator / denominator)
        return Polynomial([(-denominator * position) + numerator , denominator] , [root])


def make_polynomial_from_coefficients(*coefficients):
    """
    Makes a Polynomial from args coefficients

    :param coefficients: coefficients for Polynomial; given in ascending order of x powers
    :type coefficients: float
    """

    poly_coefficients_list = []
    for coefficient in range(len(coefficients)):
        poly_coefficients_list.append(coefficients[coefficient])
    return Polynomial(poly_coefficients_list)


def poly_maker(degree , rand_binomial_lower = -BUILD_BINOMIAL_RANGE , rand_binomial_upper = BUILD_BINOMIAL_RANGE , only_int_roots = False):
    """
    Generates a polynomial's coefficients by randomly generating factors and then expanding those factors
    Roots are calculated using each factor

    :param degree: degree of polynomial
    :type degree: int
    :param rand_binomial_lower: lower bound of randomly generated binomials' coefficients; the a & b in: (b , ax)
    :param rand_binomial_upper: upper bound of randomly generated binomials' coefficients; the a & b in: (b , ax)
    :param only_int_roots: self explanatory; sets a in binomials to be 1 or -1
    :type only_int_roots: bool
    :returns: Polynomial class
    """

    result_poly = Polynomial([1] , [])

    for i in range(degree):
        binomial = randomly_build_a_binomial(rand_binomial_lower , rand_binomial_upper , only_int_roots = only_int_roots)
        result_poly = binomial.multiply(result_poly)

    return result_poly


def root_rounder(unrounded_poly_roots):
    """
    Rounds a lot of roots

    :type unrounded_poly_roots: list[CalculatedRoot]
    :return:
    """

    rounded_poly_roots = map(lambda root : CalculatedRoot(float("{:.3f}".format(root.x_value)) , root.y_value ,
                                                            root.steps_taken , root.starting_guess , root.root_was_found) , unrounded_poly_roots)
    return rounded_poly_roots


class ZoomPlot:
    def __init__(self, polynomial, color_points_with_newton_root=False):
        self.polynomial = polynomial
        self.colorize_points_with_newton_root = color_points_with_newton_root
        self.color_assignments = MappedColorPalette(self.polynomial.poly_roots)

        if polynomial.poly_roots is not None:
            sorted_roots = polynomial.poly_roots.copy()
            sorted_roots.sort()
            self.xmin = sorted_roots[0] - .1
            self.xmax = sorted_roots[-1] + .1

        else:
            self.xmin = -10
            self.xmax = 10

        self.orig_xmin = self.xmin
        self.orig_xmax = self.xmax

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.xpress = self.xmin
        self.xrelease = self.xmax
        self.resolution = 400
        self.maxiters = 30

        self.fig.canvas.mpl_connect('button_press_event', self.onpress)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.plot()

    def plot(self):
        print("Plotting from {:.2f} to {:.2f}".format(self.xmin, self.xmax))
        x = np.linspace(self.xmin, self.xmax, self.resolution)
        y = self.polynomial.evaluate_array(x)
        if self.colorize_points_with_newton_root:
            point_colors = []
            for x_val in x:
                # This is coloring point. We could also color the background: https://stackoverflow.com/a/9957832
                newton_result = self.polynomial.get_newton_root_from_point(x_val, max_steps=50)
                if newton_result.root_was_found:
                    exact_root = self.polynomial.get_closest_exact_root(newton_result.x_value)
                    point_colors.append(self.color_assignments.get_color(exact_root))
                else:
                    point_colors.append('gray')
        else:
            point_colors = 'red'

        self.ax.clear()
        self.ax.set_title(self.polynomial.poly_printer())
        sc=self.ax.scatter(x, y, color=point_colors, s=5)
        cursor = mplcursors.cursor(sc, hover=True)

        # hovering shows newton information
        # by default the annotation displays the xy positions
        @cursor.connect("add")
        def on_add(sel : mplcursors.Selection):
            x=sel.target[0]
            y=sel.target[1]
            print("MPL Target: ({:.3e},{:.3e})".format(x,y))
            slope = self.polynomial.poly_primer().evaluate(x)
            tangent_x_intercept = x - (y/slope)
            y_value_at_tangent_x_intercept = self.polynomial.evaluate(tangent_x_intercept)

            newton_result = self.polynomial.get_newton_root_from_point(x)

            sel.annotation.set(text="Point ({:.4g} , {:.4g})\nslope={:.3g}\ntangent leads to ({:.4g},{:.4g})\nnewton result: {}". format(
                x,y,slope, tangent_x_intercept, y_value_at_tangent_x_intercept, newton_result))
            #sel.annotation.set(text=tt[sel.target.index])


        self.ax.yaxis.grid(True)

        # Find the Zeros in the current view
        zeros_x=[]
        zeros_y=[]
        zeros_c=[]
        for zero_x in self.polynomial.poly_roots:
            if self.xmin<=zero_x<=self.xmax:
                zeros_x.append(zero_x)
                zeros_y.append(0)
                zeros_c.append(self.color_assignments.get_color(zero_x))

        #https://www.adamsmith.haus/python/answers/how-to-plot-points-in-matplotlib-in-python
        self.ax.scatter(zeros_x, zeros_y, color=zeros_c)

        #https://stackoverflow.com/a/43963231
        plt.gcf().canvas.draw_idle()

    def onpress(self, event):
        if event.button == 3:
            # Reset the original range on RIGHT click
            self.xmin = self.orig_xmin
            self.xmax = self.orig_xmax
            self.plot()

        if event.button == 1:
            self.xpress = event.xdata

    def onrelease(self, event):
        if event.button != 1: return
        self.xrelease = event.xdata

        if self.xpress is None or self.xrelease is None:
            return

        # a single click (no movement) does a newton-root in debug mode
        # (less than 1/20 of screen)
        if abs(self.xrelease - self.xpress) < 0.05*(self.xmax - self.xmin):
            self.polynomial.get_newton_root_from_point(self.xpress, debug=True)
            return

        self.xmin = min(self.xpress, self.xrelease)
        self.xmax = max(self.xpress, self.xrelease)
        self.plot()


plot = ZoomPlot(poly_maker(5), color_points_with_newton_root=True)
plt.show()

input("Press Enter to continue...")


def graph(polynomial , x_min = None , x_max = None , x_resolution = 800, y_resolution=500):
    print ("Plotting: {} (Roots: {})".format(polynomial, [floatToString(r, format_string="{:.3f}") for r in polynomial.poly_roots]))
   # plt.style.use('_mpl-gallery')

    if polynomial.poly_roots is not None:
        roots = polynomial.poly_roots.copy()
        roots.sort()
        x_min = roots[0]-.1
        x_max = roots[-1]+.1
    else:
        x_min=-10
        x_max=10
    # make data
    x = np.linspace(x_min , x_max , x_resolution)

    y = polynomial.evaluate_array(x)

    # plot
    fig , ax = plt.subplots(figsize=(11, 5))

    #ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ax.plot(x , y , linewidth = 2.0, label=polynomial.poly_printer())
    ax.set_title(polynomial.poly_printer())

    #ax.set_xlabel('x label')  # Add an x-label to the axes.
    #ax.set_ylabel('y label')  # Add a y-label to the axes.
    #ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend();  # Add a legend.

    # ax.set(xlim = (x_min , x_max) , xticks = np.arange(x_min , x_max) ,
    #        ylim = (-100 , 100) , yticks = np.arange(-100 , 100 , 10))
    #
    # ax.yaxis.grid(False)
    # ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    plt.tight_layout()
    #plt.margins(0.2)

    plt.show()


# graph(poly_maker(5))
# input("Press Enter to continue...")


class StatsAccumulator:
    def __init__(self , name):
        """
        :type name: str
        """
        self.name = name
        self.column_data_ct = []
        self.column_data_avg = []
        self.column_data_std = []
        self.column_data_min = []
        self.column_data_25 = []
        self.column_data_50 = []
        self.column_data_75 = []
        self.column_data_max = []

    def __repr__(self):
        return "{} [{:d} values]".format(self.name , len(self.column_data_ct))

    def append_data(self , data):
        self.column_data_ct.append(len(data))
        self.column_data_avg.append(np.mean(data))
        self.column_data_std.append(np.std(data))
        self.column_data_min.append(np.min(data))
        self.column_data_25.append(np.percentile(data , 25))
        self.column_data_50.append(np.median(data))
        self.column_data_75.append(np.percentile(data , 75))
        self.column_data_max.append(np.max(data))

    def add_to_dataframe(self , dataframe):
        dataframe["{} {}".format(self.name , "ct")] = self.column_data_ct
        dataframe["{} {}".format(self.name , "avg")] = self.column_data_avg
        dataframe["{} {}".format(self.name , "std")] = self.column_data_std
        dataframe["{} {}".format(self.name , "min")] = self.column_data_min
        dataframe["{} {}".format(self.name , "25")] = self.column_data_25
        dataframe["{} {}".format(self.name , "50")] = self.column_data_50
        dataframe["{} {}".format(self.name , "75")] = self.column_data_75
        dataframe["{} {}".format(self.name , "max")] = self.column_data_max


def get_data_of_poly_roots_static_accuracy(num_observations , poly_degree , epsilon , only_int_roots = False , human_dividng = False):
    result_total_steps_when_completely_successful = []
    result_total_steps_successful_guess_when_completely_successful = []
    result_percent_guesses_successful_when_completely_successful = []
    result_wasted_steps = []
    complete_fail_count = 0
    for i in range(num_observations):
        """
        # For testing when speed is important (so low sample sizes will not throw errors about arrays being empty)
        solved_completely = False
        while not solved_completely:  # carry on... this is just here for testing purposes
            polynomial = poly_maker(poly_degree, only_int_roots=only_int_roots)
            solved_completely , poly_roots , fail_roots , remainders = polynomial.get_roots_with_dividing(max_steps_per_root = 4096 , epsilon = epsilon , sort_roots = False)
        """
        polynomial = poly_maker(poly_degree , only_int_roots = only_int_roots)
        solved_completely , poly_roots , fail_roots , remainders = polynomial.get_roots_with_dividing(max_steps_per_root = 4096 , epsilon = epsilon , sort_roots = False , human_dividing = False)

        if solved_completely:
            total_steps = int(np.sum([r.steps_taken for r in poly_roots])) + int(np.sum([r.steps_taken for r in fail_roots]))
            total_steps_successful_guesses = int(np.sum([r.steps_taken for r in poly_roots]))
            percent_guesses_successful = len(poly_roots) / (len(poly_roots) + len(fail_roots))
            result_wasted_steps.append(int(np.sum([r.steps_taken for r in fail_roots])))

            result_total_steps_when_completely_successful.append(total_steps)
            result_total_steps_successful_guess_when_completely_successful.append(total_steps_successful_guesses)
            result_percent_guesses_successful_when_completely_successful.append(percent_guesses_successful)
        else:
            result_wasted_steps.append(int(np.sum([r.steps_taken for r in poly_roots])) + int(np.sum([r.steps_taken for r in fail_roots])))
            complete_fail_count += 1
            i -= 1  # retry this polynomial because it didn't work
    percent_steps_wasted = int(np.sum(result_wasted_steps)) / (int(np.sum(result_total_steps_successful_guess_when_completely_successful)) + int(np.sum(result_wasted_steps)))
    return num_observations , complete_fail_count , result_wasted_steps , percent_steps_wasted , result_total_steps_when_completely_successful , result_total_steps_successful_guess_when_completely_successful , result_percent_guesses_successful_when_completely_successful


def get_data_of_poly_roots_static_speed(num_observations , poly_degree , steps_taken , only_int_roots = False , human_dividing = False):
    result_data_absolute_error = []
    for i in range(num_observations):
        polynomial = poly_maker(poly_degree , only_int_roots = only_int_roots)
        guess = (-BUILD_BINOMIAL_RANGE - 1) + (BUILD_BINOMIAL_RANGE + 2) * random.random()
        poly_root = polynomial.get_newton_root_from_point(starting_x = guess , max_steps = steps_taken , epsilon = 0)

        result_data_absolute_error.append(math.fabs(poly_root.y_value))
    return result_data_absolute_error


def static_accuracy_chart(num_observations , poly_degrees , epsilons , only_int_roots = True , human_dividing = False):
    """
    :type num_observations: int
    :type poly_degrees: list[int]
    :type epsilons: list[float]
    :param only_int_roots: whether tested polynomials will have integer roots
    :type only_int_roots: bool
    :param human_dividing: whether Newton's method will factor out the actual roots if sufficiently close
    :type human_dividing: bool
    """

    poly_degrees.sort()
    epsilons.sort(reverse = True)

    result = dict()
    result["opfts"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [that] failed to solve
    result["opsw"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] steps [that were] wasted (not successful)
    result["assw"] = pandas.DataFrame(index = poly_degrees)  # [Regardless of solve status,] all solves [total] steps wasted
    result["sspgs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] percent [of starting] guesses [that were] successful
    result["ssts"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] total steps
    result["sssnfs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the total] steps needed for solving

    for epsilon in epsilons:
        overall_percent_failure_to_solve = []
        overall_percent_steps_wasted = []
        accumulator_all_solves_steps_wasted = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_percent_guesses_successful = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_total_steps = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_steps_needed_for_successes = StatsAccumulator("{:.1e}".format(epsilon))
        for degree in poly_degrees:
            print("Working on x^{:d} polynomials with epsilon {:e}".format(degree, epsilon))
            num_samples , complete_fail_count , wasted_steps , percent_steps_wasted , \
                tot_steps_when_completely_successful , \
                tot_steps_successful_guess_when_completely_successful , \
                percent_guesses_successful_when_completely_successful = \
                get_data_of_poly_roots_static_accuracy(num_observations = num_observations ,
                                                       poly_degree = degree ,
                                                       epsilon = epsilon ,
                                                       only_int_roots = only_int_roots)
            overall_percent_failure_to_solve.append(complete_fail_count / num_samples)
            overall_percent_steps_wasted.append(percent_steps_wasted)
            accumulator_all_solves_steps_wasted.append_data(wasted_steps)
            accumulator_successful_solves_percent_guesses_successful.append_data(percent_guesses_successful_when_completely_successful)
            accumulator_successful_solves_total_steps.append_data(tot_steps_when_completely_successful)
            accumulator_successful_solves_steps_needed_for_successes.append_data(tot_steps_successful_guess_when_completely_successful)
        result["opfts"]["{:.1e}".format(epsilon)] = overall_percent_failure_to_solve
        result["opsw"]["{:.1e}".format(epsilon)] = overall_percent_steps_wasted
        accumulator_all_solves_steps_wasted.add_to_dataframe(result["assw"])
        accumulator_successful_solves_percent_guesses_successful.add_to_dataframe(result["sspgs"])
        accumulator_successful_solves_total_steps.add_to_dataframe(result["ssts"])
        accumulator_successful_solves_steps_needed_for_successes.add_to_dataframe(result["sssnfs"])
    return result


def static_speed_chart(num_observations , poly_degrees , max_steps , only_int_roots = False , human_dividing = False):
    """
    :type num_observations: int
    :type poly_degrees: list[int]
    :type max_steps: list[int]
    :param only_int_roots: whether tested polynomials will have integer roots
    :type only_int_roots: bool
    :param human_dividing: whether Newton's method will factor out the actual roots if sufficiently close
    :type human_dividing: bool
    """

    poly_degrees.sort()
    max_steps.sort()


    result = dict()
    result["opfts"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [that] failed to solve
    result["opsw"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] steps [that were] wasted (not successful)
    result["assw"] = pandas.DataFrame(index = poly_degrees)  # [Regardless of solve status,] all solves [total] steps wasted
    result["sspgs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] percent [of starting] guesses [that were] successful
    result["ssts"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] total steps
    result["sssnfs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the total] steps needed for solving

    for epsilon in epsilons:
        overall_percent_failure_to_solve = []
        overall_percent_steps_wasted = []
        accumulator_all_solves_steps_wasted = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_percent_guesses_successful = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_total_steps = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_steps_needed_for_successes = StatsAccumulator("{:.1e}".format(epsilon))
        for degree in poly_degrees:
            print("Working on x^{:d} polynomials with epsilon {:e}".format(degree, epsilon))
            num_samples , complete_fail_count , wasted_steps , percent_steps_wasted , \
                tot_steps_when_completely_successful , \
                tot_steps_successful_guess_when_completely_successful , \
                percent_guesses_successful_when_completely_successful = \
                get_data_of_poly_roots_static_accuracy(num_observations = num_observations ,
                                                       poly_degree = degree ,
                                                       epsilon = epsilon ,
                                                       only_int_roots = only_int_roots)
            overall_percent_failure_to_solve.append(complete_fail_count / num_samples)
            overall_percent_steps_wasted.append(percent_steps_wasted)
            accumulator_all_solves_steps_wasted.append_data(wasted_steps)
            accumulator_successful_solves_percent_guesses_successful.append_data(percent_guesses_successful_when_completely_successful)
            accumulator_successful_solves_total_steps.append_data(tot_steps_when_completely_successful)
            accumulator_successful_solves_steps_needed_for_successes.append_data(tot_steps_successful_guess_when_completely_successful)
        result["opfts"]["{:.1e}".format(epsilon)] = overall_percent_failure_to_solve
        result["opsw"]["{:.1e}".format(epsilon)] = overall_percent_steps_wasted
        accumulator_all_solves_steps_wasted.add_to_dataframe(result["assw"])
        accumulator_successful_solves_percent_guesses_successful.add_to_dataframe(result["sspgs"])
        accumulator_successful_solves_total_steps.add_to_dataframe(result["ssts"])
        accumulator_successful_solves_steps_needed_for_successes.add_to_dataframe(result["sssnfs"])
    return result


def save_dataframes_to_tabs_of_file(df_dict, file_path):
    writer = pandas.ExcelWriter(file_path , engine = 'xlsxwriter')
    for sheet, dataframe in df_dict.items():
        dataframe.to_excel(writer, sheet_name = sheet, startrow = 0 , startcol = 0)
    writer.save()


sample_size = 125
poly_degrees = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]
epsilons = [1e-3 , 1e-4 , 1e-5 , 1e-6 , 1e-9 ]#, 1e-12]
static_accuracy_results = static_accuracy_chart(sample_size , poly_degrees , epsilons , False)

save_dataframes_to_tabs_of_file(static_accuracy_results, r'/Temp/Math-IA/static_accuracy_class6.xlsx')
# for sheet,df in static_accuracy_results.items():
#     df.to_excel(r'/temp/Math-IA/static_accuracy_class3.xlsx',  sheet_name = sheet, index = True, header = True)
#     iter += 1

# print(get_data_of_poly_roots_static_speed(500 , 10 , 32))
shame = Polynomial([476280.00000361796 , 1905120.0])
print(shame.get_newton_root_from_point(1))


"""
my_list = [0]
adding_list = [1 , 1]
my_binomial = build_a_binomial()
print(my_binomial)
print([0] + my_binomial)
for i in range(len(my_binomial)):
    my_binomial[i] += adding_list[i]
print(my_binomial)
for i in range(3):
    my_binomial = [0] + my_binomial
print(my_binomial)

for i in range(6):
    print(build_a_binomial())
"""


def poly_mult_test(first_poly , second_poly , answer):
    """
    Test of multiply
    Compares the product of two polynomials multiplied using multiply with WolframAlpha's result
    """
    print("mine: " , first_poly.multiply(second_poly).poly_coefficients_list)
    print("wolf: ", answer.poly_coefficients_list)

    if first_poly.multiply(second_poly) == answer:
        print("Wolfram alpha agrees")
    else:
        print("Wolfram alpha disagrees")
    return


first_mult_test_poly_a = Polynomial([-3, -1, -4, 9, -1, -1])
first_mult_test_poly_b = Polynomial([8, -3, -10, -2, -5, -8])
first_mult_test_wolfram = Polynomial([-24, 1, 1, 100, 22, -58, 23, -1, -65, 13, 8])
# (-3x^5 - 1x^4 - 4x^3 + 9x^2 - 1x - 1) * (8x^5 - 3x^4 - 10x^3 - 2x^2 - 5x - 8)
second_mult_test_poly_a = Polynomial([-1, -1, 9, -4, -1, -3])
second_mult_test_poly_b = Polynomial([-8, -5, -2, -10, -3, 8])
second_mult_test_wolfram = Polynomial([8, 13, -65, -1, 23, -58, 22, 100, 1, 1, -24])
# [-24, 1, 1, 100, 22, -58, 23, -1, -65, 13, 8]

poly_mult_test(first_mult_test_poly_a , first_mult_test_poly_b , first_mult_test_wolfram)
poly_mult_test(second_mult_test_poly_a , second_mult_test_poly_b , second_mult_test_wolfram)

add_test_poly_a = Polynomial([1 , 1 , 1])
add_test_poly_b = Polynomial([1 , 2 , 1 , 0 , 2])
add_test_poly_c = Polynomial([])
add_test_poly_d = Polynomial([7])
print("Addition Test: ", add_test_poly_a.add(add_test_poly_b , add_test_poly_c , add_test_poly_d).poly_coefficients_list)

subtract_test_poly_a = Polynomial([1 , 1 , 1])
subtract_test_poly_b = Polynomial([1 , 2 , 1 , 2])
print("Subract Test: " , subtract_test_poly_a.subtract(subtract_test_poly_b).poly_coefficients_list)

divide_test_poly_a = Polynomial([1 , 2 , 1])
divide_test_poly_b = Polynomial([1 , 1])
quotient , remainder = first_mult_test_wolfram.divide(first_mult_test_poly_b)
print("Division Test: " , quotient.poly_coefficients_list)


print("{:.2f}".format(3.14159))
testing_polynomial = poly_maker(7)
print(testing_polynomial.poly_coefficients_list)
print("Printer Test: ", testing_polynomial.poly_printer())
# print(list(reversed(testing_polynomial.poly_coefficients_list)))
# print(testing_polynomial.poly_coefficients_list)
# print(list(map(lambda num : float("{:.3f}".format(num)), testing_polynomial.poly_roots)))

print("value at x = 1: ", testing_polynomial.evaluate(1))

print(testing_polynomial.poly_primer().poly_coefficients_list)
print("prime value @ x = 1: ", testing_polynomial.poly_primer().evaluate(1))
print("tangent line @x = 1: ", testing_polynomial.get_tangent_line(1).poly_coefficients_list)
print("x-intercept of line: ", testing_polynomial.get_tangent_line(1).poly_roots)


def tangent_test(x , polynomial , answer):
    """
    Test of get_tangent_line
    Compares the x-intercept of a polynomial's tangent line found using multiply with WolframAlpha's result
    """

    print("Tangent Test")
    print("mine: ", polynomial.get_tangent_line(x).poly_coefficients_list)
    print("wolf: ", answer)

    if polynomial.get_tangent_line(x).poly_coefficients_list == answer:
        print("Wolfram alpha agrees")
    else:
        print("Wolfram alpha disagrees")
    return


# wolfram alpha input (-3 - x - 4x^2 + 9x^3 - 1x^4 - 1x^5)
tangent_test_poly = Polynomial([-3, -1, -4, 9, -1, -1])
tangent_test(1 , tangent_test_poly , [-10 , 9])


class NotPoly:
    def __init__(self , message = "Binomials/monomials not counted as polynomials"):
        self.message = message
        super().__init__(self.message)


# print("RAND TEST: ", random.random() * 10 ** 2)
for i in range(10):
    print()
print("=====================================================================================================")

new_poly = poly_maker(7)
#new_poly = Polynomial([-540000, 942000, 1270600, -3988780, 2970020, -448640, -277200, 72000] , [-3.3333333333333335, 3.3333333333333335, -0.5, 1.0, 1.2, 1.25, 0.9])
print(new_poly.poly_printer(desmos_format = True))
print(new_poly.poly_printer())
print()
print("Poly Coeff: ", new_poly.poly_coefficients_list)
print("Real Roots: ", new_poly.poly_roots)
solved_completely , calc_roots , fail_roots , remainders = new_poly.get_roots_with_dividing(max_steps_per_root = 4096 , max_steps_before_quitting = 50 , epsilon = 1e-9)
print("Calc Roots: ", [root.x_value for root in calc_roots])
print("Round Root: ", [root.x_value for root in root_rounder(calc_roots)])
print("Root Steps: ", [root.steps_taken for root in calc_roots])
print("Fail Roots: ", ' || '.join(str(r) for r in fail_roots))

"""
question_poly = poly_maker(0, -5, 5, only_int_roots = True)
print("Factor this quadratic: ", poly_printer(question_poly[0]))
print("Zeros at x = ", question_poly[1])
"""


# print(poly_printer(poly_power(10, pascal = 0)))


special_basin_poly = Polynomial([12 , -11 , -2 , 1] , [4 , 1 , -3])
input("Press Enter to continue...")
poly_barcode = BarcodePoly(special_basin_poly , -15 , 15 , 1100)
poly_barcode.await_click()


def charter(poly_coefficient_list , poly_roots , steps_needed , starting_guesses , failed_roots = None , epsilon = None , print_poly = False , print_poly_degree = False):
    """
    Puts a given polynomial and some information about that polynomial in handy chart form 
    
    :param poly_coefficient_list: List coefficients for a polynomial
    :type poly_coefficient_list: list[int]
    :param poly_roots: List roots for a polynomial
    :type poly_roots: list[float]
    :param steps_needed: List of steps Newton's method needed to get to each root
    :type steps_needed: list[int]
    :param starting_guesses: List values of the starting guesses that correspond to the roots Newton's method landed on
    :type starting_guesses: list[float]
    :param failed_roots: The roots that Newton's method failed to get. Only here so output of poly_roots can be directly input
    :type failed_roots: list[str]
    :param epsilon: the epsilon value used in Newton's method. If one is given, then it is printed before the chart
    :type epsilon: float
    :param print_poly: If the polynomial (in math language) is printed before the chart. Default no
    :type print_poly: bool
    :param print_poly_degree: If the polynomial's degree is printed before the chart. Default no
    :type print_poly_degree: bool
    :return: None; prints output
    """

    if failed_roots is None:
        failed_roots = []

    # assign data
    mydata = [{"Nikhil" , "Delhi"} ,
              {"Ravi" , "Kanpur"} ,
              {"Manish" , "Ahmedabad"} ,
              {"Princ" , "Bangalore"}]

    # create header
    head = ["Name" , "City"]

    # display table
    # print(tabulate(mydata , headers = head , tablefmt = "grid"))
