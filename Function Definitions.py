# imports
import math

from barcode import *
import random
import pandas
import matplotlib.pyplot as plt
import numpy as np
from numpy import longdouble
from math import isclose

# constants
BUILD_BINOMIAL_RANGE = 10

def floatToString(f):
    return ('%.6f' % f).rstrip('0').rstrip('.')

def wrap_float(f):
    """
    All math-relevant floats are run through here to, perhaps, upgrade them to numpy.longdouble
    """
    return longdouble(f)

class CalculatedRoot:
    """A calculated root"""

    x_value = None
    y_value = None
    steps_taken = None
    root_was_found = None
    starting_guess = None

    def __init__(self , x_value , y_value , steps_taken , starting_guess , root_was_found = True):
        self.x_value = x_value
        self.y_value = y_value
        self.steps_taken = steps_taken
        self.starting_guess = starting_guess
        self.root_was_found = root_was_found

    def __repr__(self):
        if self.root_was_found:
            return "x={:.3f} #{:d} @{:.3f}".format(self.x_value , self.steps_taken , self.starting_guess)
        else:
            return "FAIL: x={:.3f}->{:.3e}(#{:d}@{:.3f})".format(self.x_value , self.y_value , self.steps_taken , self.starting_guess)

    def was_found(self):
        return self.root_was_found


class Polynomial:
    """A polynomial"""

    poly_coefficients_list = None
    poly_roots = None  # None is you don't know the roots
    poly_degree = None

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
        self.poly_degree = len(poly_coefficients_list) - 1

    def __eq__(self, other):
        return self.poly_coefficients_list == other.poly_coefficients_list

    def __repr__(self):
        return self.poly_printer()

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

    def get_newton_root_from_point(self , starting_x , max_steps = 10 , epsilon = 1e-8):
        """
        Performs Newton's method for finding roots at a given x-value

        :param starting_x:
        :param max_steps:
        :param epsilon:
        :return:
        """

        epsilon = wrap_float(epsilon)
        current_guess = wrap_float(starting_x)
        current_value = self.evaluate(current_guess)
        step_number = 0
        while not isclose(current_value,0,abs_tol=epsilon):
            step_number += 1
            if step_number > max_steps:
                break
            previous_guess=current_guess
            previous_value=current_value

            new_guess_poly = self.get_tangent_line(current_guess)
            if new_guess_poly.poly_roots is None:
                return CalculatedRoot(current_guess , current_value , step_number , starting_x , root_was_found = False)
            new_guess = new_guess_poly.poly_roots[0]  # new_guess = x_intercept of tangent line
            current_guess = new_guess
            current_value = self.evaluate(current_guess)
            if self.get_degree() == 1 and step_number == 5:
                 print("Updating guess for {} time: changed {:e} :: {}({:.5e})={:.5e} notclose[{:e}] used {}({:.5e}) to get to new_guess={:.5e}/new_value={:.5e}"
                       .format(step_number, current_guess-previous_guess, self, previous_guess, previous_value, epsilon, new_guess_poly.poly_printer(coeff_format="{}"), previous_guess, current_guess, current_value))
        if isclose(current_value, 0, abs_tol=epsilon):
            return CalculatedRoot(current_guess , current_value , step_number , starting_x)
        else:
            return CalculatedRoot(current_guess , current_value , step_number , starting_x , root_was_found = False)

    def get_roots(self , max_steps = 20 , epsilon = 1e-8 , starting_guess_count = None ,
                  random_starting_guesses = True , guess_range_min = -BUILD_BINOMIAL_RANGE - 1,
                  guess_range_max = BUILD_BINOMIAL_RANGE + 1, factor_roots = False , sort_roots = False):
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
        :param factor_roots: INCOMPATIBLE WITH EVENLY SPACED STARTING GUESSES... whether found roots will be factored out,
            results in a list of the actual roots (will only factor out roots with y values below epsilon and will try again until all roots are found)
        :type factor_roots: bool
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
                  guess_range_max = BUILD_BINOMIAL_RANGE + 1, sort_roots = False):
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
        :return: triple[list[CalculatedRoot (successful roots)] , list[CalculatedRoot (failed roots)] , list[Polynomial (remainders)]]
        """

        if max_steps_before_quitting is None:
            max_steps_before_quitting = self.poly_degree ** 2

        poly_roots = []
        failed_roots = []
        remainders = []
        factored_poly = self
        loops = 0
        while factored_poly.poly_degree > 0:
            guess = guess_range_min + (guess_range_max - guess_range_min) * random.random()
            newton_result = factored_poly.get_newton_root_from_point(guess , max_steps_per_root , epsilon)
            if newton_result.root_was_found:
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

                break
        if sort_roots:
            poly_roots.sort()
        return poly_roots , failed_roots ,remainders

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
    :param only_int_roots: self explanatory; sets a to be 1 or -1
    :type only_int_roots: bool
    :returns: Polynomial of binomial (with root)
    """

    b = random.randrange(rand_lower_bound , rand_upper_bound + 1)

    if only_int_roots:
        a = random.choice([1, -1])
    else:
        a = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        while a == 0:
            a = random.randrange(rand_lower_bound , rand_upper_bound + 1)

    root = Polynomial([b , a]).get_linear_root()
    return Polynomial([b , a] , [root])


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


def graph(polynomial , x_min = -15 , x_max = 15 , x_resolution = 300):
    plt.style.use('_mpl-gallery')

    # make data
    x_array = np.linspace(x_min , x_max , x_resolution)
    y_array = np.empty(shape = (len(x_array) , 1) , dtype = float)
    for i in range(len(x_array)):
        y_array[i] = polynomial.evaluate(x_array[i])

    # plot
    fig , ax = plt.subplots()

    ax.plot(x_array , y_array , linewidth = 2.0)

    ax.set(xlim = (x_min , x_max) , xticks = np.arange(x_min , x_max) ,
           ylim = (-100 , 100) , yticks = np.arange(-100 , 100 , 10))

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis(True)

    plt.show()


#graph(poly_maker(5))
#input("Press Enter to continue...")


def get_data_of_poly_roots_static_accuracy(num_observations , poly_degree , epsilon):
    result_data_steps_when_successful = []
    result_data_percent_successful = []
    for i in range(num_observations):
        polynomial = poly_maker(poly_degree)

        poly_roots , fail_roots , remainders = polynomial.get_roots_with_dividing(max_steps_per_root = 4096 , epsilon = epsilon , sort_roots = False)
        percent_successful = len(poly_roots) / (len(poly_roots) + len(fail_roots))
        total_steps_taken = int(np.sum([r.steps_taken for r in poly_roots]))

        result_data_steps_when_successful.append(total_steps_taken)
        result_data_percent_successful.append(percent_successful)
    return result_data_steps_when_successful , result_data_percent_successful


def static_accuracy_successful_steps_chart(num_observations , poly_degrees , epsilons):
    """
    :type num_observations: int
    :type poly_degrees: list[int]
    :type epsilons: list[float]
    """

    poly_degrees.sort()
    epsilons.sort(reverse = True)

    result_chart = pandas.DataFrame(index = poly_degrees)
    for column in range(len(epsilons)):
        column_data_ct = []
        column_data_avg = []
        column_data_std = []
        column_data_min = []
        column_data_25 = []
        column_data_50 = []
        column_data_75 = []
        column_data_max = []
        for row in range(len(poly_degrees)):
            data , trash = get_data_of_poly_roots_static_accuracy(num_observations = num_observations , poly_degree = poly_degrees[row] , epsilon = epsilons[column])
            column_data_ct.append(len(data))
            column_data_avg.append(np.mean(data))
            column_data_std.append(np.std(data))
            column_data_min.append(np.min(data))
            column_data_25.append(np.percentile(data , 25))
            column_data_50.append(np.median(data))
            column_data_75.append(np.percentile(data , 75))
            column_data_max.append(np.max(data))
        result_chart["{:.1e} {}".format(epsilons[column] , "ct")] = column_data_ct
        result_chart["{:.1e} {}".format(epsilons[column] , "avg")] = column_data_avg
        result_chart["{:.1e} {}".format(epsilons[column] , "std")] = column_data_std
        result_chart["{:.1e} {}".format(epsilons[column] , "min")] = column_data_min
        result_chart["{:.1e} {}".format(epsilons[column] , "25")] = column_data_25
        result_chart["{:.1e} {}".format(epsilons[column] , "50")] = column_data_50
        result_chart["{:.1e} {}".format(epsilons[column] , "75")] = column_data_75
        result_chart["{:.1e} {}".format(epsilons[column] , "max")] = column_data_max
    return result_chart


poly_degrees = [5 , 6 , 7 , 8 , 9 , 10]
epsilons = [1e-3 , 1e-4 , 1e-5 , 1e-6 , 1e-7 , 1e-8 , 1e-16]
data_frame_test = static_accuracy_successful_steps_chart(30 , poly_degrees , epsilons)
data_frame_test.to_excel(r'/tmp/Math-IA/static_accuracy_successful_steps_chart2.xlsx', index = True, header=True)
print(data_frame_test)


def get_data_of_poly_roots_static_speed(num_observations , poly_degree , steps_taken):
    result_data_absolute_error = []
    for i in range(num_observations):
        polynomial = poly_maker(poly_degree)
        guess = (-BUILD_BINOMIAL_RANGE - 1) + (BUILD_BINOMIAL_RANGE + 2) * random.random()
        poly_root = polynomial.get_newton_root_from_point(starting_x = guess , max_steps = steps_taken , epsilon = 0)

        result_data_absolute_error.append(math.fabs(poly_root.y_value))
    result_dataframe = pandas.DataFrame(result_data_absolute_error)
    return result_dataframe , result_dataframe.describe()


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
calc_roots , fail_roots , remainders = new_poly.get_roots_with_dividing(max_steps_per_root = 4096 , max_steps_before_quitting = 50 , epsilon = 1e-9)
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
poly_barcode = None


def BarcodePoly(polynomial , minimum , maximum , window_width , epsilon = 1e-8):
    """

    :param polynomial:
    :type polynomial: Polynomial
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

    global poly_barcode
    poly_barcode = BarCode("{:s} | Roots @x = {:s}".format(polynomial.poly_printer() , ",".join(["{:.3f}".format(r) for r in sorted(polynomial.poly_roots)]))
                           , minimum , maximum , window_width , 200 , polynomial.poly_degree + 1)
    poly_barcode.close_on_click()

    for i in range(len(polynomial.poly_roots)):
        poly_barcode.assign_color_number_to_item(polynomial.poly_roots[i] , i)

    i = 0
    for x in poly_barcode.get_x_range():
        i += 1
        if i % 100 == 0:
            poly_barcode.draw()
        root = polynomial.get_newton_root_from_point(starting_x = x , max_steps = 128 , epsilon = epsilon)
        print("Newton Result: ", root)
        if root.root_was_found:
            # Looking for what color the root bar should be
            closest_exact_root = None
            closest_epsilon = None
            for r in range(len(polynomial.poly_roots)):
                current_epsilon = abs(root.x_value - polynomial.poly_roots[r])
                if closest_exact_root is None or closest_epsilon > current_epsilon:
                    closest_exact_root = polynomial.poly_roots[r]
                    closest_epsilon = current_epsilon
            # print("Closest Epsilon: {:.2e} | Closest Root: {:.2f}".format(closest_epsilon , closest_exact_root))
            poly_barcode.add_bar(x = x , color_item = closest_exact_root , y = root.steps_taken)
        else:
            poly_barcode.add_bar(x = x , color = GREY)
    for r in range(len(polynomial.poly_roots)):
        poly_barcode.add_bar(x = polynomial.poly_roots[r] , color = WHITE , y = None)
    poly_barcode.draw()
    print("Done drawing")


input("Press Enter to continue...")
BarcodePoly(new_poly , -15 , 15 , 1100)
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
