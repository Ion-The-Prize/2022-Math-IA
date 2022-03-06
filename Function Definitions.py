# imports
import random
# import matplotlib.pyplot as plt
# from tabulate import tabulate

# constants
import barcode

BUILD_BINOMIAL_RANGE = 10


def build_a_binomial(rand_lower_bound = -BUILD_BINOMIAL_RANGE , rand_upper_bound = BUILD_BINOMIAL_RANGE , only_int_roots = False):
    """
    Randomly generates a binomial with the format ax + b, where a and b are integers

    :param rand_lower_bound: the lower bound of the coefficients
    :param rand_upper_bound: the upper bound of the coefficients
    :param only_int_roots: self explanatory; sets a to be 1 or -1
    :type only_int_roots: bool
    :returns: list coefficients of binomial in format [b , a]
    """

    b = random.randrange(rand_lower_bound , rand_upper_bound + 1)

    if only_int_roots:
        a = random.choice([1, -1])
    else:
        a = random.randrange(rand_lower_bound , rand_upper_bound + 1)
        while a == 0:
            a = random.randrange(rand_lower_bound , rand_upper_bound + 1)
    return [b , a]


def poly_degree(poly_coefficient_list):
    """Returns int degree of the polynomial"""

    poly_degree = len(poly_coefficient_list) - 1
    return poly_degree


def poly_multiplier(polynomial_first , polynomial_second = None):
    """
    Multiplies two polynomials' coefficients with each other.

    :param polynomial_first: coefficients of one polynomial
    :type polynomial_first: list[int]
    :param polynomial_second: coefficients of a second polynomial (default is [1] (function returns polynomial_first))
    :type polynomial_second: list[int]
    :returns: list of coefficients of the polynomial multiplication product
    """
    """
    Vars:
        result_degree: degree of result (coefficient is of x w/ degree of list pos)
        sec_poly_pos: iterator current coefficient of second polynomial (list pos is coefficient of x w/ same degree)
        first_poly_pos: iterator current coefficient of first polynomial (list pos is coefficient of x w/ same degree)
        result_a: list result of (sec_poly_pos'th coefficient of polynomial_second) * (all of polynomial_first)
        result: running total of all result_a's
    """

    if polynomial_second is None:
        polynomial_second = [1]

    result_degree = poly_degree(polynomial_first) + poly_degree(polynomial_second)
    result = [0] * (result_degree + 1) # read docstring
    # make result lists long enough to avoid IndexError: list assignment index out of range

    for first_poly_pos in range(len(polynomial_first)):
        # distribute current component of poly_a into poly_b
        result_a = [0] * (poly_degree(polynomial_second) + 1)  # read docstring
        for sec_poly_pos in range(len(polynomial_second)):
            result_a[sec_poly_pos] = polynomial_second[sec_poly_pos] * polynomial_first[first_poly_pos]
        # result_a = product of the sec_poly_pos'th coefficient of polynomial_second and every digit of polynomial_first
        for i in range(len(result_a)):
            # add result_a to the currect part of running total
            result[first_poly_pos + i] += result_a[i]
    return result


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
    Test of poly_multiplier
    Compares the product of two polynomials multiplied using poly_multiplier with WolframAlpha's result
    """
    print("me: ", poly_multiplier(first_poly , second_poly))
    print("wo: ", answer)

    if poly_multiplier(first_poly, second_poly) == answer:
        print("Wolfram alpha agrees")
    else:
        print("Wolfram alpha disagrees")
    return


first_test_poly_a = [-3, -1, -4, 9, -1, -1]
first_test_poly_b = [8, -3, -10, -2, -5, -8]
first_test_wolfram = [-24, 1, 1, 100, 22, -58, 23, -1, -65, 13, 8]
# (-3x^5 - 1x^4 - 4x^3 + 9x^2 - 1x - 1) * (8x^5 - 3x^4 - 10x^3 - 2x^2 - 5x - 8)
second_test_poly_a = [-1, -1, 9, -4, -1, -3]
second_test_poly_b = [-8, -5, -2, -10, -3, 8]
second_test_wolfram = [8, 13, -65, -1, 23, -58, 22, 100, 1, 1, -24]
# [-24, 1, 1, 100, 22, -58, 23, -1, -65, 13, 8]

# poly_mult_test(first_test_poly_a , first_test_poly_b , first_test_wolfram)
# poly_mult_test(second_test_poly_a , second_test_poly_b , second_test_wolfram)


def poly_adder(first_polynomial , *other_polynomials):
    """Adds polynomials"""

    result_polynomial = first_polynomial.copy()

    for polynomial in other_polynomials:
        for poly_pos in range(max(len(result_polynomial) , len(polynomial))):
            if len(result_polynomial) <= poly_pos:
                result_polynomial += [0]
            b = polynomial[poly_pos] if len(polynomial) > poly_pos else 0
            result_polynomial[poly_pos] += b
    return result_polynomial


print("Addition Test: ", poly_adder([1 , 1 , 1] , [1 , 2 , 1 , 2] , [] , [7]))


def poly_subtractor(first_polynomial , subtracting_polynomial = None):
    """Subtracts subtracting_polynomial from first_polynomial"""
    if subtracting_polynomial is None:
        subtracting_polynomial = [0]

    result_polynomial = []

    for poly_pos in range(max(len(first_polynomial) , len(subtracting_polynomial))):
        a = first_polynomial[poly_pos] if len(first_polynomial) > poly_pos else 0
        b = subtracting_polynomial[poly_pos] if len(subtracting_polynomial) > poly_pos else 0
        result_polynomial += [a - b]
    while len(result_polynomial) > 1 and result_polynomial[-1] == 0:
        del result_polynomial[-1]
    return result_polynomial


print("Subract Test: " , poly_subtractor([1 , 1 , 1] , [1 , 2 , 1 , 2]))


def poly_divider(dividend_polynomial , divisor = None):
    """
    Divides a polynomial by another polynomial.

    :param dividend_polynomial: coefficients of one polynomial (the dividend)
    :type dividend_polynomial: list[int]
    :param divisor: coefficients of a second polynomial (the divisor) (default is [1] (function returns dividend_polynomial))
    :type divisor: list[int]
    :return: tuple[list[int] , list[int]] list of coefficients of the polynomial division quotient & list remaineders
    """

    if divisor is None:
        divisor = [1]

    result = []
    dividend_remaining = dividend_polynomial
    while poly_degree(dividend_remaining) >= poly_degree(divisor):
        leading_term_coeff = dividend_remaining[-1]
        leading_term_degree = poly_degree(dividend_remaining)
        partial_quotient_coeff = int(leading_term_coeff / divisor[-1])
        partial_quotient_degree = leading_term_degree - poly_degree(divisor)
        partial_quotient_poly = [0] * (partial_quotient_degree + 1)
        partial_quotient_poly[-1] = partial_quotient_coeff
        result = poly_adder(partial_quotient_poly , result)

        poly_to_subtract = poly_multiplier(divisor , partial_quotient_poly)
        dividend_remaining = poly_subtractor(dividend_remaining , poly_to_subtract)
    return result , dividend_remaining


print("Division Test: " , poly_divider([1 , 2 , 1] , [1 , 1]))


def poly_maker(degree , build_binomial_lower = -BUILD_BINOMIAL_RANGE , build_binomial_upper = BUILD_BINOMIAL_RANGE , only_int_roots = False):
    """
    Generates a polynomial's coefficients by randomly generating factors and then expanding those factors
    Roots are calculated using each factor

    Returns:
        poly_coefficients — list of coefficients for polynomial (list pos = degree of x, so first item is the constant)
        poly_roots — list of roots for a polynomial

    :param degree: degree of polynomial
    :type degree: int
    :param build_binomial_lower: lower bound of randomly generated binomials' coefficients; the a & b in: (b , ax)
    :param build_binomial_upper: upper bound of randomly generated binomials' coefficients; the a & b in: (b , ax)
    :param only_int_roots: self explanatory; sets a in binomials to be 1 or -1
    :type only_int_roots: bool
    :returns: list of polynomial coefficients (list[int]) and the roots of that polynomial (list[float])
    """

    poly_coefficients = [1]
    poly_roots = []

    for i in range(degree):
        binomial = build_a_binomial(build_binomial_lower, build_binomial_upper, only_int_roots = only_int_roots)
        poly_coefficients = poly_multiplier(poly_coefficients , binomial)
        poly_roots = poly_roots + [-1.0 * binomial[0] / binomial[1]]

    return poly_coefficients , poly_roots


print("{:.2f}".format(3.14159))
(poly_coeff, poly_roots) = poly_maker(5)
print(poly_coeff)
print(list(reversed(poly_coeff)))
print(poly_coeff)
print(list(map(lambda num : float("{:.3f}".format(num)), poly_roots)))


def poly_value(x , poly_coefficient_list):
    """
    Given an x value and a polynomial, computes y value

    :param x: x value
    :type x: float
    :param poly_coefficient_list: coefficients for a polynomial
    :type poly_coefficient_list: list[int]
    Returns: float value of polynomial with coefficients [poly_coefficients] at x value
    """

    result = 0.0

    for i in range(len(poly_coefficient_list)):
        result += poly_coefficient_list[i] * (x ** i)
    return result


print("value at x = 1: ", poly_value(1 , poly_coeff))


def poly_primer(poly_coefficient_list):
    """
    Differentiates a polynomial

    :param poly_coefficient_list: coefficients for a polynomial (coeff correspond to x with degree of list pos)
    :type poly_coefficient_list: list[int]
    :returns: list coefficients for derivative of input polynomial
    """

    result = []

    for i in range(1 , len(poly_coefficient_list)):
        result.append(poly_coefficient_list[i] * i)
    return result


print(poly_primer(poly_coeff))


def tangent_line(x , poly_coefficient_list):
    """
    Invokes poly_f & poly_f_prime to generate a tangent line

    :param x: x value
    :type x: float
    :param poly_coefficient_list: coefficients for a polynomial (list pos = degree of x, so pos = 0 is the constant)
    :type poly_coefficient_list: list[int]
    :returns: tangent line as list polynomial coefficients
    """

    slope = poly_value(x , poly_primer(poly_coefficient_list))
    # plug x = 0 into y = m(x - x1) + y1
    y_intercept = slope * (0 - x) + poly_value(x , poly_coefficient_list)

    return [y_intercept, slope]  # representing tangent eq as my standard polynomial format


def x_intercept(line_coefficient_list):
    """Returns x-intercept of a LINEAR EQUATION"""

    assert len(line_coefficient_list) == 2
    assert line_coefficient_list[1] != 0

    x_intercept = (-1.0 * line_coefficient_list[0]) / line_coefficient_list[1]
    return x_intercept


# print("prime value: ", poly_value(1 , poly_primer(first_test_poly_a)))
print("value @ x = 1: ", poly_value(1, first_test_poly_a))
print(first_test_poly_a)
print("tan: ", tangent_line(1 , first_test_poly_a))

print("x-int: ", x_intercept(tangent_line(1 , first_test_poly_a)))

def tangent_test(x , poly_coefficient_list , answer):
    """
    Test of tangent_line
    Compares the x-intercept of a polynomial's tangent line found using poly_multiplier with WolframAlpha's result
    """

    print("BEGIN TAN_TEST")
    print("me: ", tangent_line(x , poly_coefficient_list))
    print("wo: ", answer)

    if tangent_line(x , poly_coefficient_list) == answer:
        print("Wolfram alpha agrees")
    else:
        print("Wolfram alpha disagrees")
    return


# wolfram alpha input (-3 - x - 4x^2 + 9x^3 - 1x^4 - 1x^5)
# first_test_poly_a = [-3, -1, -4, 9, -1, -1]
tangent_test(1 , first_test_poly_a , [-10 , 9])


def poly_printer(poly_coefficient_list , desmos_format = False):
    """
    Given a tuple list of polynomial coefficients and roots, returns that polynomial and roots in math language form
    (e.g. given [-1, 5, 10, 10, -5, 1] and [1], returns x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 and roots @ x = 1)

    :param poly_coefficient_list: output of poly_maker()[0]
    :type poly_coefficient_list: list[int]
    :param desmos_format: whether the output will be copy-able into Desmos (default to standard math format)
    :type desmos_format: bool
    :return: tuple string of polynomial and roots
    """

    result = ""
    reverse_poly_coefficient_list = list(reversed(poly_coefficient_list))
    for i in range(len(reverse_poly_coefficient_list)):
        coefficient = reverse_poly_coefficient_list[i]
        power = poly_degree(reverse_poly_coefficient_list) - i
        if coefficient == 0:
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
            coefficient_string = str(abs(coefficient))
            if coefficient_string == "1" and power != 0:
                coefficient_string = ""
            if power == 0:
                result += "{:}".format(coefficient_string)  # formats coeff as decimal (integer) number
            elif power == 1:
                result += "{:}x".format(coefficient_string)
            else:
                if not desmos_format:
                    result += "{:}x^{:d}".format(coefficient_string , power)
                else:
                    result += "{:}x^{{{:d}}}".format(coefficient_string , power)
                    # To paste into desmos, the single curly braces are changed into triple curly braces
    return result


print("WOLF COPY: ", poly_printer(poly_maker(20)[0]))
print("PRIME TEST: ", poly_primer([2]))
print(poly_degree(poly_primer([2])))


class NotPoly:
    def __init__(self , message = "Binomials/monomials not counted as polynomials"):
        self.message = message
        super().__init__(self.message)


def poly_relative_extrema(poly_coefficient_list):
    """
    Given a polynomial, returns tuple lists of x values of relative maxima and minima

    :param poly_coefficient_list: list coefficients for a polynomial with degree >1
    :type poly_coefficient_list: list[int]
    :return: tuple list of relative maxima , relative minima (e.g. max @ x = -4.0, min @ x = 4.0 is [-4.0] [4.0]
    :raises NotPoly: if a monomial or binomial is entered, returns None
    """

    first_derivative = poly_primer(poly_coefficient_list)
    first_derivative_zeros = []
    second_derivative = poly_primer(first_derivative)

    # Find first derivative zeroes somehow (maybe by finding GCF of coeff and pulling that out then looping??)

    rel_maxima = []
    rel_minima = []

    for i in range(poly_degree(poly_coefficient_list)):
        if poly_value(first_derivative_zeros[i] , second_derivative) > 0:
            rel_maxima += [i]
        elif poly_value(first_derivative_zeros[i] , second_derivative) < 0:
            rel_minima += [i]
        elif poly_degree(second_derivative):
            raise NotPoly
    return rel_maxima , rel_minima


def root_reorderer(unordered_poly_roots , remove_repeats = False , *parallel_sorting_lists):
    """
    Orders the roots of a list from most negative to most positive (smallest to largest).

    :param unordered_poly_roots: the list of roots that will be sorted
    :type unordered_poly_roots: list[float]
    :param remove_repeats: whether repeated roots will be removed (whether [1 , 0 , 0] will be turned into [0 , 1]) (default False)
    :type remove_repeats: bool
    :param parallel_sorting_lists: lists that will be rearranged the same way unordered_poly_roots is, in case values need to line up (so lists [6 , 4 , 5] and [0 , 1 , 2] turn into [4 , 5 , 6] and [1 , 2 , 0])
    :return: list[float] reordered list of roots and all the newly sorted *parallel_sorting_lists, if any, in the order they were input
    """

    reordered_roots = [0.0]
    reordered_roots[0] = unordered_poly_roots[0]
    if remove_repeats:
        for i in range(unordered_poly_roots.count(poly_roots[0])):
            unordered_poly_roots.remove(poly_roots[0])
    else:
        del unordered_poly_roots[0]

    while len(unordered_poly_roots) > 0:
        new_position = 0
        item = unordered_poly_roots[0]
        while new_position < len(reordered_roots) and item > reordered_roots[new_position]:
            new_position += 1

        if new_position == len(reordered_roots):
            reordered_roots.append(item)
        else:
            reordered_roots.insert(new_position , item)

        if remove_repeats:
            for i in range(unordered_poly_roots.count(item)):
                unordered_poly_roots.remove(item)
        else:
            del unordered_poly_roots[0]

    return reordered_roots


print(root_reorderer([6 , 4 , 8 , 8 , 5 , 7 , 9 , 10] , False))


def poly_Newton(poly_coefficient_list , starting_point , max_steps = 10 , epsilon = 1e-8):
    """

    :param poly_coefficient_list:
    :param starting_point:
    :param max_steps:
    :param epsilon:
    :return:
    """

    current_guess = starting_point
    current_value = poly_value(current_guess , poly_coefficient_list)
    step_number = 0
    while abs(current_value) > epsilon:
        step_number += 1
        if step_number > max_steps:
            break
        new_guess = x_intercept(tangent_line(current_guess , poly_coefficient_list))
        current_guess = new_guess
        current_value = poly_value(current_guess , poly_coefficient_list)
    if abs(current_value) < epsilon:
        return current_guess , step_number
    else:
        return None , max_steps


def poly_root(poly_coefficient_list , max_steps = 10 , epsilon = 1e-8 , starting_guess_count = None , random_starting_guesses = True , factor_roots = False , retry_root_failures = False , sort_roots = False):
    """
    Uses Newton's method for finding roots of higher order polynomials.

    :param poly_coefficient_list: list coefficients for polynomial
    :type poly_coefficient_list: list[int]
    :param max_steps: the number of tangent lines used (default 5)
    :type max_steps: int
    :param starting_guess_count: the number of starting starting_guesses (default is degree of inputted polynomial)
    :type starting_guess_count: int
    :param epsilon: really small value, only roots with y values below epsilon will be counted as actual roots
    :type epsilon: float
    :param random_starting_guesses: if the starting guesses are randomly distributed within the root range or evenly spaced (default true)
    :type random_starting_guesses: bool
    :param factor_roots: whether found roots will be factored out, resulting in a list of the actual roots (will only factor out roots with y values below epsilon)
    :type factor_roots: bool
    :param retry_root_failures: whether found roots that reached the max_steps limit without being below epsilon will be retried (the failure will still be recorded) (default False)
    :type retry_root_failures: bool
    :param sort_roots: whether the final roots string will be sorted from most negative to most positive (and repeats removed) (default False)
    :type sort_roots: bool
    :return: triple[list[float]list[int]list[string]] of [roots][# steps to get to roots][failed roots]
    """

    if starting_guess_count is None:
        starting_guess_count = poly_degree(poly_coefficient_list)
    guess_range = [-BUILD_BINOMIAL_RANGE , BUILD_BINOMIAL_RANGE]  # will be subbed in for the farthest relative extrema once that's built

    starting_guesses = [0] * starting_guess_count
    (starting_guesses[0] , starting_guesses[-1]) = (guess_range[0] - 1 , guess_range[1] + 1)

    temp = [starting_guesses[0]]
    for i in range(starting_guess_count - 2):
        temp.append(random.randrange(guess_range[0] , guess_range[1]) + 2 * random.random() - 1)  # generate a random float in the range
    temp.append(starting_guesses[-1])
    starting_guesses = temp

    poly_roots = []
    steps_needed = []
    failed_roots = []
    for i in range(starting_guess_count):
        current_guess = starting_guesses[i]
        current_value = poly_value(current_guess , poly_coefficient_list)
        step_number = 0
        while abs(current_value) > epsilon:
            step_number += 1
            if step_number > max_steps:
                break
            new_guess = x_intercept(tangent_line(current_guess , poly_coefficient_list))
            current_guess = new_guess
            current_value = poly_value(current_guess , poly_coefficient_list)
        if abs(current_value) < epsilon:
            poly_roots += [float("{:.3f}".format(current_guess))]
            steps_needed += [step_number]
        else:
            failed_roots += ["x = {:.3f} | y = {:.3f}".format(current_guess , current_value)]
    return poly_roots , steps_needed , starting_guesses , failed_roots


# print("RAND TEST: ", random.random() * 10 ** 2)
for i in range(10):
    print()
print("=====================================================================================================")

(new_poly , new_poly_roots) = poly_maker(9)
print(poly_printer(new_poly , desmos_format = True))
print(poly_printer(new_poly))
print()
(calc_roots , calc_steps , starting_guesses , fail_roots) = poly_root(new_poly , max_steps = 4096 , epsilon = 1e-6)
print("Poly Coeff: ", new_poly)
print("Real Roots: ", new_poly_roots)
print("Calc Roots: ", calc_roots)
print("Roots Step: ", calc_steps)
print("Fail Roots: ", fail_roots)

"""
question_poly = poly_maker(0, -5, 5, only_int_roots = True)
print("Factor this quadratic: ", poly_printer(question_poly[0]))
print("Zeros at x = ", question_poly[1])
"""

def poly_power(power, seed_polynomial = [1 , 1] , pascal = 0):
    """
    Raises a given polynomial to a specified power

    :param power: the power that the polynomial will be raised to
    :type power: int
    :param seed_polynomial: list coefficients for polynomial
    :type seed_polynomial: list[int]
    :param pascal: whether or not every step should be printed (giving pascal's triangle) (default no) (!= 0 is yes)
    :type pascal: any
    :return: list coefficients for polynomial
    """

    current_answer = seed_polynomial

    if power == 0:
        return [1]
    elif power < 0:
        return "no"
    else:
        for i in range(1 , power):
            current_answer = poly_multiplier(seed_polynomial , current_answer)
            if pascal != 0:
                print(i + 1 , current_answer)  # numbers rows of pascal's triangle as degree of poly from a binomial (so [1 2 1] is row 2)
        return current_answer


# print(poly_printer(poly_power(10, pascal = 0)))


def MakePolyAndBarcodeIt(poly_degree, minimum, maximum, window_width , epsilon = 1e-8):
    """

    :param poly_degree:
    :param minimum:
    :param maximum:
    :param window_width:
    :return:
    """

    assert(maximum > minimum)

    poly , poly_roots = poly_maker(poly_degree)
    increment = (maximum - minimum) / window_width

    barcode.init(minimum , maximum , window_width , 200 , poly_degree+1)

    for i in range(window_width):
        x = minimum + (i * increment)
        root , steps = poly_Newton(poly , x , 4096, epsilon)
        if root:
            for r in range(len(poly_roots)):
                if abs(root - poly_roots[r]) < epsilon:
                    barcode.add_bar(x , color_num = r , y = 10 * steps)
                    break
    for r in range(len(poly_roots)):
        barcode.add_bar(poly_roots[r], poly_degree, y=10)


MakePolyAndBarcodeIt(5 , -15 , 15 , 1100)
while 1 == 1:
    barcode.draw()

class TooManyPoly(Exception):
    def __init__(self , message = "Read the docstring"):
        self.message = message
        super().__init__(self.message)


def charter(poly_coefficient_list , poly_roots , steps_needed , starting_guesses , failed_roots = [] , epsilon = None , print_poly = False , print_poly_degree = False):
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

    # assign data
    mydata = [{"Nikhil" , "Delhi"} ,
              {"Ravi" , "Kanpur"} ,
              {"Manish" , "Ahmedabad"} ,
              {"Prince" , "Bangalore"}]

    # create header
    head = ["Name" , "City"]

    # display table
    print(tabulate(mydata , headers = head , tablefmt = "grid"))



def tester(wolfram_answer , x = 1.0 , *test_type , **polynomials):
    """
    Tests a function test_type
    Output is usually a print of "function answer: ", "wolfram answer: ", "wolfram agrees/disagrees"

    :param wolfram_answer: the answer wolfram alpha gives (i.e. what the answer *should* be)
    :param x: x value (if applicable)
    :param polynomials: lists for coefficents of polynomials (usually one or two) (number used depends on test_type) (max 2)
    :type polynomials: list
    :param test_type: the function being tested
    :type test_type: str
    :return: varies depending on the function being tested
    :raises tooManyPoly: if more than two polynomials are inputted, code breaks :p
    """

    function_answer = None
    first_poly = []  # temporary
    second_poly = [] # temporary
    i = 0

    for key , value in zip(polynomials.keys() , polynomials.values()):
        key = value
        i += 1
    if i >= 3:
        raise TooManyPoly()

    if test_type is not None:
        if test_type == "poly_multiplier" or "poly_mult":
            function_answer = poly_multiplier(first_poly , second_poly)
        if test_type == "poly_value":
            function_answer = poly_value(x , first_poly)
        if test_type == "poly_primer" or "poly_prime" or "prime":
            function_answer = poly_primer(first_poly)
        if test_type == "tangent_line" or "tangent":
            function_answer = tangent_line(x , first_poly)
        if function_answer is wolfram_answer:
            print("Wolfram alpha agrees")
        else:
            print("Wolfram alpha disagrees")
        print("Function answer: " , function_answer)
        print("Wolfram answer:  " , wolfram_answer)
    else:
        return
