# imports
import math

from polynomials import *
from barcode import *
import random
import pandas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import longdouble
#Fix bug: https://github.com/aleju/imgaug/issues/537
np.random.BitGenerator = np.random.bit_generator.BitGenerator

from math import isclose
from enum import Enum
from styleframe import StyleFrame, Styler, utils

import mplcursors

# constants
BUILD_BINOMIAL_RANGE = 10


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


def get_data_of_poly_roots_static_accuracy(testing_polynomials = None , epsilon = None , max_attempts_before_quitting = 100 , human_dividng = False , no_progress_threshold = 1e-12 , stop_when_no_progress = False , debug = False):
    result_total_steps_when_completely_successful = []
    result_total_steps_successful_guess_when_completely_successful = []
    result_percent_guesses_successful_when_completely_successful = []
    result_wasted_steps = []
    complete_fail_count = 0
    hit_steps_limit_fail_count = 0
    zero_derivative_fail_count = 0
    no_progress_fail_count = 0
    problematic_poly_count = 0
    times_failed_single_poly = 0
    for i in range(len(testing_polynomials)):
        """
        # For testing when speed is important (so low sample sizes will not throw errors about arrays being empty)
        solved_completely = False
        while not solved_completely:  # carry on... this is just here for testing purposes
            polynomial = poly_maker(poly_degree, only_int_roots=only_int_roots)
            solved_completely , poly_roots , fail_roots , remainders = polynomial.get_roots_with_dividing(max_steps_per_root = 4096 , epsilon = epsilon , sort_roots = False)
        """
        polynomial = testing_polynomials[i]  # "for polynomial in testing_polynomials" is not done so polynomials can be retried
        solved_completely , calculated_poly_roots_set , fail_roots , remainders = \
            polynomial.get_roots_with_dividing(max_steps_per_root = 4096 , epsilon = epsilon,
                                               max_attempts_before_quitting = max_attempts_before_quitting,
                                               human_dividing = human_dividng , no_progress_threshold = no_progress_threshold,
                                               stop_when_no_progress = stop_when_no_progress , debug = debug)

        for failed_root in fail_roots:
            if failed_root.failure_reason == NewtonResult.FAILURES.HORIZONTAL:
                zero_derivative_fail_count += 1
            elif failed_root.failure_reason == NewtonResult.FAILURES.NO_PROGRESS:
                no_progress_fail_count += 1
            elif failed_root.failure_reason == NewtonResult.FAILURES.HIT_STEP_LIMIT:
                hit_steps_limit_fail_count += 1
        if not stop_when_no_progress:
            for root in calculated_poly_roots_set:
                if root.failure_reason == NewtonResult.FAILURES.NO_PROGRESS:
                    no_progress_fail_count += 1

        if solved_completely:
            total_steps = int(np.sum([r.steps_taken for r in calculated_poly_roots_set])) + int(np.sum([r.steps_taken for r in fail_roots]))
            total_steps_successful_guesses = int(np.sum([r.steps_taken for r in calculated_poly_roots_set]))
            percent_guesses_successful = len(calculated_poly_roots_set) / (len(calculated_poly_roots_set) + len(fail_roots))
            result_wasted_steps.append(int(np.sum([r.steps_taken for r in fail_roots])))

            result_total_steps_when_completely_successful.append(total_steps)
            result_total_steps_successful_guess_when_completely_successful.append(total_steps_successful_guesses)
            result_percent_guesses_successful_when_completely_successful.append(percent_guesses_successful)
        else:
            result_wasted_steps.append(int(np.sum([r.steps_taken for r in calculated_poly_roots_set])) + int(np.sum([r.steps_taken for r in fail_roots])))
            complete_fail_count += 1
            i -= 1  # retry this polynomial because it didn't work
            times_failed_single_poly += 1
            if times_failed_single_poly > 1:
                i += 1
                problematic_poly_count += 1
                if debug:
                    print("PROBLEMATIC POLYNOMIAL: {}".format(polynomial))
                polynomial.save_polynomial("Problem Child")
                times_failed_single_poly = 0
    percent_steps_wasted = int(np.sum(result_wasted_steps)) / (int(np.sum(result_total_steps_successful_guess_when_completely_successful)) + int(np.sum(result_wasted_steps)))

    return len(testing_polynomials) , complete_fail_count , hit_steps_limit_fail_count , zero_derivative_fail_count , no_progress_fail_count , problematic_poly_count , result_wasted_steps , percent_steps_wasted , result_total_steps_when_completely_successful , result_total_steps_successful_guess_when_completely_successful , result_percent_guesses_successful_when_completely_successful


def get_data_of_poly_roots_static_speed(testing_polynomials = None , steps_taken = None , no_progress_threshold = 1e-12 , stop_when_no_progress = False , debug = False):
    ZERO_ERROR_MAGNITUDE = -20  # For when the x- or y-errors are 0 (it's on the root)
    result_x_precision = []
    result_y_precision = []
    no_progress_fail_count = 0
    zero_derivative_fail_count = 0
    first_step_without_progress = []
    additional_steps_until_progress = []
    for polynomial in testing_polynomials:
        guess = (-BUILD_BINOMIAL_RANGE - 1) + (BUILD_BINOMIAL_RANGE + 2) * random.random()
        poly_root = polynomial.get_newton_root_from_point(starting_x = guess , max_steps = steps_taken,
                                                          epsilon = 1e-400,
                                                          no_progress_threshold = no_progress_threshold,
                                                          stop_when_no_progress = stop_when_no_progress , debug = debug)
        poly_root.associate_with_real_root(polynomial.get_closest_exact_root(poly_root.x_value))

        if poly_root.root_was_found or poly_root.failure_reason == NewtonResult.FAILURES.HIT_STEP_LIMIT:
            if debug and poly_root.root_was_found:
                print("FOUND - {}".format(poly_root))
            x_error_magnitude = math.log10(poly_root.x_error) if poly_root.x_error != 0 else ZERO_ERROR_MAGNITUDE
            y_error_magnitude = math.log10(poly_root.y_error) if poly_root.y_error != 0 else ZERO_ERROR_MAGNITUDE

            if x_error_magnitude < ZERO_ERROR_MAGNITUDE:
                polynomial.save_polynomial("X-precision big ({}). Root: {}".format(-x_error_magnitude , poly_root))
                result_x_precision.append(-ZERO_ERROR_MAGNITUDE)
            else:
                result_x_precision.append(-x_error_magnitude)
            if y_error_magnitude < ZERO_ERROR_MAGNITUDE:
                polynomial.save_polynomial("Y-precision big ({}). Root: {}".format(-y_error_magnitude , poly_root))
                result_y_precision.append(-ZERO_ERROR_MAGNITUDE)
            else:
                result_y_precision.append(-y_error_magnitude)
        elif poly_root.failure_reason == NewtonResult.FAILURES.HORIZONTAL:
            zero_derivative_fail_count += 1

        if poly_root.first_step_with_no_progress is not None:
            first_step_without_progress.append(poly_root.first_step_with_no_progress)
            no_progress_fail_count += 1
            if poly_root.additional_steps_taken is not None:
                additional_steps_until_progress.append(poly_root.additional_steps_taken)
    if len(first_step_without_progress) == 0:
        first_step_without_progress = [-1]  # prevents errors for appending zero-length lists (also, a negative value doesn't make sense in this context and will stick out)
    if len(additional_steps_until_progress) == 0:
        additional_steps_until_progress = [-1]  # prevents errors for appending zero-length lists (also, a negative value doesn't make sense in this context and will stick out)
    return len(testing_polynomials) , no_progress_fail_count , zero_derivative_fail_count , result_x_precision , result_y_precision , first_step_without_progress , additional_steps_until_progress


def static_accuracy_chart(num_observations = None, poly_degrees = None , epsilons = None , max_attempts_before_quitting = 100 , only_int_roots = True , human_dividing = False , no_progress_threshold = 1e-12 , stop_when_no_progress = False , debug = False):
    """
    :type num_observations: int
    :type poly_degrees: list[int]
    :type epsilons: list[float]
    :param max_attempts_before_quitting: maximum attempts before quitting
    :param only_int_roots: whether tested polynomials will have integer roots
    :type only_int_roots: bool
    :param human_dividing: whether Newton's method will factor out the actual roots if sufficiently close
    :type human_dividing: bool
    :param no_progress_threshold: the delta-x between steps that is considered no progress.
    :type no_progress_threshold: float
    :param stop_when_no_progress: if False (default), the first step at which delta-x between steps is less than
        no_progress_threshold will be recorded but no action taken. If True, then the root will be returned failed
        because of NewtonResult.FAILURES.NO_PROGRESS when delta-x is less than no_progress_threshold
    :param debug: whether informational/debugging strings will be printed (some (mostly error/failure case ones)
        will still be printed even if set to false)
    """

    poly_degrees.sort()
    epsilons.sort(reverse = True)

    result = dict()
    result["opfts"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [that] failed to solve
    result["opsw"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] steps [that were] wasted (not successful)
    result["oppp"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] polynomials [that were] problematic
    result["opfrfsl"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] failed roots [that] failed [because of hitting] step limit
    result["opfrfzd"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] failed roots [that] failed [because of] zero derivative
    result["opfrfnp"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [of] failed roots [that] failed [because of] no progress
    result["assw"] = pandas.DataFrame(index = poly_degrees)  # [Regardless of solve status,] all solves [total] steps wasted
    result["sspgs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] percent [of starting] guesses [that were] successful
    result["ssts"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the] total steps
    result["sssnfs"] = pandas.DataFrame(index = poly_degrees)  # [For] successful(ly) solve(d) [roots, the total] steps needed for solving

    testing_polynomials = dict()
    sample_polynomials = []
    degree_pos = 0
    for degree in poly_degrees:
        testing_polynomials[degree] = []
        for i in range(num_observations):
            testing_polynomials[degree].append(poly_maker(degree , only_int_roots = only_int_roots))
        sample_polynomials += testing_polynomials[degree][0:9]
        if debug:
            print(degree)
            print(testing_polynomials[degree][degree_pos:9*(degree_pos+1)])
        degree_pos += 1

    for epsilon in epsilons:
        overall_percent_failure_to_solve = []
        overall_percent_steps_wasted = []
        overall_percent_polys_problematic = []
        overall_percent_fails_hit_steps_limit = []
        overall_percent_fails_zero_derivative = []
        overall_percent_fails_no_progress = []
        accumulator_all_solves_steps_wasted = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_percent_guesses_successful = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_total_steps = StatsAccumulator("{:.1e}".format(epsilon))
        accumulator_successful_solves_steps_needed_for_successes = StatsAccumulator("{:.1e}".format(epsilon))
        for degree in poly_degrees:
            print("Working on x^{:d} polynomials with epsilon {:e}".format(degree, epsilon))
            num_samples , complete_fail_count , hit_steps_limit_fail_count , zero_derivative_fail_count ,\
                no_progress_fail_count , problem_poly_count , wasted_steps , percent_steps_wasted , \
                tot_steps_when_completely_successful , \
                tot_steps_successful_guess_when_completely_successful , \
                percent_guesses_successful_when_completely_successful = \
                get_data_of_poly_roots_static_accuracy(testing_polynomials = testing_polynomials[degree],
                                                       epsilon = epsilon,
                                                       max_attempts_before_quitting = max_attempts_before_quitting,
                                                       human_dividng = human_dividing,
                                                       no_progress_threshold = no_progress_threshold,
                                                       stop_when_no_progress = stop_when_no_progress , debug = debug)

            overall_percent_failure_to_solve.append(complete_fail_count / num_samples)
            overall_percent_steps_wasted.append(percent_steps_wasted)
            overall_percent_polys_problematic.append(problem_poly_count / num_samples)

            total_number_failed_roots = hit_steps_limit_fail_count + zero_derivative_fail_count + no_progress_fail_count
            if total_number_failed_roots == 0:
                total_number_failed_roots = 1  # It is the denominator so it being 0 will cause problems and it being 1 won't change anything
            overall_percent_fails_hit_steps_limit.append(hit_steps_limit_fail_count / total_number_failed_roots)
            overall_percent_fails_zero_derivative.append(zero_derivative_fail_count / total_number_failed_roots)
            overall_percent_fails_no_progress.append(no_progress_fail_count / total_number_failed_roots)

            accumulator_all_solves_steps_wasted.append_data(wasted_steps)
            accumulator_successful_solves_percent_guesses_successful.append_data(percent_guesses_successful_when_completely_successful)
            accumulator_successful_solves_total_steps.append_data(tot_steps_when_completely_successful)
            accumulator_successful_solves_steps_needed_for_successes.append_data(tot_steps_successful_guess_when_completely_successful)
        result["opfts"]["{:.1e}".format(epsilon)] = overall_percent_failure_to_solve
        result["opsw"]["{:.1e}".format(epsilon)] = overall_percent_steps_wasted
        result["oppp"]["{:.1e}".format(epsilon)] = overall_percent_polys_problematic

        result["opfrfsl"]["{:.1e}".format(epsilon)] = overall_percent_fails_hit_steps_limit
        result["opfrfzd"]["{:.1e}".format(epsilon)] = overall_percent_fails_zero_derivative
        result["opfrfnp"]["{:.1e}".format(epsilon)] = overall_percent_fails_no_progress

        accumulator_all_solves_steps_wasted.add_to_dataframe(result["assw"])
        accumulator_successful_solves_percent_guesses_successful.add_to_dataframe(result["sspgs"])
        accumulator_successful_solves_total_steps.add_to_dataframe(result["ssts"])
        accumulator_successful_solves_steps_needed_for_successes.add_to_dataframe(result["sssnfs"])
    # saved_polynomials = [item for item in testing_polynomials if item.save_reason is not None]  # gross code i (desparately) wanna see work
    saved_polynomials = []
    for degree , polynomials in testing_polynomials.items():
        saved_polynomials += filter(lambda item: item.save_reason is not None , polynomials)
    result["oneat_polynomials"] = pandas.DataFrame(index = range(len(saved_polynomials)))  # The polynomials that were used
    result["oneat_polynomials"]["reason"] = [item.save_reason for item in saved_polynomials]
    result["oneat_polynomials"]["degree"] = [item.get_degree() for item in saved_polynomials]
    result["oneat_polynomials"]["coefficients"] = [item.poly_coefficients_list for item in saved_polynomials]
    result["oneat_polynomials"]["printed"] = [item.poly_printer() for item in saved_polynomials]
    result["oneat_polynomials"]["actual roots"] = [item.poly_roots for item in saved_polynomials]
    result["osample_polynomials"] = pandas.DataFrame(index = range(len(sample_polynomials)))  # The polynomials that were used
    result["osample_polynomials"]["degree"] = [item.get_degree() for item in sample_polynomials]
    result["osample_polynomials"]["coefficients"] = [item.poly_coefficients_list for item in sample_polynomials]
    result["osample_polynomials"]["printed"] = [item.poly_printer() for item in sample_polynomials]
    result["osample_polynomials"]["actual roots"] = [item.poly_roots for item in sample_polynomials]
    return result


def static_speed_chart(num_observations = None, poly_degrees = None , max_steps = None , only_int_roots = False , human_dividing = False , no_progress_threshold = 1e-12 , stop_when_no_progress = False , debug = False):
    """
    :type num_observations: int
    :type poly_degrees: list[int]
    :type max_steps: list[int]
    :param only_int_roots: whether tested polynomials will have integer roots
    :type only_int_roots: bool
    :param human_dividing: whether Newton's method will factor out the actual roots if sufficiently close
    :type human_dividing: bool
    :param no_progress_threshold: the delta-x between steps that is considered no progress.
    :type no_progress_threshold: float
    :param stop_when_no_progress: if False (default), the first step at which delta-x between steps is less than
        no_progress_threshold will be recorded but no action taken. If True, then the root will be returned failed
        because of NewtonResult.FAILURES.NO_PROGRESS when delta-x is less than no_progress_threshold
    :param debug: whether informational/debugging strings will be printed (some (mostly error/failure case ones)
        will still be printed even if set to false)
    """

    poly_degrees.sort()
    max_steps.sort()

    result = dict()
    result["opfcnp"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [that] failed [to] converge [because] no progress
    result["opfcsz"] = pandas.DataFrame(index = poly_degrees)  # Overall percentage [that] failed [to] converge [because] slope zero
    result["x-error"] = pandas.DataFrame(index = poly_degrees)  # After reaching the step limit, the negative of the magnitude of the x-error (difference between approximation and nearest root)
    result["y-error"] = pandas.DataFrame(index = poly_degrees)  # After reaching the step limit, the negative of the magnitude of the y-error (difference between approximation and nearest root)
    result["fsnp"] = pandas.DataFrame(index = poly_degrees)  # [The] first step [where] no progress [started] being made
    result["asup"] = pandas.DataFrame(index = poly_degrees)  # [The number of] additional steps [made] until progress [was made] (until NoProgress error went away)

    testing_polynomials = dict()
    sample_polynomials = []
    for degree in poly_degrees:
        testing_polynomials[degree] = []
        for i in range(num_observations):
            testing_polynomials[degree].append(poly_maker(degree , only_int_roots = only_int_roots))
        sample_polynomials += testing_polynomials[degree][0:9]

    for max_step in max_steps:
        overall_percent_failed_no_progress = []
        overall_percent_failed_slope_zero = []
        accumulator_x_precision = StatsAccumulator("{} steps,".format(max_step))
        accumulator_y_precision = StatsAccumulator("{} steps,".format(max_step))
        accumulator_first_step_without_progress = StatsAccumulator("{} steps,".format(max_step))
        accumulator_additional_steps_until_progress = StatsAccumulator("{} steps,".format(max_step))
        for degree in poly_degrees:
            print("Working on x^{:d} polynomials with {:d} max steps".format(degree, max_step))
            num_samples , no_progress_fail_count , horizontal_fail_count , x_error_precision , y_error_precision , \
                first_step_without_progress , additional_steps_until_progress = \
                get_data_of_poly_roots_static_speed(testing_polynomials = testing_polynomials[degree],
                                                    steps_taken = max_step,
                                                    no_progress_threshold = no_progress_threshold,
                                                    stop_when_no_progress = stop_when_no_progress , debug = debug)
            total_fail_count = no_progress_fail_count + horizontal_fail_count
            if total_fail_count == 0:
                total_fail_count = 1  # It is the denominator so it being 0 will cause problems and it being 1 won't change anything
            overall_percent_failed_no_progress.append(no_progress_fail_count / total_fail_count)
            overall_percent_failed_slope_zero.append(horizontal_fail_count / total_fail_count)

            if len(x_error_precision) == 0:
                print("No x-errors")
            accumulator_x_precision.append_data(x_error_precision)
            accumulator_y_precision.append_data(y_error_precision)
            accumulator_first_step_without_progress.append_data(first_step_without_progress)
            accumulator_additional_steps_until_progress.append_data(additional_steps_until_progress)

        result["opfcnp"]["{:d}".format(max_step)] = overall_percent_failed_no_progress
        result["opfcsz"]["{:d}".format(max_step)] = overall_percent_failed_slope_zero
        accumulator_x_precision.add_to_dataframe(result["x-error"])
        accumulator_y_precision.add_to_dataframe(result["y-error"])
        accumulator_first_step_without_progress.add_to_dataframe(result["fsnp"])
        accumulator_additional_steps_until_progress.add_to_dataframe(result["asup"])
    saved_polynomials = []
    for degree , polynomials in testing_polynomials.items():
        saved_polynomials += filter(lambda item: item.save_reason is not None , polynomials)
    result["oneat_polynomials"] = pandas.DataFrame(index = range(len(saved_polynomials)))  # The polynomials that were used
    result["oneat_polynomials"]["reason"] = [item.save_reason for item in saved_polynomials]
    result["oneat_polynomials"]["degree"] = [item.get_degree() for item in saved_polynomials]
    result["oneat_polynomials"]["coefficients"] = [item.poly_coefficients_list for item in saved_polynomials]
    result["oneat_polynomials"]["printed"] = [item.poly_printer() for item in saved_polynomials]
    result["oneat_polynomials"]["actual roots"] = [item.poly_roots for item in saved_polynomials]
    result["osample_polynomials"] = pandas.DataFrame(index = range(len(sample_polynomials)))  # The polynomials that were used
    result["osample_polynomials"]["degree"] = [item.get_degree() for item in sample_polynomials]
    result["osample_polynomials"]["coefficients"] = [item.poly_coefficients_list for item in sample_polynomials]
    result["osample_polynomials"]["printed"] = [item.poly_printer() for item in sample_polynomials]
    result["osample_polynomials"]["actual roots"] = [item.poly_roots for item in sample_polynomials]
    return result


def order_dataframe_columns_in_subcolumn_order(dataframe , subcolumn_suffix_order , column_separations = False , row_title = "Degree"):
    original_dataframe_column_names = dataframe.columns.tolist()

    if column_separations:
        original_dataframe_index = dataframe.index.tolist()
        titled_dataframe_index = ["{} {}".format(row_title, row) for row in original_dataframe_index]
        blank_column = [None] * len(original_dataframe_index)
        dataframe["BLANK"] = blank_column
        dataframe["{}".format(row_title)] = titled_dataframe_index

    resulting_column_order = []
    for suffix in subcolumn_suffix_order:
        for df_column in original_dataframe_column_names:
            if df_column.endswith(suffix):
                resulting_column_order.append(df_column)
        if column_separations:
            resulting_column_order.append("BLANK")
            resulting_column_order.append("{}".format(row_title))
    return dataframe[resulting_column_order]


def save_dataframes_to_tabs_of_file(df_dict, file_path, subcolumn_suffix_order = None):
    writer = pandas.ExcelWriter(file_path , engine = 'xlsxwriter')
    for sheet, dataframe in df_dict.items():
        if subcolumn_suffix_order is not None:
            if not sheet.startswith("o"):
                # Can't sort the "overall" entries... really janky monkeypatch, but it wooorks
                dataframe = order_dataframe_columns_in_subcolumn_order(dataframe , subcolumn_suffix_order , column_separations = True)
        dataframe.to_excel(writer, sheet_name = sheet, startrow = 0 , startcol = 0)
    writer.save()


sample_size = 4
poly_degrees = [2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]
epsilons = [1e-3 , 5e-4 , 1e-4 , 5e-5 , 1e-5 , 5e-6 , 1e-6 , 5e-7 , 1e-7 , 5e-8 , 1e-8 , 5e-9 , 1e-9 , 5e-10 , 1e-10]
max_steps_list = [3 , 4 , 5 , 6 , 7 , 8 , 16 , 32 , 64 , 128 , 256 , 512 , 1024 , 2048 , 4096]
chart_only_int_roots = False

static_accuracy_results = static_accuracy_chart(num_observations = sample_size , poly_degrees = poly_degrees , epsilons = epsilons , only_int_roots = chart_only_int_roots , human_dividing = False , debug = False)
save_dataframes_to_tabs_of_file(static_accuracy_results, r'/Temp/Math-IA/legacy_static_accuracy_chart.xlsx', subcolumn_suffix_order = ["ct" , "avg" , "std" , "min" , "max" , "25" , "50" , "75"])
static_speed_results = static_speed_chart(num_observations = sample_size , poly_degrees = poly_degrees , max_steps = max_steps_list , only_int_roots = chart_only_int_roots , human_dividing = False , stop_when_no_progress = False , no_progress_threshold = 1e-12)
save_dataframes_to_tabs_of_file(static_speed_results, r'/Temp/Math-IA/legacy_static_speed_chart.xlsx', subcolumn_suffix_order = ["ct" , "avg" , "std" , "min" , "max" , "25" , "50" , "75"])
input("Press Enter to continue...")


# print(get_data_of_poly_roots_static_speed(500 , 10 , 32))
shame = Polynomial([476280.00000361796 , 1905120.0])
print(shame.get_newton_root_from_point(1))

"""
sf = StyleFrame(dataframe)
        sf.apply_column_style(cols_to_style = dataframe.columns ,
                              styler_obj = Styler(bg_color = utils.colors.white , bold = True ,
                                                  font = utils.fonts.calibri , font_size = 12) , style_header = True)
        sf.apply_headers_style(styler_obj = Styler(bg_color = utils.colors.grey , bold = True , font_size = 11 ,
                                                   font_color = utils.colors.black ,
                                                   number_format = utils.number_formats.general , protection = False))
"""
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
solved_completely , calc_roots , fail_roots , remainders , temp_cool_polys = new_poly.get_roots_with_dividing(max_steps_per_root = 4096 , max_attempts_before_quitting = 50 , epsilon = 1e-9)
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
poly_barcode = BarCode(special_basin_poly , -15 , 15 , 1100)
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
