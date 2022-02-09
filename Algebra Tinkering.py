def multiply(input_list):
    result = 1
    for element in input_list:
        result *= element
    return result


def better_multiply(*multicands):
    result = 1
    for element in multicands:
        result *= element
    return result


def display(**kwargs):
    for key , value in zip(kwargs.keys() , kwargs.values()):
        print(key , value)


def arg_convention(required_input , optional_input = 5 , *args , **kwargs):
    print(required_input , optional_input , args , kwargs)
    return 3


my_list = [1 , 2 , 3 , 4 , 5]
print(len(my_list))
print(display(x = "test", y = "ing ", z = "kwargs"))
print(multiply([6 , 9]))
print(better_multiply(9 , 6))
