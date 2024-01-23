import pytest
import numpy as np


'''
Test to make sure desired output is correct
'''
def test_VALUES_TEMPLATE():
    '''
    Test description: what are we testing?
    '''

    # Import class/function to test
    from modopt.directory.file import Class
    object = Class()

    # Run test scenario
    input_val1 = 3.0
    input_val2 = 4.0
    desired_val = 7.0
    actual_val = object.add_two_numbers(
        input_val1, 
        input_val2
    )

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

'''
Test to make sure desired output is correct
'''
def test_EQUAL_TEMPLATE():
    '''
    Test description: what are we testing?
    '''

    # Import class/function to test
    from modopt.directory.file import Class
    object = Class()

    # Run test scenario
    input_val = 3.5
    desired_output = False
    actual_output = object.check_is_integer(
        input_val
    )

    # Check to make sure values are correct
    assert actual_output == desired_output


'''
Test to make sure exceptions are raised
'''
def test_EXCEPTIONS_TEMPLATE():

    '''
    Test description: what are we testing?
    '''

    # Import class/function to test
    from modopt.directory.file import Class
    object = Class()

    # Check to make sure exceptions are raised
    with pytest.raises(Exception) as exc_info:

        # Run test scenario
        variable_name_that_does_not_exist = 'x'
        object.get_variable_name(
            variable_name_that_does_not_exist
        )

    # Make sure the correct errors are being raised
    assert exc_info.type is KeyError
    assert str(exc_info.value) == 'variable not found'