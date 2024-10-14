def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true")
    parser.addoption("--with-desc", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.fast
    if 'fast' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("fast", [option_value])
        
    option_value = metafunc.config.option.with_desc
    if 'with_desc' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("with_desc", [option_value])
