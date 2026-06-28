import cabench.__main__ as cabench_main_module


def test_package_main_exposes_callable():
    assert callable(cabench_main_module.main)
