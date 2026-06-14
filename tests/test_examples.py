import importlib

import numpy as np


EXAMPLES = [
    "examples.two_body_orbit",
    "examples.j2_precession",
    "examples.drag_decay",
    "examples.srp_shadow",
    "examples.third_body_sun",
]


def test_examples_run_and_return_finite_results():
    for module_name in EXAMPLES:
        result = importlib.import_module(module_name).run()
        assert result
        for value in result.values():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))
            elif isinstance(value, float | np.floating):
                assert np.isfinite(value)


def test_two_body_example_has_small_closure_error():
    result = importlib.import_module("examples.two_body_orbit").run()
    assert result["final_error_m"] < 0.05


def test_drag_and_third_body_examples_show_expected_trends():
    drag = importlib.import_module("examples.drag_decay").run()
    third_body = importlib.import_module("examples.third_body_sun").run()

    assert drag["energy_delta"] < 0.0
    assert third_body["position_delta_m"] > 0.0
