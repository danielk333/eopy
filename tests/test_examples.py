import hashlib
import importlib

import numpy as np

EXAMPLES = [
    "examples.two_body_orbit",
    "examples.j2_precession",
    "examples.drag_decay",
    "examples.srp_shadow",
    "examples.third_body_sun",
    "examples.propagation_profile",
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


def test_propagation_profile_example_reports_enabled_effects():
    result = importlib.import_module("examples.propagation_profile").run(duration_s=60.0)

    assert result["stats"].profile is result["profile"]
    assert "drag_atmosphere" in result["profile"].timings
    assert "third_bodies" in result["profile"].timings
    assert "solar_radiation_pressure" in result["report"]


def test_native_gravity_example_runs_with_local_gfc(tmp_path):
    gfc = tmp_path / "tiny.gfc"
    gfc.write_text(
        "\n".join(
            [
                "modelname tiny",
                "earth_gravity_constant 398600441800000.0",
                "radius 6378136.3",
                "norm unnormalized",
                "end_of_head",
                "gfc 0 0 1.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    result = importlib.import_module("examples.native_gravity_model").run(
        gfc, degree=0, order=0, backend="python", duration_s=60.0
    )

    assert result["model_name"] == "tiny"
    assert np.all(np.isfinite(result["states"]))
    assert result["stats"].accepted_steps > 0


def test_download_data_files_example_writes_to_target_folder(tmp_path):
    source = tmp_path / "source.gfc"
    source.write_text("example gravity\n", encoding="utf-8")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = importlib.import_module("examples.download_data_files").download_requested_files(
        tmp_path / "target",
        gravity_url=source.as_uri(),
        gravity_filename="tiny.gfc",
        gravity_sha256=expected_hash,
        include_gravity=True,
        include_space_weather=False,
        include_eop=False,
    )

    assert result["gravity"].name == "tiny.gfc"
    assert result["gravity"].read_text(encoding="utf-8") == "example gravity\n"
