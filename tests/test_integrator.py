import numpy as np

from eopy.integrators import Dop853Options, integrate_dop853


def test_dop853_integrates_harmonic_oscillator_batch():
    y0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    def rhs(_t, y):
        out = np.empty_like(y)
        out[:, 0] = y[:, 1]
        out[:, 1] = -y[:, 0]
        return out

    _t, y, stats = integrate_dop853(
        rhs,
        0.0,
        y0,
        2.0 * np.pi,
        options=Dop853Options(rtol=1e-11, atol=1e-12, max_step=0.25),
    )

    np.testing.assert_allclose(y, y0, rtol=0.0, atol=2e-10)
    assert stats.accepted_steps > 0
    assert stats.function_evals > 0
