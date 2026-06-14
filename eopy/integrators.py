"""SciPy-backed DOP853 integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray
Rhs = Callable[[float, Array], Array]


@dataclass(slots=True)
class Dop853Options:
    rtol: float = 1.0e-10
    atol: float | Array = 1.0e-3
    first_step: float | None = None
    max_step: float = np.inf


@dataclass(slots=True)
class Dop853Stats:
    accepted_steps: int
    rejected_steps: int
    function_evals: int
    status: int
    message: str
    event_times: tuple[Array, ...] | None = None
    event_states: tuple[Array, ...] | None = None


Event = Callable[[float, Array], float]


def integrate_dop853(
    fun: Rhs,
    t0: float,
    y0: Array,
    t1: float,
    *,
    options: Dop853Options | None = None,
    return_history: bool = False,
    events: Event | Sequence[Event] | None = None,
) -> tuple[float, Array, Dop853Stats] | tuple[Array, Array, Dop853Stats]:
    """Integrate ``y' = fun(t, y)`` with SciPy's ``solve_ivp`` DOP853.

    ``fun`` still receives and returns arrays with the original shape of
    ``y0``. Internally SciPy integrates a flat vector.
    """

    opts = Dop853Options() if options is None else options
    y0_arr = np.array(y0, dtype=float, copy=True)
    shape = y0_arr.shape

    def flat_fun(t: float, y_flat: Array) -> Array:
        return np.asarray(fun(t, y_flat.reshape(shape)), dtype=float).reshape(-1)

    wrapped_events = None
    if events is not None:
        event_list = [events] if callable(events) else list(events)
        wrapped_events = []
        for event in event_list:

            def flat_event(t: float, y_flat: Array, event: Event = event) -> float:
                return float(event(t, y_flat.reshape(shape)))

            if hasattr(event, "terminal"):
                flat_event.terminal = event.terminal
            if hasattr(event, "direction"):
                flat_event.direction = event.direction
            wrapped_events.append(flat_event)

    kwargs = {
        "method": "DOP853",
        "rtol": opts.rtol,
        "atol": opts.atol,
        "max_step": opts.max_step,
    }
    if opts.first_step is not None:
        kwargs["first_step"] = opts.first_step

    if wrapped_events is not None:
        kwargs["events"] = wrapped_events
    solution = solve_ivp(flat_fun, (float(t0), float(t1)), y0_arr.reshape(-1), **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)

    event_times = None
    event_states = None
    if events is not None:
        event_times = tuple(np.asarray(t) for t in solution.t_events)
        event_states = tuple(np.asarray(y).reshape((-1,) + shape) for y in solution.y_events)

    stats = Dop853Stats(
        accepted_steps=max(0, solution.t.size - 1),
        rejected_steps=0,
        function_evals=solution.nfev,
        status=solution.status,
        message=solution.message,
        event_times=event_times,
        event_states=event_states,
    )
    if return_history:
        ys = np.moveaxis(solution.y, 1, 0).reshape((solution.t.size,) + shape)
        return solution.t, ys, stats
    return float(solution.t[-1]), solution.y[:, -1].reshape(shape), stats
