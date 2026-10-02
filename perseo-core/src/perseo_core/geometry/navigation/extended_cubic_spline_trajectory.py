# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Extended Cubic Spline Trajectory module.

This module provides the `ExtendedCubicSplineOrbit` class, which enables orbit propagation
outside of the domain defined by state vectors
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import satkit as sk

from perseo_core.geometry.coordinates.conversions import ecef2eci, eci2ecef
from perseo_core.geometry.navigation.cubic_spline_trajectory import CubicSplineTrajectory
from perseo_core.geometry.navigation.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime


class ExtendedCubicSplineOrbit(Trajectory[PreciseDateTime]):
    """Trajectory based on a Cubic Spline interpolator."""

    _INITIAL_EXTENSION_S: int = 120  # propagation performed at initialization
    _EXTENSION_MARGIN_S: int = 60  # extra propagation beyond the requested time
    _WARNING_THRESHOLD_S: int = 3600 * 3  # warn when evaluating more than 3 hours past the original trajectory
    _GRAVITY_DEGREE: int = 70
    _GRAVITY_ORDER: int = 70

    def __init__(
        self,
        times: npt.NDArray,
        positions: npt.NDArray[np.floating],
        velocities: npt.NDArray[np.floating],
    ) -> None:
        """Create an ExtendedCubicSplineOrbit from state vectors: times, positions and velocities.

        Times must be of type PreciseDateTime.

        Positions and velocities must be specified as (N, 3) arrays of floats.

        ExtendedCubicSplineOrbit can extend trajectory for low orbit satellites outside the boundaries of time axis
        through high precision numerical integration.

        Parameters
        ----------
        times : npt.NDArray
            time axis as numpy array of shape (N,)
        positions : npt.NDArray[np.floating]
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : npt.NDArray[np.floating]
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z

        """
        if not all(isinstance(t, PreciseDateTime) for t in times):
            msg = "Times must be an array of PreciseDateTime"
            raise TypeError(msg)

        if times.ndim != 1:
            msg = "Times must be a 1D array"
            raise ValueError(msg)

        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "Positions must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if velocities.ndim != 2 or velocities.shape[1] != 3:
            msg = "Velocities must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if not (len(times) == positions.shape[0] == velocities.shape[0]):
            msg = "Times, positions and velocities must have the same number of samples"
            raise ValueError(msg)

        self._original = CubicSplineTrajectory(times=times, positions=positions, velocities=velocities)

        t_anchor = self._floor_to_microsecond(times[-1])
        self._anchor = t_anchor
        ecef_pos = self._original.position(t_anchor)
        ecef_vel = self._original.velocity(t_anchor)

        self._ext_times: npt.NDArray = np.array([t_anchor], dtype=object)
        self._ext_pos = np.atleast_2d(ecef_pos)
        self._ext_vel = np.atleast_2d(ecef_vel)
        eci_pos, eci_vel = ecef2eci(ecef_pos, ecef_vel, t_anchor)
        self._eci_state = np.hstack([eci_pos, eci_vel])

        self._extension: CubicSplineTrajectory | None = None
        self._extend_by(self._INITIAL_EXTENSION_S)

    def _extend_by(self, duration_s: int) -> None:
        """Propagate `duration_s` (integer) seconds past the current end of the extension."""
        t_last = self._ext_times[-1]
        new_times = np.arange(0, duration_s + 1, dtype=float) + t_last
        sk_times = self._to_satkit_time(new_times)

        result = sk.propagate(
            self._eci_state,
            sk_times[0],
            end=sk_times[-1],
            propsettings=sk.propsettings(
                gravity_model=sk.gravmodel.egm2008,
                gravity_degree=self._GRAVITY_DEGREE,
                gravity_order=self._GRAVITY_ORDER,
            ),
        )
        evaluation = np.atleast_2d(result.interp(sk_times))
        new_pos, new_vel = eci2ecef(evaluation[:, 0:3], evaluation[:, 3:6], new_times)

        self._ext_times = np.concatenate([self._ext_times, new_times[1:]])
        self._ext_pos = np.vstack([self._ext_pos, new_pos[1:]])
        self._ext_vel = np.vstack([self._ext_vel, new_vel[1:]])
        self._eci_state = evaluation[-1]

        self._extension = CubicSplineTrajectory(self._ext_times, self._ext_pos, self._ext_vel)

    def _ensure_covers(self, t_max: PreciseDateTime) -> None:
        """Extend the propagated trajectory so that it covers `t_max` plus a safety margin."""
        elapsed_s = float(t_max - self._original.domain[1])
        if elapsed_s > self._WARNING_THRESHOLD_S:
            warnings.warn(
                f"Requested time is {elapsed_s:.0f} s past the end of the original trajectory: "
                f"propagation beyond {self._WARNING_THRESHOLD_S} s may be inaccurate",
                RuntimeWarning,
                stacklevel=4,
            )
        t_end = self._ext_times[-1]
        if t_max <= t_end:
            return
        missing_s = float(t_max - t_end)
        self._extend_by(int(np.ceil(missing_s)) + self._EXTENSION_MARGIN_S)

    @staticmethod
    def _to_satkit_time(times: npt.NDArray) -> list[sk.time]:
        return [
            sk.time(
                t.year,
                t.month,
                t.day_of_the_month,
                t.hour_of_day,
                t.minute_of_hour,
                t.second_of_minute + t.picosecond_of_second * 1e-12,
            )
            for t in times
        ]

    @staticmethod
    def _floor_to_microsecond(t: PreciseDateTime) -> PreciseDateTime:
        return PreciseDateTime.from_numeric_datetime(
            year=t.year,
            month=t.month,
            day=t.day_of_the_month,
            hours=t.hour_of_day,
            minutes=t.minute_of_hour,
            seconds=t.second_of_minute,
            picoseconds=(t.picosecond_of_second // 1_000_000) * 1_000_000,
        )

    @property
    def positions(self) -> np.ndarray:
        """Accessing trajectory positions vector."""
        return self._original.positions

    @property
    def velocities(self) -> np.ndarray:
        """Accessing trajectory velocities vector."""
        return self._original.velocities

    @property
    def times(self) -> np.ndarray:
        """Accessing trajectory times vector."""
        return self._original.times

    @property
    def domain(self) -> tuple[PreciseDateTime, PreciseDateTime]:
        """Accessing time domain (the upper bound grows as the trajectory is propagated)."""
        return (self._original.domain[0], self._ext_times[-1])

    def _evaluate(self, time: PreciseDateTime | npt.NDArray, method: str) -> np.ndarray:
        t = np.atleast_1d(time)
        if np.any(t < self.domain[0]):
            msg = "One (or more) of the input times is before the start of the trajectory"
            raise RuntimeError(msg)

        self._ensure_covers(t.max())

        in_original = t <= self._original.domain[1]
        out = np.empty((t.size, 3))
        if in_original.any():
            out[in_original] = getattr(self._original, method)(t[in_original])
        if (~in_original).any():
            out[~in_original] = getattr(self._extension, method)(t[~in_original])

        return out[0] if np.ndim(time) == 0 else out

    def position(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z position at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            position with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "position")

    def velocity(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate vx, vy, vz velocity at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "velocity")

    def acceleration(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate ax, ay, az acceleration at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            acceleration with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "acceleration")


__all__ = ["ExtendedCubicSplineOrbit"]
