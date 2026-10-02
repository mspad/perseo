# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Extended Cubic Spline Trajectory module.

This module provides the `ExtendedCubicSplineTrajectory` class, which enables orbit propagation
outside of the domain defined by state vectors
"""

from __future__ import annotations

import os
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import satkit as sk

from perseo_core.geometry.coordinates.conversions import ecef2eci, eci2ecef
from perseo_core.geometry.navigation import CubicSplineTrajectory
from perseo_core.geometry.navigation.trajectory import Trajectory
from perseo_core.timing import PreciseDateTime

T = TypeVar("T", bound=np.generic)

os.environ["SATKIT_JPLEPHEM_FILE"] = "lnxp1900p2053.421"


class ExtendedCubicSplineOrbit(Trajectory[T]):
    """Trajectory based on a Cubic Spline interpolator."""

    def __init__(
        self,
        times: npt.NDArray[T],
        positions: npt.NDArray[np.floating],
        velocities: npt.NDArray[np.floating],
        extension_duration_s: int,
    ) -> None:
        """Create a ExtendedCubicSplineOrbit from state vectors: times, positions and velocities.

        Times must be of type T, either dates or floats.

        Positions and velocities must be specified as (N, 3) arrays of floats.

        ExtendedCubicSplineOrbit can extend trajectory for low orbit satellites outside the boundaries of time axis
        through numerical integration.

        Parameters
        ----------
        times : npt.NDArray[T]
            time axis as numpy array of shape (N,)
        positions : npt.NDArray[np.floating]
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : npt.NDArray[np.floating]
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z
        extension_duration_s : int
            seconds for which the orbit is extended

        """
        if extension_duration_s <= 0 or int(extension_duration_s) != extension_duration_s:
            msg = "Extension_duration_s must be a positive integer"
            raise ValueError(msg)

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
        self._domain: tuple[T, T] = (times[0], times[-1] + extension_duration_s)

        t_anchor = self._floor_to_microsecond(times[-1])
        ecef_pos_anchor = self._original.position(t_anchor)
        ecef_vel_anchor = self._original.velocity(t_anchor)
        ext_times = np.arange(0, extension_duration_s + 1, dtype=float) + t_anchor
        sk_ext_times = self._to_satkit_time(ext_times)

        eci_pos_anchor, ecu_vel_anchor = ecef2eci(ecef_pos_anchor, ecef_vel_anchor, t_anchor)
        state = np.hstack([eci_pos_anchor, ecu_vel_anchor])
        result = sk.propagate(
            state,
            sk_ext_times[0],
            end=sk_ext_times[-1],
            propsettings=sk.propsettings(
                gravity_model=sk.gravmodel.egm96,
                gravity_degree=10,
                gravity_order=10,
            ),
        )
        evaluation = np.atleast_2d(result.interp(sk_ext_times))
        ext_pos_eci, ext_vel_eci = evaluation[:, 0:3], evaluation[:, 3:6]
        ext_pos, ext_vel = eci2ecef(ext_pos_eci, ext_vel_eci, ext_times)
        self._extension = CubicSplineTrajectory(ext_times, ext_pos, ext_vel)

    @staticmethod
    def _to_satkit_time(times: PreciseDateTime | npt.NDArray) -> sk.time | npt.NDArray:
        if isinstance(times, PreciseDateTime):
            return sk.time(
                times.year,
                times.month,
                times.day_of_the_month,
                times.hour_of_day,
                times.minute_of_hour,
                times.second_of_minute + times.picosecond_of_second * 1e-12,
            )
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
    def domain(self) -> tuple[T, T]:
        """Trajectory time domain."""
        return self._domain

    def _evaluate(self, time: PreciseDateTime | npt.NDArray, method: str) -> np.ndarray:
        t = np.atleast_1d(time)
        if np.any(t < self._domain[0]) or np.any(t > self._domain[1]):
            msg = "One (or more) of the input times is outside of trajectory time boundaries"
            raise RuntimeError(msg)

        in_original = t <= self._original.times[-1]
        out = np.empty((t.size, 3))
        if in_original.any():
            out[in_original] = getattr(self._original, method)(t[in_original])
        if (~in_original).any():
            out[~in_original] = getattr(self._extension, method)(t[~in_original])

        return out[0] if np.ndim(time) == 0 else out

    def position(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z position at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            position with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "position")

    def velocity(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate vx, vy, vz velocity at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "velocity")

    def acceleration(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Evaluate ax, ay, az acceleration at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        np.ndarray
            acceleration with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "acceleration")


__all__ = ["ExtendedCubicSplineOrbit"]
