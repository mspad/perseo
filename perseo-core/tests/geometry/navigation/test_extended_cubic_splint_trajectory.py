# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/navigation/extended_cubic_spline_trajectory.py ExtendedCubicSplineOrbit object"""

import numpy as np
import pytest

from perseo_core.geometry.navigation import ExtendedCubicSplineOrbit, Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime


class TestFloorToMicrosecond:
    def test_whole_second_is_unchanged(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=0
        )
        assert ExtendedCubicSplineOrbit._floor_to_microsecond(t) == t

    def test_sub_microsecond_part_is_dropped(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_234_567
        )
        floored = ExtendedCubicSplineOrbit._floor_to_microsecond(t)
        assert floored.picosecond_of_second == 1_000_000
        assert floored.second_of_minute == 3


class TestOrbit:
    @pytest.fixture(autouse=True)
    def setup_orbit_data(self, orbit_test_data: dict) -> None:
        """Load test data from fixtures."""
        self._time_axis = orbit_test_data["time_axis"]
        self._positions = orbit_test_data["positions"]
        self._velocities = orbit_test_data["velocities"]
        self._tolerance = orbit_test_data["tolerance"]

        self._N = 20  # number of state vectors used for actual initialization
        self._extended_orbit = ExtendedCubicSplineOrbit(
            times=self._time_axis[0 : self._N],
            positions=self._positions[0 : self._N],
            velocities=self._velocities[0 : self._N],
        )

    def test_trajectory_subclass(self) -> None:
        """Test that ExtendedCubicSplineOrbit is subclass of Trajectory protocol."""
        assert issubclass(ExtendedCubicSplineOrbit, Trajectory)

    def test_trajectory_creation(self) -> None:
        """Test ExtendedCubicSplineOrbit constructor creates valid instance."""
        assert isinstance(self._extended_orbit, ExtendedCubicSplineOrbit)

    def test_trajectory_properties(self) -> None:
        """Test that ExtendedCubicSplineOrbit properties return correct times, positions, velocities."""
        np.testing.assert_array_equal(self._extended_orbit.positions, self._positions[0 : self._N])
        np.testing.assert_array_equal(self._extended_orbit.velocities, self._velocities[0 : self._N])
        delta_times = self._extended_orbit.times - self._time_axis[0 : self._N]
        np.testing.assert_array_equal(delta_times.astype(float), np.zeros_like(delta_times, dtype=float))

    def test_extended_orbit_methods(self) -> None:
        """Test ExtendedCubicSplineOrbit interpolation and propagation methods for position, velocity, acceleration."""
        np.testing.assert_allclose(
            self._extended_orbit.position(self._time_axis),
            self._positions,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self._extended_orbit.velocity(self._time_axis),
            self._velocities,
            atol=self._tolerance,
            rtol=0,
        )

    def test_warning_past_three_hour(self) -> None:
        with pytest.warns(RuntimeWarning, match="past the end of the original trajectory"):
            self._extended_orbit.position(self._time_axis[self._N] + 60 * 60 * 3 + 1)

    def test_changing_domain(self) -> None:
        orbit = ExtendedCubicSplineOrbit(times=self._time_axis, positions=self._positions, velocities=self._velocities)
        initial_domain = orbit.domain
        orbit.position(time=initial_domain[1] + 600)
        final_domain = orbit.domain
        np.testing.assert_equal(initial_domain[0], final_domain[0])
        assert final_domain[1] > initial_domain[1]

    def test_time_before_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before the start"):
            self._extended_orbit.position(self._time_axis[0] - 1.0)

    def test_array_with_one_time_before_start_raises(self) -> None:
        times = np.array([self._time_axis[0] - 0.5, self._time_axis[1]])
        with pytest.raises(RuntimeError, match="before the start"):
            self._extended_orbit.velocity(times)

    @pytest.mark.parametrize("bad_shape", [(5,), (5, 2), (5, 3, 1)])
    def test_positions_shape(self, bad_shape: tuple) -> None:
        with pytest.raises(ValueError, match="Positions"):
            ExtendedCubicSplineOrbit(self._extended_orbit.times, np.zeros(bad_shape), self._extended_orbit.velocities)

    @pytest.mark.parametrize("bad_shape", [(5,), (5, 2), (5, 3, 1)])
    def test_velocities_shape(self, bad_shape: tuple) -> None:
        with pytest.raises(ValueError, match="Velocities"):
            ExtendedCubicSplineOrbit(self._extended_orbit.times, self._extended_orbit.positions, np.zeros(bad_shape))

    def test_length_mismatch(self) -> None:
        times = self._time_axis
        pos = self._positions
        vel = self._velocities
        with pytest.raises(ValueError, match="same number of samples"):
            ExtendedCubicSplineOrbit(times, pos[:-1], vel)
        with pytest.raises(ValueError, match="same number of samples"):
            ExtendedCubicSplineOrbit(times, pos, vel[:-1])
