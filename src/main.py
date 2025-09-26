# module main.py
"""Uses the Rydberg-Klein-Rees method to obtain the internuclear potential V(r)."""

# Copyright (C) 2025 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.integrate import quad

plt.style.use(["science", "grid"])

# Equal to 0.5 * ћ^2 in the appropriate units. Given in "RKR1" by LeRoy.
factor: float = 16.857629206  # [amu * Å^2 * cm^-1]

m_carbon: float = 12.011  # [amu]
m_oxygen: float = 15.999  # [amu]

mass: float = (m_carbon * m_oxygen) / (m_carbon + m_oxygen)


def g(v: int) -> float:
    x: float = v + 0.5

    return (
        2169.81801 * x
        - 13.2906899 * x**2
        + 1.09777979e-2 * x**3
        + 2.29371618e-5 * x**4
        + 2.10035541e-6 * x**5
        - 4.49979099e-8 * x**6
    )


def b(v: int) -> float:
    x: float = v + 0.5

    return 1.93126515 - 1.75054229e-2 * x + 1.81117949e-7 * x**2


def integrand_f(v: int, upper_bound: int) -> float:
    return 1.0 / np.sqrt(g(upper_bound) - g(v))


def integrand_g(v: int, upper_bound: int) -> float:
    return b(v) / np.sqrt(g(upper_bound) - g(v))


def rkr(v: int) -> tuple[float, float]:
    f: float = np.sqrt(factor / mass) * quad(integrand_f, -0.50009, v, args=(v))[0]
    g: float = np.sqrt(mass / factor) * quad(integrand_g, -0.50009, v, args=(v))[0]

    sqrt_term: float = np.sqrt(f**2 + f / g)

    r_min: float = sqrt_term - f  # [Å]
    r_max: float = sqrt_term + f  # [Å]

    return r_min, r_max


def main() -> None:
    v_max: int = 50

    r_mins: list[float] = []
    r_maxs: list[float] = []
    energies: list[float] = []

    for v in range(0, v_max):
        r_min, r_max = rkr(v)

        r_mins.append(r_min)
        r_maxs.append(r_max)
        energies.append(g(v))

    plt.plot(r_mins, energies, "bo")
    plt.plot(r_maxs, energies, "ro")

    plt.xlabel(r"Internuclear Distance, $r$ [$\AA$]")
    plt.ylabel(r"Vibrational Energy, $G(v)$ [cm$^{-1}$]")
    plt.show()


if __name__ == "__main__":
    main()
