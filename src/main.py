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

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from numpy.typing import NDArray
from scipy.integrate import quad, trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.sparse import diags_array

plt.style.use(["science", "grid"])

# Equal to 0.5 * ћ^2 in the appropriate units. Given in "RKR1" by LeRoy.
hbar2_over_2: float = 16.857629206  # [amu * Å^2 * cm^-1]

m_carbon: float = 12.011  # [amu]
m_oxygen: float = 15.999  # [amu]

mass: float = (m_carbon * m_oxygen) / (m_carbon + m_oxygen)

# Constants for CO taken from "Rydberg-Klein-Rees Potential for the X1Σ+ State of the CO Molecule"
# by Mantz.
g_consts: list[float] = [
    2169.81801,
    -13.2906899,
    1.09777979e-2,
    2.29371618e-5,
    2.10035541e-6,
    -4.49979099e-8,
]

b_consts: list[float] = [1.93126515, -1.75054229e-2, 1.81117949e-7]


def g(v: int) -> float:
    x: float = v + 0.5

    return sum(val * x ** (idx + 1) for idx, val in enumerate(g_consts))


def b(v: int) -> float:
    x: float = v + 0.5

    return sum(val * x**idx for idx, val in enumerate(b_consts))


def integrand_f(v: int, upper_bound: int) -> float:
    return 1.0 / np.sqrt(g(upper_bound) - g(v))


def integrand_g(v: int, upper_bound: int) -> float:
    return b(v) / np.sqrt(g(upper_bound) - g(v))


def rkr(v: int) -> tuple[float, float]:
    f: float = np.sqrt(hbar2_over_2 / mass) * quad(integrand_f, -0.5, v, args=(v))[0]
    g: float = np.sqrt(mass / hbar2_over_2) * quad(integrand_g, -0.5, v, args=(v))[0]

    sqrt_term: float = np.sqrt(f**2 + f / g)

    r_min: float = sqrt_term - f  # [Å]
    r_max: float = sqrt_term + f  # [Å]

    return r_min, r_max


def radial_schrodinger(
    r: NDArray[np.float64],
    v_max: int,
    potential_term: NDArray,
    dim: int,
    j_qn: int = 0,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    dr: float = r[1] - r[0]

    # Construct the kinetic energy operator via a second-order central finite difference. A sparse
    # array is used to save space.
    kinetic_term: NDArray[np.float64] = (-hbar2_over_2 / (mass * dr**2)) * diags_array(
        [1, -2, 1],
        offsets=[-1, 0, 1],  # pyright: ignore[reportArgumentType]
        shape=(dim, dim),
    ).toarray()

    # The rotational operator. Not sure if this is needed if I'm only interested in the vibrational
    # wavefunctions since J = 0 will cancel it out anyway. Might want to check out
    # https://onlinelibrary.wiley.com/doi/10.1155/2018/1487982.
    rotational_term: NDArray[np.float64] = (hbar2_over_2 / (mass * r**2)) * j_qn * (j_qn + 1)

    hamiltonian = kinetic_term + np.diag(rotational_term + potential_term)

    # The Hamiltonian will always be Hermitian, so the use of eigh is warranted here.
    eigvals, eigvecs = eigh(hamiltonian)

    norm_wavefns: list[NDArray[np.float64]] = []

    # Normalize the wavefunctions ψ(r) such that ∫ ψ'ψ dr = 1.
    for i in range(v_max):
        wavefn: NDArray[np.float64] = eigvecs[:, i]
        norm: float = trapezoid(wavefn**2, r)
        norm_wavefns.append(wavefn / np.sqrt(norm))

    return eigvals[:v_max], norm_wavefns


def plot_extrapolation(
    fit_fn: Callable,
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
) -> NDArray[np.float64]:
    params, _, info, _, _ = curve_fit(fit_fn, xdata, ydata, maxfev=20000, full_output=True)
    print(f"Fit took {info['nfev']} iterations.")

    return params


def extrapolate_inner(
    r_sorted: NDArray[np.float64], g_sorted: NDArray[np.float64], fn_type: str = "exp"
) -> tuple[NDArray[np.float64], Callable]:
    inner_points: NDArray[np.float64] = r_sorted[0:3]
    inner_energy: NDArray[np.float64] = g_sorted[0:3]

    # LeRoy's LEVEL extrapolates the potential inward with an exponential function fitted to the
    # first three points.
    def fit(x, a, b):
        match fn_type:
            case "exp":
                return a * np.exp(-b * x)

            case "inv":
                return a / x**b

    params: NDArray[np.float64] = plot_extrapolation(fit, inner_points, inner_energy)

    return params, fit


def extrapolate_outer(
    r_sorted: NDArray[np.float64], g_sorted: NDArray[np.float64], fn_type: str = "exp"
) -> tuple[NDArray[np.float64], Callable]:
    outer_points: NDArray[np.float64] = r_sorted[-3:]
    outer_energy: NDArray[np.float64] = g_sorted[-3:]

    # The dissociation limit given in Herzberg is D_e = ω_e^2 / (4ω_ex_e), but I'm not sure how
    # accurate this is when the potential is solved to high vibrational quantum numbers.
    dissociation: float = g_consts[0] ** 2 / (4 * abs(g_consts[1]))

    # All three of these fit functions are given in the documentation for LeRoy's LEVEL.
    def fit(x, a, b, c):
        match fn_type:
            case "exp":
                return dissociation - a * np.exp(-b * (x - c) ** 2)
            case "inv":
                return dissociation - a / x**b
            case "mix":
                return dissociation - a * x**b * np.exp(-c * x)

    params: NDArray[np.float64] = plot_extrapolation(fit, outer_points, outer_energy)

    return params, fit


def main() -> None:
    v_max: int = 50
    dim: int = 1000

    r_mins: list[float] = []
    r_maxs: list[float] = []
    energies: list[float] = []

    plt.scatter(r_mins, energies)
    plt.scatter(r_maxs, energies)

    for v in range(0, v_max):
        r_min, r_max = rkr(v)

        r_mins.append(r_min)
        r_maxs.append(r_max)
        energies.append(b(v) + g(v))

    r_all: list[float] = r_mins + r_maxs
    g_all: list[float] = energies + energies

    # The x values in CubicSpline must be listed in increasing order, so sort to ensure this.
    sorted_indices: NDArray[np.int64] = np.argsort(r_all)

    r_sorted: NDArray[np.float64] = np.array(r_all)[sorted_indices]
    g_sorted: NDArray[np.float64] = np.array(g_all)[sorted_indices]

    cubic_spline: CubicSpline = CubicSpline(r_sorted, g_sorted)
    r: NDArray[np.float64] = np.linspace(r_min - 0.2, r_max + 2, dim)

    params_inner, fit_innter = extrapolate_inner(r_sorted, g_sorted)
    params_outer, fit_outer = extrapolate_outer(r_sorted, g_sorted)

    lmask: NDArray[np.bool] = r < r_min
    mmask: NDArray[np.bool] = (r >= r_min) & (r <= r_max)
    rmask: NDArray[np.bool] = r > r_max

    potential: NDArray[np.float64] = np.empty_like(r)

    potential[lmask] = fit_innter(r[lmask], *params_inner)
    potential[mmask] = cubic_spline(r[mmask])
    potential[rmask] = fit_outer(r[rmask], *params_outer)

    plt.plot(r[lmask], potential[lmask], color="blue")
    plt.plot(r[mmask], potential[mmask], color="black")
    plt.plot(r[rmask], potential[rmask], color="red")

    eigvals, wavefns = radial_schrodinger(r, v_max, potential, dim)

    scaling_factor: int = 500

    for i, psi in enumerate(wavefns):
        plt.plot(r, psi * scaling_factor + eigvals[i])

    plt.xlabel(r"Internuclear Distance, $r$ [$\AA$]")
    plt.ylabel(r"Rovibrational Energy, $B(v) + G(v)$ [cm$^{-1}$]")
    plt.show()


if __name__ == "__main__":
    main()
