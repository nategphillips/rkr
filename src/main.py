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

m_oxygen: float = 15.999  # [amu]

mass: float = (m_oxygen * m_oxygen) / (m_oxygen + m_oxygen)

# Constants for O2 from the NIST Chemistry WebBook.

# [T_e, ω_e, ω_ex_e, ω_ey_e, ...]
g_consts_up: list[float] = [49793.28, 709.31, -10.65, -0.139]
# [B_e, α_e, γ_e, ...]
b_consts_up: list[float] = [0.81902, -0.01206, -5.56e-4]

g_consts_lo: list[float] = [0, 1580.193, -11.981, 0.04747]
b_consts_lo: list[float] = [1.4376766, -0.01593]


def g(v: int, g_consts: list[float]) -> float:
    x: float = v + 0.5

    return sum(val * x**idx for idx, val in enumerate(g_consts))


def b(v: int, b_consts: list[float]) -> float:
    x: float = v + 0.5

    return sum(val * x**idx for idx, val in enumerate(b_consts))


def integrand_f(v: int, upper_bound: int, g_consts: list[float]) -> float:
    return 1.0 / np.sqrt(g(upper_bound, g_consts) - g(v, g_consts))


def integrand_g(v: int, upper_bound: int, g_consts: list[float], b_consts: list[float]) -> float:
    return b(v, b_consts) / np.sqrt(g(upper_bound, g_consts) - g(v, g_consts))


def rkr(v: int, g_consts: list[float], b_consts: list[float]) -> tuple[float, float]:
    f: float = np.sqrt(hbar2_over_2 / mass) * quad(integrand_f, -0.5, v, args=(v, g_consts))[0]
    g: float = (
        np.sqrt(mass / hbar2_over_2) * quad(integrand_g, -0.5, v, args=(v, g_consts, b_consts))[0]
    )

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
    params, _, info, _, _ = curve_fit(fit_fn, xdata, ydata, maxfev=100000, full_output=True)
    print(f"Fit took {info['nfev']} iterations.")

    return params


def extrapolate_inner(
    rkr_sorted: NDArray[np.float64], energies_sorted: NDArray[np.float64], fn_type: str = "exp"
) -> tuple[NDArray[np.float64], Callable]:
    inner_points: NDArray[np.float64] = rkr_sorted[0:3]
    inner_energy: NDArray[np.float64] = energies_sorted[0:3]

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
    rkr_sorted: NDArray[np.float64],
    energies_sorted: NDArray[np.float64],
    g_consts: list[float],
    fn_type: str = "inv",
) -> tuple[NDArray[np.float64], Callable]:
    outer_points: NDArray[np.float64] = rkr_sorted[-3:]
    outer_energy: NDArray[np.float64] = energies_sorted[-3:]

    # The dissociation limit given in Herzberg is D_e = ω_e^2 / (4ω_ex_e), but I'm not sure how
    # accurate this is when the potential is solved to high vibrational quantum numbers.
    dissociation: float = g_consts[1] ** 2 / (4 * abs(g_consts[2]))

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


def get_bounds(
    v_max: int, g_consts: list[float], b_consts: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rkr_mins: NDArray[np.float64] = np.empty(v_max)
    rkr_maxs: NDArray[np.float64] = np.empty(v_max)
    energies: NDArray[np.float64] = np.empty(v_max)

    for v in range(0, v_max):
        r_min, r_max = rkr(v, g_consts, b_consts)

        rkr_mins[v] = r_min
        rkr_maxs[v] = r_max
        energies[v] = b(v, b_consts) + g(v, g_consts)

    plt.scatter(rkr_mins, energies)
    plt.scatter(rkr_maxs, energies)

    return rkr_mins, rkr_maxs, energies


def get_potential(
    r: NDArray[np.float64],
    rkr_mins: NDArray[np.float64],
    rkr_maxs: NDArray[np.float64],
    energies: NDArray[np.float64],
    g_consts: list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rkr_all: NDArray[np.float64] = np.concatenate((rkr_mins, rkr_maxs))
    energies_all: NDArray[np.float64] = np.concatenate((energies, energies))

    # The x values in CubicSpline must be listed in increasing order, so sort to ensure this.
    sorted_indices: NDArray[np.int64] = np.argsort(rkr_all)

    rkr_sorted: NDArray[np.float64] = rkr_all[sorted_indices]
    energies_sorted: NDArray[np.float64] = energies_all[sorted_indices]

    cubic_spline: CubicSpline = CubicSpline(rkr_sorted, energies_sorted)

    params_inner, fit_innter = extrapolate_inner(rkr_sorted, energies_sorted)
    params_outer, fit_outer = extrapolate_outer(rkr_sorted, energies_sorted, g_consts)

    rkr_min: float = rkr_sorted[0]
    rkr_max: float = rkr_sorted[-1]

    lmask: NDArray[np.bool] = r < rkr_min
    mmask: NDArray[np.bool] = (r >= rkr_min) & (r <= rkr_max)
    rmask: NDArray[np.bool] = r > rkr_max

    potential: NDArray[np.float64] = np.empty_like(r)

    potential[lmask] = fit_innter(r[lmask], *params_inner)
    potential[mmask] = cubic_spline(r[mmask])
    potential[rmask] = fit_outer(r[rmask], *params_outer)

    plt.plot(r[lmask], potential[lmask], color="blue")
    plt.plot(r[mmask], potential[mmask], color="black")
    plt.plot(r[rmask], potential[rmask], color="red")

    return r, potential


def main() -> None:
    v_max_up: int = 15
    v_max_lo: int = 40

    dim: int = 1000

    rkr_mins_up, rkr_maxs_up, energies_up = get_bounds(v_max_up, g_consts_up, b_consts_up)
    rkr_mins_lo, rkr_maxs_lo, energies_lo = get_bounds(v_max_lo, g_consts_lo, b_consts_lo)

    r_min: float = min(rkr_mins_up.min(), rkr_mins_lo.min())
    r_max: float = max(rkr_maxs_up.max(), rkr_maxs_lo.max())

    r: NDArray[np.float64] = np.linspace(r_min, r_max, dim)

    r_up, potential_up = get_potential(r, rkr_mins_up, rkr_maxs_up, energies_up, g_consts_up)
    r_lo, potential_lo = get_potential(r, rkr_mins_lo, rkr_maxs_lo, energies_lo, g_consts_lo)

    eigvals_up, wavefns_up = radial_schrodinger(r_up, v_max_up, potential_up, dim)
    eigvals_lo, wavefns_lo = radial_schrodinger(r_lo, v_max_lo, potential_lo, dim)

    scaling_factor: int = 500

    for i, psi in enumerate(wavefns_up):
        plt.plot(r_up, psi * scaling_factor + eigvals_up[i])

    for i, psi in enumerate(wavefns_lo):
        plt.plot(r_lo, psi * scaling_factor + eigvals_lo[i])

    plt.xlabel(r"Internuclear Distance, $r$ [$\AA$]")
    plt.ylabel(r"Rovibrational Energy, $B(v) + G(v)$ [cm$^{-1}$]")
    plt.show()


if __name__ == "__main__":
    main()
