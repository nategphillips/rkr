# rkradial

A Python implementation of the Rydberg-Klein-Rees method and radial Schrödinger equation.

## Roadmap

RKR:

- [ ] Implement iterative subdivision integration for $f$ and $g$
- [ ] Use Gauss-Mehler and Gauss-Legendre quadrature where appropriate

Radial:

- [x] Extrapolate data inward and outward using exponential and/or inverse-power terms
- [x] Compute Franck-Condon factors from wavefunction overlaps
- [ ] Interpolate using piecewise polynomials

## License and Copyright

Copyright (C) 2025 Nathan G. Phillips

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
