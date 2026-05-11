from sympy.physics.units import length

from openpnm.models.physics._utils import _poisson_conductance
from openpnm.models import _doctxt
import warnings
import numpy as np
from scipy import integrate

__all__ = ["generic_thermal",
           "series_resistors",
           "yovanovich",
           "dixon_bridge_model",
           "extended_dixon_bridge_model",
           "batchelor",
           "kunii_smith",
           "tsotsas_bob",
           "argento",
           "fei_narsilio",
           "birkholz",
           "zehner_bauer_schlunder",
           "tsotsas_zbs"
           ]


@_doctxt
def generic_thermal(phase,
                    pore_conductivity='pore.thermal_conductivity',
                    throat_conductivity='throat.thermal_conductivity',
                    size_factors='throat.diffusive_size_factors'):
    r"""
    Calculate the thermal conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_conductivity : str
        %(dict_blurb)s thermal conductivity
    throat_conductivity : str
        %(dict_blurb)s throat thermal conductivity
    size_factors : str
        %(dict_blurb)s conduit diffusive size factors

    Returns
    -------
    %(return_arr)s thermal conductance

    """
    return _poisson_conductance(phase=phase,
                                pore_conductivity=pore_conductivity,
                                throat_conductivity=throat_conductivity,
                                size_factors=size_factors)


@_doctxt
def series_resistors(
        phase,
        pore_thermal_conductivity='pore.thermal_conductivity',
        throat_thermal_conductivity='throat.thermal_conductivity',
        size_factors='throat.diffusive_size_factors'
):
    r"""
    Calculate the thermal conductance of conduits in network.

    Parameters
    ----------
    %(phase)s
    pore_conductivity : str
        %(dict_blurb)s thermal conductivity
    throat_conductivity : str
        %(dict_blurb)s throat thermal conductivity
    size_factors : str
        %(dict_blurb)s conduit diffusive size factors

    Returns
    -------
    %(return_arr)s thermal conductance

    """
    return _poisson_conductance(phase=phase,
                                pore_conductivity=pore_thermal_conductivity,
                                throat_conductivity=throat_thermal_conductivity,
                                size_factors=size_factors)


def yovanovich(solid_p,
                throat_solid_conductivity="throat.thermal_solid_conductivity",
                relative_contact_radius="throat.relative_contact_throat_radius",
                mean_curvature="throat.mean_curvature",
                relative_gas_conductivity_radius=1e12,
                radiation_exchange_factor=0
                ):
    r"""
    Calculates conductance based on Yovanovich 1967 contact model. DOI: 10.2514/3.28821
    The model was original developed for thermal conductance thrpugh ball bearings in space.
    Currently, only conduction is implemented as the remaining resistances proved to be unreliabl. The parameter of
    relative gas contact radius remains a mystery.

    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    relative_contact_radius : str
        %(dict_burb)s relative contact radius
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    Returns
    -------
    %(return_arr)s

    """
    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    # stef_bolz_const = 5.670374419e-8
    # radiation_conductance = self.r_particle**2*np.pi*radiation_exchange_factor*4*stef_bolz_const*T_loc
    # Nelze implementovat, neumime lokalni teplotu
    # fluid_conductance = 7.1 * fluid_conductivity / (
    #   relative_contact_radius ** 2 * (relative_gas_conductivity_radius ** 2 - 5.1 / relative_gas_conductivity_radius))

    # Calculates the radius of expected contact area
    contact_radius = net[relative_contact_radius] * net[mean_curvature]

    # calculates the resistance coming from the conduction through solid
    solid_resistance = 1 / (2 * solid_p[throat_solid_conductivity] * contact_radius) - np.log(2) / (
            np.pi * solid_p[throat_solid_conductivity] * net[mean_curvature])

    return (1 / solid_resistance)


# ---Main chapter---
def dixon_bridge_model(
    solid_p,
    throat_solid_conductivity="throat.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    relative_bridge_radius="throat.relative_bridge_radius",
    diameter="pore.diameter",
    throat_length="throat.length",
):
    r"""
    Calculates conductance based on the Bridge model published by Dixon et al. (2013),
    using a vectorized analytical solution for the radial integral.

    Notes
    -----
    The original implementation numerically evaluates, for each throat:

        I = ∫_0^{r_bridge} r / (h_f(r) * k_s + h_s(r) * k_f) dr

    This version replaces that numerical quadrature with the closed-form solution.

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].
    """
    net = solid_p.network

    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn(
            f"No {relative_bridge_radius} provided in solid phase. "
            f"Using default value of 0.1"
        )

    # Particle radius is taken as the smaller of the two connected pores
    r_particle = np.min(net[diameter][net.conns], axis=1)/2

    # Bridge radius as a fraction of particle radius
    r_bridge = r_particle * net[relative_bridge_radius]

    # Numerical safety: enforce 0 <= r_bridge <= r_particle
    r_bridge = np.clip(r_bridge, 0.0, r_particle)

    # Length from contact point to symmetry plane
    # h_f(a) with a = r_bridge
    # np.max is used to avoid sqrt of negative due to floating point issues when r_bridge is very close to r_particle
    sqrt_term = np.sqrt(np.maximum(r_particle**2 - r_bridge**2, 0.0))
    length_bridge = r_particle - sqrt_term

    # Conductivities
    k_s = np.asarray(solid_p[throat_solid_conductivity], dtype=float)
    k_f = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)

    # Analytical integral from 0 to r_bridge
    #
    # I = L / (k_f - k_s)
    #   + [k_f*L + (k_s - k_f)*R] / (k_f - k_s)^2 * ln(|k_s / k_f|)
    #
    # with special case k_f == k_s:
    # I = r_bridge^2 / (2 * k_f * L)
    integral_value = np.empty_like(r_bridge, dtype=float)

    equal_k = np.isclose(k_f, k_s)
    diff_k = ~equal_k

    if np.any(diff_k):
        B = k_f[diff_k] - k_s[diff_k]
        R = r_particle[diff_k]
        L = length_bridge[diff_k]
        A = k_f[diff_k] * L + (k_s[diff_k] - k_f[diff_k]) * R

        integral_value[diff_k] = (
            L / B + (A / B**2) * np.log(np.abs(k_s[diff_k] / k_f[diff_k]))
        )

    if np.any(equal_k):
        # When k_f == k_s, denominator becomes constant = k_f * length_bridge
        # Use the limiting expression directly
        integral_value[equal_k] = (
            r_bridge[equal_k] ** 2 / (2.0 * k_f[equal_k] * length_bridge[equal_k])
        )

    # Total bridge length (full bridge = 2 * half-bridge + throat gap)
    bridge_length = 2.0 * length_bridge + net[throat_length]

    # Cross-sectional area
    bridge_crosssection = np.pi * r_bridge**2

    # Effective conductivity (same definition as original code)
    effective_conductivity = 2.0 * length_bridge / (r_bridge**2) * integral_value * (k_s * k_f)

    # Conductance [W/K]
    conductance = effective_conductivity * bridge_crosssection / bridge_length

    return conductance


def extended_dixon_bridge_model(
    solid_p,
    pore_solid_conductivity="pore.thermal_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    relative_bridge_radius="throat.relative_bridge_radius",
    diameter="pore.diameter",
    throat_length="throat.length",
):
    r"""
    Calculate thermal conductance of an asymmetric particle bridge.

    This implementation extends the Dixon bridge model to the case where the two
    connected spheres may have different sizes and different solid
    conductivities. The bridge is decomposed into three thermal resistances in
    series:

    1. conduction through the bridge region on pore 1 side,
    2. conduction through the fluid gap in the throat,
    3. conduction through the bridge region on pore 2 side.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing thermal conductivity data and a reference to the
        associated network.
    pore_solid_conductivity : str, optional
        Dictionary key of the pore solid thermal conductivity values [W/m.K].
    throat_fluid_conductivity : str, optional
        Dictionary key of the throat fluid thermal conductivity values [W/m.K].
    relative_bridge_radius : str, optional
        Dictionary key of the relative bridge radius. The bridge radius is
        calculated as the specified fraction of the smaller connected particle
        size.
    diameter : str, optional
        Dictionary key of the pore size values used to determine the particle
        size on each side of the throat.
    throat_length : str, optional
        Dictionary key of the throat gap length [m].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    Notes
    -----
    For each throat, a common bridge radius is defined from the smaller of the
    two connected particle sizes:

    .. math::

        r_b = \min(R_1, R_2)\,\xi

    where :math:`\xi` is the relative bridge radius.

    For each sphere side, the bridge-side conductance is computed analytically
    from the corresponding particle size and solid conductivity. The fluid gap
    is modeled as a cylinder of radius :math:`r_b` and length
    ``throat_length``. The total conductance is obtained by placing the two
    bridge-side resistances and the throat-gap resistance in series.

    If ``relative_bridge_radius`` is not present in the network, a default value
    of ``0.1`` is assigned and a warning is issued.
    """

    # ---Main chapter---
    def _single_sphere_bridge_conductance(r_particle, r_bridge, k_s, k_f):
        """
        Analytical conductance of one sphere-side bridge segment.

        Parameters
        ----------
        r_particle : ndarray
            Radius-like particle size for one side of each throat.
        r_bridge : ndarray
            Common bridge radius for each throat.
        k_s : ndarray
            Solid conductivity for the given pore side [W/m.K].
        k_f : ndarray
            Fluid conductivity for the throat [W/m.K].

        Returns
        -------
        ndarray
            Conductance of the sphere-side bridge segment [W/K].
        """
        r_particle = np.asarray(r_particle, dtype=float)
        r_bridge = np.asarray(r_bridge, dtype=float)
        k_s = np.asarray(k_s, dtype=float)
        k_f = np.asarray(k_f, dtype=float)

        # Ensure the bridge radius is physically valid for this sphere
        r_bridge_local = np.clip(r_bridge, 0.0, r_particle)

        # Half-bridge length for this sphere side
        sqrt_term = np.sqrt(np.maximum(r_particle**2 - r_bridge_local**2, 0.0))
        length_bridge = r_particle - sqrt_term

        integral_value = np.zeros_like(r_bridge_local, dtype=float)

        # Only positive, physically meaningful cases can conduct
        valid = (
            (r_bridge_local > 0.0)
            & (length_bridge > 0.0)
            & (k_s > 0.0)
            & (k_f > 0.0)
        )

        if not np.any(valid):
            return integral_value

        equal_k = valid & np.isclose(k_f, k_s)
        diff_k = valid & ~np.isclose(k_f, k_s)

        # Closed form for k_f != k_s
        if np.any(diff_k):
            R = r_particle[diff_k]
            L = length_bridge[diff_k]
            ks = k_s[diff_k]
            kf = k_f[diff_k]

            A = kf * L + (ks - kf) * R
            B = kf - ks

            integral_value[diff_k] = (
                L / B
                + (A / B**2) * np.log(ks / kf)
            )

        # Limiting expression for k_f == k_s
        if np.any(equal_k):
            integral_value[equal_k] = (
                r_bridge_local[equal_k] ** 2
                / (2.0 * k_f[equal_k] * length_bridge[equal_k])
            )

        # G_side = 2*pi*k_s*k_f*I
        conductance = 2.0 * np.pi * k_s * k_f * integral_value

        # Remove tiny negative values caused by floating-point noise
        return np.maximum(conductance, 0.0)

    net = solid_p.network

    # ---Main chapter---
    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn(
            "No relative bridge radius provided in solid phase. "
            "Using default value of 0.1"
        )

    conns = net.conns

    # ---Main chapter---
    # Particle size on each connected side
    r1 = np.asarray(net[diameter][conns[:, 0]], dtype=float)/2
    r2 = np.asarray(net[diameter][conns[:, 1]], dtype=float)/2

    # Common bridge radius is based on the smaller connected particle
    r_min = np.minimum(r1, r2)
    r_bridge = np.asarray(r_min * net[relative_bridge_radius], dtype=float)

    # Solid conductivity on each pore side
    ks1 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 0]], dtype=float)
    ks2 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 1]], dtype=float)

    # Throat fluid conductivity and gap length
    kf = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)
    Lt = np.asarray(net[throat_length], dtype=float)

    # ---Main chapter---
    # Sphere-side bridge conductances
    G1 = _single_sphere_bridge_conductance(
        r_particle=r1,
        r_bridge=r_bridge,
        k_s=ks1,
        k_f=kf,
    )

    G2 = _single_sphere_bridge_conductance(
        r_particle=r2,
        r_bridge=r_bridge,
        k_s=ks2,
        k_f=kf,
    )

    # ---Main chapter---

    # Fluid gap conductance through a cylindrical throat region
    area = np.pi * r_bridge ** 2

    G_gap = np.zeros_like(r_bridge, dtype=float)

    # Positive throat length -> cylindrical fluid conductance
    mask_gap = (area > 0.0) & (kf > 0.0) & (Lt > 0.0)
    G_gap[mask_gap] = kf[mask_gap] * area[mask_gap] / Lt[mask_gap]

    # Zero or negative throat length -> effectively no gap resistance
    mask_gap_short = (area > 0.0) & (kf > 0.0) & (Lt <= 0.0)
    G_gap[mask_gap_short] = np.inf

    # Series combination:
    # 1 / G_total = 1 / G1 + 1 / G_gap + 1 / G2
    total_resistance = np.zeros_like(r_bridge, dtype=float)

    mask1 = G1 > 0.0
    maskg = G_gap > 0.0
    mask2 = G2 > 0.0

    total_resistance[mask1] += 1.0 / G1[mask1]
    total_resistance[maskg] += 1.0 / G_gap[maskg]
    total_resistance[mask2] += 1.0 / G2[mask2]

    conductance = np.zeros_like(r_bridge, dtype=float)
    valid = total_resistance > 0.0
    conductance[valid] = 1.0 / total_resistance[valid]

    return conductance


def batchelor(solid_p,
              pore_thermal_conductivity="pore.thermal_conductivity",
              throat_solid_conductivity="throat.thermal_solid_conductivity",
              throat_fluid_conductivity="throat.thermal_fluid_conductivity",
              mean_curvature="throat.mean_curvature",
              relative_contact_radius="throat.relative_contact_throat_radius",
              throat_length="throat.length",
              effective_mean_radius_curvature_fraction="throat.relative_mean_curvature"
              ):
    r"""
    Calculates conductance of the bridge model based on Batchelor, O'Brien 1977 model, presented by Yun and Evans (2010).
    They suggest  the fraction of the mean radius of curvature should be 0.5 for dry and 0.8 for wet conditions.
    Value of 0.25 was used in the more recent article by Fei, Wenbin; Narsilio, Guillermo  10.1016/j.jrmge.2021.08.008.
    They however used it for intra particle conductivity.

    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    throat_solid_conductivity : str
        %(dict_burb)s throat thermal conductivity
    effective_mean_radius_curvature_fraction : str
        %(dict_burb)s mean radius of curvature fraction
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    throat_length : str
        %(dict_burb)s throat length

    Returns
    -------
    %(return_arr)s

    """
    net = solid_p.network

    if effective_mean_radius_curvature_fraction not in net.keys():
        net[effective_mean_radius_curvature_fraction] = 0.5
        warnings.warn(
            f"No {effective_mean_radius_curvature_fraction} provided in solid phase. Using default value of 0.5")

    if relative_contact_radius not in net.keys():
        no_contact = True
    #    elif solid_p[relative_contact_radius] <= 0:
    #       no_contact = True DODELAT - potreba kontroly pro kazdy element
    else:
        no_contact = False

    conductivity_ratios = (solid_p[throat_solid_conductivity] /
                           solid_p[throat_fluid_conductivity])  # alpha in text
    effective_mean_particle_radius = net[mean_curvature] * net[effective_mean_radius_curvature_fraction]

    Cc = np.zeros_like(conductivity_ratios)
    C1p = np.zeros_like(conductivity_ratios)
    C2p = np.zeros_like(conductivity_ratios)
    if no_contact:
        separation_parameters = np.square(conductivity_ratios) * net[throat_length] / net[
            mean_curvature]  # lambda in text
        for i, separation_param in enumerate(separation_parameters):
            if separation_param < 0.1:
                Cc[i] = (np.pi * solid_p[throat_fluid_conductivity][i] * net[mean_curvature][i] *
                         np.log(conductivity_ratios[i] ** 2))
            else:
                Cc[i] = np.pi * solid_p[throat_fluid_conductivity][i] * net[mean_curvature][i] * np.log(
                    1 + net[effective_mean_radius_curvature_fraction][i] ** 2 * net[mean_curvature][i] /
                    net[throat_length][i])

    else:
        beta = conductivity_ratios * net[relative_contact_radius]
        for i, overlap_param in enumerate(beta):
            if overlap_param < 1:
                Cc[i] = np.pi * solid_p[throat_fluid_conductivity][i] * net[mean_curvature][i] * (
                        0.17 * overlap_param ** 2 + np.log(conductivity_ratios[i] ** 2))
            else:
                Cc[i] = np.pi * solid_p[throat_fluid_conductivity][i] * net[mean_curvature][i] * (
                        2 * overlap_param / np.pi - 2 * np.log(overlap_param) + np.log(conductivity_ratios[i] ** 2))

    r1, r2 = (net['pore.diameter'][net.conns] / 2).T
    solid_conductivities = solid_p[pore_thermal_conductivity][net.conns]
    for i, solid_conductivity in enumerate(solid_conductivities):
        C1p[i] = np.pi * solid_conductivity[0] * (effective_mean_particle_radius[i] *
                                                  net[effective_mean_radius_curvature_fraction][i]) ** 2 / r1[i]
        C2p[i] = np.pi * solid_conductivity[1] * (effective_mean_particle_radius[i] *
                                                  net[effective_mean_radius_curvature_fraction][i]) ** 2 / r2[i]

    conductance = 1 / (1 / C1p + 1 / Cc + 1 / C2p)

    return conductance


def batchelor_old(solid_p,
                  pore_thermal_conductivity="pore.thermal_conductivity",
                  throat_solid_conductivity="throat.thermal_conductivity",
                  throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                  mean_curvature="throat.mean_curvature",
                  throat_length="throat.length",
                  mean_radius_curvature_fraction="throat.relative_mean_curvature"
                  ):
    r"""
    /// DEPRECIATED

    Calculates conductance of the bridge model based on Batchelor, O'Brien 1977 model, presented by Yun and Evans (2010).
    They suggest  the fraction of the mean radius of curvature should be 0.5 for dry and 0.8 for wet conditions.
    Value of 0.25 was used in the more recent article by Fei, Wenbin; Narsilio, Guillermo  10.1016/j.jrmge.2021.08.008.
    They however used it for intra particle conductivity.

    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    throat_thermal_conductivity : str
        %(dict_burb)s throat thermal conductivity
    mean_radius_curvature_fraction : str
        %(dict_burb)s mean radius of curvature fraction
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    throat_length : str
        %(dict_burb)s throat length

    %(fluid_p)s
    throat_thermal_conductivity : str
        %(dict_burb)s thermal conductivity

    Returns
    -------
    %(return_arr)s

    """

    net = solid_p.network

    if mean_radius_curvature_fraction not in net.keys():
        net[mean_radius_curvature_fraction] = 0.5
        warnings.warn("No 'throat.relative_mean_curvature' provided in solid phase. Using default value of 0.5")

    solid_conductivity = np.min(solid_p[pore_thermal_conductivity][net.conns], 1)

    conductivity_ratios = solid_conductivity / solid_p[throat_fluid_conductivity]

    separation_parameters = np.power(conductivity_ratios, 2) * net[throat_length] / net[mean_curvature]
    conductance = np.empty_like(separation_parameters)
    for i, sep_param in enumerate(separation_parameters):
        if sep_param < 0.1:
            conductance[i] = np.pi * solid_p[throat_solid_conductivity][i] * net[mean_curvature][
                i] * np.log(conductivity_ratios[i])
        else:
            print(i)
            conductance[i] = (np.pi * solid_p[throat_solid_conductivity][i] * net[mean_curvature][i]
                              * np.log(1 + net[mean_radius_curvature_fraction] ** 2 * net[mean_curvature][i] /
                                       net[throat_length][i]))
    return conductance


def kunii_smith(solid_p,
                pore_thermal_conductivity="pore.thermal_conductivity",
                throat_solid_conductivity="throat.thermal_solid_conductivity",
                throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                mean_curvature="throat.mean_curvature",
                diameter="pore.diameter",
                boundary_throats="throat.boundary"
                ):
    r"""
    Calculates conductance of the bridge based on the contact model propsed by Kunii and Smith 1960
    DOI: 10.1002/aic.690060115.
    The model combines 2 resistances: Resistance of a cylinder equivalent to a pore represnetd by a sphere.
    and a resistance of boundary fluid in the proximity of the contact point.

    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    throat_thermal_conductivity : str
        %(dict_burb)s throat thermal conductivity
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    diameter : str
        %(dict_burb)s diameter
    boundary_throats : str
        %(dict_burb)s boundary throats

    %(fluid_p)s
    throat_thermal_conductivity : str
        %(dict_burb)s thermal conductivity

    Returns
    -------
    %(return_arr)s

    """
    net = solid_p.network
    # local declaration of the general particle network
    particle_conductivities = solid_p[pore_thermal_conductivity][net.conns]
    # 2xN array with thermal conductivities of spheres bounding the throat, N is the number of throats

    n = np.mean(net.num_neighbors(pores=net.Ps, flatten=False)[net.conns], axis=1) / 2
    cos_theta = np.sqrt(1 - 1 / n)

    contact_area = np.pi * net[mean_curvature] ** 2 * 1 / n
    # Area influenced by the contacts
    # sin2(theta) substituted by 1/n.
    # Original paper presumes particles of same size, Batchelors mean curvature was thought to
    # better represent this parameter.

    kappa = solid_p[throat_solid_conductivity] / solid_p[throat_fluid_conductivity]
    # Original paper presumes same material, modified with the presumption, that lower conductivity influences the
    # system more.

    conductance = (np.pi * net[mean_curvature] * solid_p[throat_fluid_conductivity] * (kappa/(kappa-1))**2 *
                    np.log(kappa - (kappa - 1) * cos_theta) - (kappa - 1) / kappa * (1 - cos_theta))

    return conductance


def tsotsas_bob(solid_p,
            throat_solid_conductivity="throat.thermal_solid_conductivity",
            relative_contact_radius="throat.relative_contact_throat_radius",
            mean_curvature="throat.mean_curvature"):
    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    particle_cross_section = np.pi * np.square(net[mean_curvature])

    conductance = (net[relative_contact_radius]) * solid_p[throat_solid_conductivity] / (
            2 * np.pi * net[mean_curvature]) * particle_cross_section

    return conductance


def argento(solid_p,
            throat_solid_conductivity="throat.thermal_solid_conductivity",
            relative_contact_radius="throat.relative_contact_throat_radius",
            mean_curvature="throat.mean_curvature",
            throat_lenght="throat.length"):
    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    bridge_len = net[throat_lenght] + np.sum((net['pore.diameter'][net.conns] / 2), axis=1)

    particle_cross_section = np.pi * np.square(net[mean_curvature])
    resistance = 0.889 / (
            net[relative_contact_radius] * solid_p[throat_solid_conductivity] * particle_cross_section) * bridge_len

    conductance = 1 / resistance

    return conductance


def fei_narsilio(solid_p,
                 pore_thermal_conductivity="pore.thermal_conductivity",
                 throat_solid_conductivity="throat.thermal_solid_conductivity",
                 throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                 relative_contact_radius="throat.relative_contact_throat_radius",
                 relative_bridge_radius="throat.relative_bridge_radius",
                 mean_curvature="throat.mean_curvature",
                 boundary_throats="throat.boundary",
                 diameter="pore.diameter"):
    def particle_conductance(_solid_conductivity, _particle_volume, _distance_to_conatct, _shape_factor):
        _conductance = _solid_conductivity * _shape_factor * _particle_volume / _distance_to_conatct ** 2
        return _conductance

    def gap_conductance_integral(r, particle_radius, _fluid_conductivity):
        _conductance = _fluid_conductivity * 2 * np.pi * r / (particle_radius - np.sqrt(particle_radius ** 2 - r ** 2))
        return _conductance

    def contact_conductance(_contact_radius, _contact_length, _roughness_coef, _solid_conductivity):
        _contact_area = np.pi * _contact_radius ** 2
        _conductance = _solid_conductivity * _roughness_coef * _contact_area / _contact_length
        return _conductance

    net = solid_p.network

    if relative_contact_radius not in net.keys():
        solid_p[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    if relative_bridge_radius not in net.keys():
        solid_p[relative_bridge_radius] = 0.1
        warnings.warn(f"No {relative_bridge_radius} provided in solid phase. Using default value of 0.1")

    fluid_conductivity = solid_p[throat_fluid_conductivity]
    solid_conductivities = solid_p[pore_thermal_conductivity][net.conns]
    shape_factor = 1 / net.num_neighbors(pores=net.Ps, flatten=False)
    r1, r2 = (net[diameter][net.conns] / 2).T
    bridge_radius = net[mean_curvature] * net[relative_bridge_radius]
    contact_radius = net[mean_curvature] * net[relative_contact_radius]
    particle_volume1 = 4 / 3 * np.pi * r1 ** 3
    particle_volume2 = 4 / 3 * np.pi * r2 ** 3

    conductance = np.zeros_like(r1)
    for i, conns in enumerate(r1):
        Cc = contact_conductance(contact_radius, contact_radius, 0.9, solid_p[throat_solid_conductivity])
        # contact length is arbitrary other option: net[mean_curvature] / 5

        if not net[boundary_throats][i]:
            _integral1 = integrate.quad(lambda r: gap_conductance_integral(r, r1[i], fluid_conductivity[i]),
                                        contact_radius[i], bridge_radius[i])
            _integral2 = integrate.quad(lambda r: gap_conductance_integral(r, r2[i], fluid_conductivity[i]),
                                        contact_radius[i], bridge_radius[i])
            Cg = 1 / (1 / _integral1[0] + 1 / _integral2[0])

            poreid = net.conns[i]

            C1p = particle_conductance(solid_conductivities[i, 0], particle_volume1[i], r1[i], shape_factor[poreid[0]])
            C2p = particle_conductance(solid_conductivities[i, 1], particle_volume2[i], r2[i], shape_factor[poreid[1]])

        else:
            _integral = integrate.quad(lambda r: gap_conductance_integral(r, r1[i], fluid_conductivity[i]),
                                       contact_radius[i], bridge_radius[i])
            Cg = _integral[0]

            poreid = net.conns[i]

            C1p = particle_conductance(solid_conductivities[i, 0], particle_volume1[i], r1[i], poreid[0])
            C2p = particle_conductance(solid_conductivities[i, 1], particle_volume2[i], r2[i], 1)

        conductance[i] = 1 / (1 / C1p + 1 / (Cg + Cc[i]) + 1 / C2p)
    return conductance


def birkholz(solid_p,
             relative_contact_radius="throat.relative_contact_throat_radius",
             mean_curvature="throat.mean_curvature",
             pore_thermal_conductivity="pore.thermal_conductivity",
             ):
    net = solid_p.network

    if relative_contact_radius not in net.keys():
        solid_p[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    solid_conductivities = solid_p[pore_thermal_conductivity][net.conns].T
    return 4 * net[mean_curvature] * net[relative_contact_radius] / (
            1 / solid_conductivities[0] + 1 / solid_conductivities[1])


def zehner_bauer_schlunder(solid_p,
                           solid_volume="throat.solid_volume",
                           fluid_volume="throat.fluid_volume",
                           throat_solid_conductivity="throat.thermal_solid_conductivity",
                           throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                           relative_contact_radius="throat.relative_contact_throat_radius",
                           diameter="pore.diameter",
                           mean_curvature="throat.mean_curvature",
                           return_conductance=True
                           ):
    net = solid_p.network
    r1, r2 = (net[diameter][net.conns] / 2).T

    if solid_volume not in net.keys():
        net[solid_volume] = 2 / 3 * np.pi * r1 ** 3 + 2 / 3 * np.pi * r2 ** 3
        warnings.warn(f"No {solid_volume} volume provided in the network. Calculated from praticle diameter.")

    if fluid_volume not in net.keys():
        total_volume = (np.max((r1, r2)) * 2) ** 2 * (r1 + r2)
        net[fluid_volume] = total_volume - net[solid_volume]
        warnings.warn(
            f"No {fluid_volume} provided in the network. Calculated from total volume assuming unit cell of the larger particle diameter.")
    else:
        total_volume = net[solid_volume] + net[fluid_volume]

    if relative_contact_radius not in net.keys():
        solid_p[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    flattening_coef = net[relative_contact_radius] ** 2
    porosity = net[fluid_volume] / total_volume

    rel_solid_conductivity = solid_p[throat_solid_conductivity] / solid_p[throat_fluid_conductivity]

    b = 1.25 * ((1 - porosity) / porosity) ** (10 / 9)
    n = 1 - b / rel_solid_conductivity

    rel_unit_cell_cunductivity = 2 / n * (b / n ** 2 * (rel_solid_conductivity - 1) /
                                          rel_solid_conductivity * np.log(rel_solid_conductivity / b) - (b + 1) / 2 - (
                                                  b - 1) / n)
    rel_throat_conductivity = ((1 - np.sqrt(1 - porosity)) + np.sqrt(1 - porosity) *
                               (flattening_coef * rel_solid_conductivity + (
                                       1 - flattening_coef) * rel_unit_cell_cunductivity))

    if return_conductance:
        cross_section = (np.max((r1, r2)) * 2) ** 2
        length_throat = (r1 + r2)

        conductance = rel_throat_conductivity * solid_p[throat_fluid_conductivity] * cross_section / length_throat
        return conductance
    else:
        return rel_throat_conductivity * solid_p[throat_fluid_conductivity], porosity


def tsotsas_zbs(solid_p,
                solid_volume="throat.solid_volume",
                fluid_volume="throat.fluid_volume",
                throat_solid_conductivity="throat.thermal_solid_conductivity",
                throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                relative_contact_radius="throat.relative_contact_throat_radius",
                diameter="pore.diameter",
                mean_curvature="throat.mean_curvature"):

    net = solid_p.network
    r1, r2 = (net[diameter][net.conns] / 2).T

    if solid_volume not in net.keys():
        net[solid_volume] = 2 / 3 * np.pi * r1 ** 3 + 2 / 3 * np.pi * r2 ** 3
        warnings.warn(f"No {solid_volume} volume provided in the network. Calculated from praticle diameter.")

    if fluid_volume not in net.keys():
        total_volume = (np.max((r1, r2)) * 2) ** 2 * (r1 + r2)
        net[fluid_volume] = total_volume - net[solid_volume]
        warnings.warn(
            f"No {fluid_volume} provided in the network. Calculated from total volume assuming unit cell of the larger particle diameter.")


    if relative_contact_radius not in net.keys():
        solid_p[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    throat_conductivity, porosity = zehner_bauer_schlunder(solid_p,
                                                 solid_volume,
                                                 fluid_volume,
                                                 throat_solid_conductivity,
                                                 throat_fluid_conductivity,
                                                 relative_contact_radius,
                                                 diameter,
                                                 mean_curvature,
                                                 False)

    return net[solid_volume]/(4*(1-porosity)*net[mean_curvature]**2)*throat_conductivity


