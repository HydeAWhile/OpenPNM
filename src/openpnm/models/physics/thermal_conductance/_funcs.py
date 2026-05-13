from openpnm.models.physics._utils import _poisson_conductance
from openpnm.models import _doctxt
import warnings
import numpy as np

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
           "tsotsas_zbs",
           "bahrami_rough_joint",
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


def yovanovich(
    solid_p,
    throat_solid_conductivity="throat.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    relative_contact_radius="throat.relative_contact_throat_radius",
    relative_bridge_radius="throat.relative_bridge_radius",
    effective_radius="throat.effective_radius",
    temperature="throat.temperature",
    radiation_exchange_factor=0.0,
    include_fluid_conduction=False,
    include_radiation=False,
    eta_min=10.0,
):
    r"""
    Throat conductance model based on Yovanovich (1967) DOI: 10.2514/3.28821, adapted for OpenPNM.

    Implemented branches
    --------------------
    1. Solid conduction through the contacting body:
           R_solid = 1/(2*k_s*a) - ln(2)/(pi*k_s*R)

       where:
           a = e * R
           e = relative_contact_radius
           R = effective_radius

    2. Optional fluid conduction through the separation zone
       using the asymptotic continuum approximation:

           R_fluid_half = e^2 * R * (eta^2 - 5.1/eta) / (7.1 * k_f)

       where:
           eta = r_b / a = xi / e
           r_b = bridge radius
           xi  = relative_bridge_radius = r_b / R

       The total fluid path is modeled as two identical half-gaps in series:
           G_fluid = 1 / (2 * R_fluid_half)

    3. Optional radiation branch (linearized):
           G_rad = A_proj * F_ps * 4 * sigma_SB * T_avg^3

       with:
           A_proj = pi * R^2

    Total conductance
    -----------------
        G_total = G_solid + G_fluid + G_rad

    Notes
    -----
    - This is a throat-level adaptation of the Yovanovich sphere-plane model.
    - The solid branch follows the paper's constriction resistance approximation.
    - The fluid branch is tied to bridge radius rather than using a free eta parameter. This parameter is still highly
      influential and does not work well. More tweaking necessary to get good results.
    - The fluid branch here is the asymptotic continuum approximation for liquids
      and dense gases.
    - Rarefied-gas correction is not included in this version.
    - eta_min can be used to prevent unrealistically small eta values from making
      the fluid conductance too large. However the model seems to start working with value over 1000, which seems arbitrary.
    """

    net = solid_p.network
    Nt = net.Nt
    sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/m^2/K^4]

    # ---------------------------------------------------------------------
    # Geometry: contact radius and bridge radius
    # ---------------------------------------------------------------------
    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(
            f"No {relative_contact_radius} provided in network. "
            f"Using default value of 0.009"
        )

    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn(
            f"No {relative_bridge_radius} provided in network. "
            f"Using default value of 0.1"
        )

    e = np.asarray(net[relative_contact_radius], dtype=float)   # e = a / R
    xi = np.asarray(net[relative_bridge_radius], dtype=float)   # xi = r_b / R
    R = np.asarray(net[effective_radius], dtype=float)

    contact_radius = e * R
    bridge_radius = xi * R

    # eta = r_b / a = xi / e
    eta = np.zeros(Nt, dtype=float)
    mask_eta = e > 0.0
    eta[mask_eta] = xi[mask_eta] / e[mask_eta]

    # Optional safeguard against too-small eta values
    if eta_min is not None:
        eta = np.maximum(eta, float(eta_min))

    # ---------------------------------------------------------------------
    # 1) Solid conduction branch
    # ---------------------------------------------------------------------
    k_s = np.asarray(solid_p[throat_solid_conductivity], dtype=float)
    G_solid = np.zeros(Nt, dtype=float)

    valid_s = (k_s > 0.0) & (R > 0.0) & (contact_radius > 0.0)
    if np.any(valid_s):
        R_solid = (
            1.0 / (2.0 * k_s[valid_s] * contact_radius[valid_s])
            - np.log(2.0) / (np.pi * k_s[valid_s] * R[valid_s])
        )

        good_s = R_solid > 0.0
        G_solid_valid = np.zeros_like(R_solid)
        G_solid_valid[good_s] = 1.0 / R_solid[good_s]
        G_solid[valid_s] = G_solid_valid

    # ---------------------------------------------------------------------
    # 2) Fluid conduction branch (optional, asymptotic closed form)
    # ---------------------------------------------------------------------
    G_fluid = np.zeros(Nt, dtype=float)

    if include_fluid_conduction:
        if throat_fluid_conductivity in solid_p.keys():
            k_f = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)

            valid_f = (k_f > 0.0) & (R > 0.0) & (e > 0.0) & (eta > 1.0)
            if np.any(valid_f):
                shape_term = eta[valid_f]**2 - 5.1 / eta[valid_f]
                good_shape = shape_term > 0.0

                R_fluid_half = np.zeros_like(shape_term)
                R_fluid_half[good_shape] = (
                    e[valid_f][good_shape]**2
                    * R[valid_f][good_shape]
                    * shape_term[good_shape]
                    / (7.1 * k_f[valid_f][good_shape])
                )

                good_half = R_fluid_half > 0.0
                G_fluid_valid = np.zeros_like(R_fluid_half)
                G_fluid_valid[good_half] = 1.0 / (2.0 * R_fluid_half[good_half])

                G_fluid[valid_f] = G_fluid_valid
        else:
            warnings.warn(
                f"include_fluid_conduction=True but '{throat_fluid_conductivity}' "
                f"was not found in phase. Fluid contribution set to zero."
            )

    # ---------------------------------------------------------------------
    # 3) Radiation branch (optional)
    # ---------------------------------------------------------------------
    G_rad = np.zeros(Nt, dtype=float)

    if include_radiation:
        # Radiation exchange factor F_ps can be scalar, network key, or phase key
        if isinstance(radiation_exchange_factor, str):
            if radiation_exchange_factor in net.keys():
                F_ps = np.asarray(net[radiation_exchange_factor], dtype=float)
            elif radiation_exchange_factor in solid_p.keys():
                F_ps = np.asarray(solid_p[radiation_exchange_factor], dtype=float)
            else:
                raise KeyError(
                    f"Radiation exchange factor key '{radiation_exchange_factor}' "
                    f"not found in network or phase."
                )
        else:
            F_ps = np.full(Nt, float(radiation_exchange_factor), dtype=float)

        # Temperature handling:
        # 1) explicit key in phase
        # 2) explicit key in network
        # 3) average pore.temperature onto throats
        # 4) scalar
        if isinstance(temperature, str):
            if temperature in solid_p.keys():
                T_avg = np.asarray(solid_p[temperature], dtype=float)
            elif temperature in net.keys():
                T_avg = np.asarray(net[temperature], dtype=float)
            elif (temperature == "throat.temperature") and ("pore.temperature" in solid_p.keys()):
                T_avg = np.mean(
                    np.asarray(solid_p["pore.temperature"], dtype=float)[net.conns],
                    axis=1,
                )
            else:
                raise KeyError(
                    f"Temperature key '{temperature}' not found in network or phase."
                )
        else:
            T_avg = np.full(Nt, float(temperature), dtype=float)

        A_proj = np.pi * R**2
        valid_r = (A_proj > 0.0) & (F_ps > 0.0) & (T_avg > 0.0)

        if np.any(valid_r):
            G_rad[valid_r] = (
                A_proj[valid_r]
                * F_ps[valid_r]
                * 4.0
                * sigma_SB
                * T_avg[valid_r]**3
            )

    # ---------------------------------------------------------------------
    # Total conductance: parallel branches
    # ---------------------------------------------------------------------
    G_total = G_solid + G_fluid + G_rad
    return G_total



def dixon_bridge_model(
    solid_p,
    throat_solid_conductivity="throat.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    relative_bridge_radius="throat.relative_bridge_radius",
    diameter="pore.diameter",
    throat_length="throat.length",
):
    r"""
    Calculate throat conductance using the Dixon et al. (2013) bridge model DOI: 10.1016/j.compchemeng.2012.08.011.

    Notes
    -----
    This implementation uses the analytical closed-form version of the bridge
    effective conductivity corresponding to Eq. (10) in Dixon et al. when the
    reduced fluid conductivity k_fr is treated as constant (i.e. the
    Smoluchowski correction of Eq. (11) is neglected).

    The resulting effective bridge conductivity is then converted into a throat
    conductance by:
        G = k_eff * A / L_total

    where:
        A       = pi * r_b^2
        L_total = 2*h_b + throat_length
        h_b     = R - sqrt(R^2 - r_b^2)

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].
    """
    net = solid_p.network

    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn(
            f"No {relative_bridge_radius} provided in network. "
            f"Using default value of 0.1"
        )

    # Particle radius: smaller of the two connected pores
    r_particle = np.min(np.asarray(net[diameter], dtype=float)[net.conns], axis=1) / 2.0

    # Bridge radius
    r_bridge = r_particle * np.asarray(net[relative_bridge_radius], dtype=float)
    r_bridge = np.clip(r_bridge, 0.0, r_particle)

    # Bridge half-thickness h_b
    sqrt_term = np.sqrt(np.maximum(r_particle**2 - r_bridge**2, 0.0))
    h_b = r_particle - sqrt_term

    # Conductivities
    k_s = np.asarray(solid_p[throat_solid_conductivity], dtype=float)
    k_f = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)

    # Initialize output
    conductance = np.zeros_like(r_bridge, dtype=float)

    # Only physically meaningful entries
    valid = (
        (r_particle > 0.0)
        & (r_bridge > 0.0)
        & (h_b > 0.0)
        & (k_s > 0.0)
        & (k_f > 0.0)
    )

    if not np.any(valid):
        return conductance

    rb = r_bridge[valid]
    R = r_particle[valid]
    L = h_b[valid]
    ks = k_s[valid]
    kf = k_f[valid]

    # Analytical integral from 0 to r_b
    integral_value = np.empty_like(rb, dtype=float)

    equal_k = np.isclose(kf, ks, rtol=1e-12, atol=0.0)
    diff_k = ~equal_k

    if np.any(diff_k):
        B = kf[diff_k] - ks[diff_k]
        A = kf[diff_k] * L[diff_k] + (ks[diff_k] - kf[diff_k]) * R[diff_k]

        integral_value[diff_k] = (
            L[diff_k] / B
            + (A / B**2) * np.log(ks[diff_k] / kf[diff_k])
        )

    if np.any(equal_k):
        # Limiting expression for k_f == k_s
        integral_value[equal_k] = rb[equal_k]**2 / (2.0 * kf[equal_k] * L[equal_k])

    # Effective bridge conductivity (constant-k_f version of Dixon Eq. 10)
    k_eff = 2.0 * L / (rb**2) * integral_value * (ks * kf)

    # Full bridge length used as OpenPNM throat adaptation
    L_total = 2.0 * L + np.asarray(net[throat_length], dtype=float)[valid]
    A_bridge = np.pi * rb**2

    good = (k_eff > 0.0) & (L_total > 0.0)
    G_valid = np.zeros_like(k_eff)
    G_valid[good] = k_eff[good] * A_bridge[good] / L_total[good]

    conductance[valid] = G_valid
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

    Notes
    -----
    This implementation extends the Dixon et al. (2013) bridge model to the case
    where the two connected particles may have different sizes and different
    solid conductivities.

    Each particle-side bridge conductance is computed from the analytical
    closed-form version of the Dixon bridge conductivity corresponding to Eq. (10)
    when the reduced fluid conductivity k_fr is treated as constant, i.e. the
    Smoluchowski correction of Eq. (11) is neglected.

    The total throat conductance is modeled as three resistances in series:
        1. bridge-side conductance on pore 1 side,
        2. cylindrical fluid gap conductance through the throat,
        3. bridge-side conductance on pore 2 side.

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].
    """

    def _single_sphere_bridge_conductance(r_particle, r_bridge, k_s, k_f):
        """
        Analytical conductance of one sphere-side bridge segment using the
        constant-k_f closed-form Dixon bridge model.

        Parameters
        ----------
        r_particle : ndarray
            Particle radius on one side of each throat [m].
        r_bridge : ndarray
            Common bridge radius for each throat [m].
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

        # Ensure physically valid local bridge radius
        r_bridge_local = np.clip(r_bridge, 0.0, r_particle)

        # Half-bridge thickness h_b for this sphere side
        sqrt_term = np.sqrt(np.maximum(r_particle**2 - r_bridge_local**2, 0.0))
        h_b = r_particle - sqrt_term

        conductance = np.zeros_like(r_bridge_local, dtype=float)

        # Only positive, physically meaningful cases can conduct
        valid = (
            (r_particle > 0.0)
            & (r_bridge_local > 0.0)
            & (h_b > 0.0)
            & (k_s > 0.0)
            & (k_f > 0.0)
        )

        if not np.any(valid):
            return conductance

        rb = r_bridge_local[valid]
        R = r_particle[valid]
        L = h_b[valid]
        ks = k_s[valid]
        kf = k_f[valid]

        # Analytical integral from 0 to r_b
        integral_value = np.empty_like(rb, dtype=float)

        equal_k = np.isclose(kf, ks, rtol=1e-12, atol=0.0)
        diff_k = ~equal_k

        # Closed form for k_f != k_s
        if np.any(diff_k):
            B = kf[diff_k] - ks[diff_k]
            A = kf[diff_k] * L[diff_k] + (ks[diff_k] - kf[diff_k]) * R[diff_k]

            integral_value[diff_k] = (
                L[diff_k] / B
                + (A / B**2) * np.log(ks[diff_k] / kf[diff_k])
            )

        # Limiting expression for k_f == k_s
        if np.any(equal_k):
            integral_value[equal_k] = (
                rb[equal_k]**2 / (2.0 * kf[equal_k] * L[equal_k])
            )

        # Effective conductivity of one bridge-side segment
        k_eff_side = 2.0 * L / (rb**2) * integral_value * (ks * kf)

        # Convert side effective conductivity to side conductance:
        # G_side = k_eff_side * A / h_b, with A = pi * r_b^2
        A_bridge = np.pi * rb**2
        good = (k_eff_side > 0.0) & (L > 0.0)

        G_valid = np.zeros_like(k_eff_side)
        G_valid[good] = k_eff_side[good] * A_bridge[good] / L[good]

        conductance[valid] = G_valid
        return conductance

    net = solid_p.network

    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn(
            f"No {relative_bridge_radius} provided in network. "
            f"Using default value of 0.1"
        )

    conns = net.conns

    # Particle radius on each connected side
    r1 = np.asarray(net[diameter][conns[:, 0]], dtype=float) / 2.0
    r2 = np.asarray(net[diameter][conns[:, 1]], dtype=float) / 2.0

    # Common bridge radius based on the smaller connected particle
    r_min = np.minimum(r1, r2)
    r_bridge = np.asarray(
        r_min * np.asarray(net[relative_bridge_radius], dtype=float),
        dtype=float,
    )
    r_bridge = np.clip(r_bridge, 0.0, r_min)

    # Solid conductivity on each pore side
    ks1 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 0]], dtype=float)
    ks2 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 1]], dtype=float)

    # Throat fluid conductivity and throat gap length
    kf = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)
    Lt = np.asarray(net[throat_length], dtype=float)

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

    # Fluid gap conductance through a cylindrical throat region
    area = np.pi * r_bridge**2
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
    valid_total = total_resistance > 0.0
    conductance[valid_total] = 1.0 / total_resistance[valid_total]

    return np.maximum(conductance, 0.0)


def batchelor(
    solid_p,
    pore_thermal_conductivity="pore.thermal_conductivity",
    throat_solid_conductivity="throat.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    effective_radius="throat.effective_radius",
    relative_contact_radius="throat.relative_contact_throat_radius",
    throat_length="throat.length",
    effective_radius_fraction="throat.relative_effective_radius",
):
    r"""
    Calculates conductance of the bridge/contact model based on the
    Batchelor–O'Brien formulation as used in later particulate-network
    heat-transfer models.

    Notes
    -----
    This implementation is an OpenPNM throat-level adaptation in which the total
    conductance is modeled as three resistances in series:

        1. particle-side conductance through pore 1,
        2. contact / near-contact region conductance,
        3. particle-side conductance through pore 2.

    The contact-region term is evaluated piecewise for contact / no-contact
    conditions using the formulas in the current library implementation, while
    the particle-side terms are computed from an effective radius scale.

    Returns
    -------
    ndarray
        Conductance from sphere to sphere [W/K].
    """
    net = solid_p.network
    conns = net.conns
    Nt = net.Nt

    if effective_radius_fraction not in net.keys():
        net[effective_radius_fraction] = 0.5
        warnings.warn(
            f"No {effective_radius_fraction} provided in network. "
            f"Using default value of 0.5"
        )

    # ---------------------------------------------------------------------
    # Input arrays
    # ---------------------------------------------------------------------
    k_s_throat = np.asarray(solid_p[throat_solid_conductivity], dtype=float)
    k_f = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)

    Rm = np.asarray(net[effective_radius], dtype=float)
    frac = np.asarray(net[effective_radius_fraction], dtype=float)
    Lt = np.asarray(net[throat_length], dtype=float)

    # Connected pore radii and pore solid conductivities
    r1 = np.asarray(net["pore.diameter"][conns[:, 0]], dtype=float) / 2.0
    r2 = np.asarray(net["pore.diameter"][conns[:, 1]], dtype=float) / 2.0

    ks1 = np.asarray(solid_p[pore_thermal_conductivity][conns[:, 0]], dtype=float)
    ks2 = np.asarray(solid_p[pore_thermal_conductivity][conns[:, 1]], dtype=float)

    # Effective particle radius scale used for particle-side conductance
    effective_mean_particle_radius = Rm * frac

    # ---------------------------------------------------------------------
    # Contact / no-contact logic
    # ---------------------------------------------------------------------
    if relative_contact_radius not in net.keys():
        rel_contact = np.zeros(Nt, dtype=float)
        mask_no_contact = np.ones(Nt, dtype=bool)
        mask_contact = np.zeros(Nt, dtype=bool)
    else:
        rel_contact = np.asarray(net[relative_contact_radius], dtype=float)
        mask_no_contact = rel_contact <= 0.0
        mask_contact = rel_contact > 0.0

    # ---------------------------------------------------------------------
    # Contact-region conductance Cc
    # ---------------------------------------------------------------------
    Cc = np.zeros(Nt, dtype=float)

    # Core validity for contact-region formulas
    valid_core = (
        (k_s_throat > 0.0)
        & (k_f > 0.0)
        & (Rm > 0.0)
        & (frac > 0.0)
    )

    conductivity_ratios = np.zeros(Nt, dtype=float)
    mask_ratio = valid_core & (k_f > 0.0)
    conductivity_ratios[mask_ratio] = k_s_throat[mask_ratio] / k_f[mask_ratio]

    # -------------------------
    # No-contact branch
    # -------------------------
    # separation_parameter = alpha^2 * Lt / Rm
    separation_parameters = np.full(Nt, np.nan, dtype=float)
    mask_sep = valid_core & (Lt > 0.0)
    separation_parameters[mask_sep] = (
        conductivity_ratios[mask_sep] ** 2 * Lt[mask_sep] / Rm[mask_sep]
    )

    mask_no_contact_small = (
        mask_no_contact
        & valid_core
        & (conductivity_ratios > 1.0)   # ensures log(alpha^2) > 0
        & np.isfinite(separation_parameters)
        & (separation_parameters < 0.1)
    )

    if np.any(mask_no_contact_small):
        tmp = (
            np.pi
            * k_f[mask_no_contact_small]
            * Rm[mask_no_contact_small]
            * np.log(conductivity_ratios[mask_no_contact_small] ** 2)
        )
        Cc[mask_no_contact_small] = np.maximum(tmp, 0.0)

    mask_no_contact_large = (
        mask_no_contact
        & valid_core
        & (Lt > 0.0)
        & ~mask_no_contact_small
    )

    if np.any(mask_no_contact_large):
        arg = (
            1.0
            + frac[mask_no_contact_large] ** 2
            * Rm[mask_no_contact_large]
            / Lt[mask_no_contact_large]
        )
        good = arg > 1.0
        tmp = np.zeros(np.count_nonzero(mask_no_contact_large), dtype=float)
        tmp[good] = (
            np.pi
            * k_f[mask_no_contact_large][good]
            * Rm[mask_no_contact_large][good]
            * np.log(arg[good])
        )
        Cc[mask_no_contact_large] = np.maximum(tmp, 0.0)

    # -------------------------
    # Contact branch
    # -------------------------
    beta = conductivity_ratios * rel_contact

    mask_contact_small = (
        mask_contact
        & valid_core
        & (beta > 0.0)
        & (beta < 1.0)
        & (conductivity_ratios > 1.0)
    )

    if np.any(mask_contact_small):
        tmp = (
            np.pi
            * k_f[mask_contact_small]
            * Rm[mask_contact_small]
            * (
                0.17 * beta[mask_contact_small] ** 2
                + np.log(conductivity_ratios[mask_contact_small] ** 2)
            )
        )
        Cc[mask_contact_small] = np.maximum(tmp, 0.0)

    mask_contact_large = (
        mask_contact
        & valid_core
        & (beta >= 1.0)
    )

    if np.any(mask_contact_large):
        tmp = (
            np.pi
            * k_f[mask_contact_large]
            * Rm[mask_contact_large]
            * (
                2.0 * beta[mask_contact_large] / np.pi
                - 2.0 * np.log(beta[mask_contact_large])
                + np.log(conductivity_ratios[mask_contact_large] ** 2)
            )
        )
        Cc[mask_contact_large] = np.maximum(tmp, 0.0)

    # ---------------------------------------------------------------------
    # Particle-side conductances
    # ---------------------------------------------------------------------
    C1p = np.zeros(Nt, dtype=float)
    C2p = np.zeros(Nt, dtype=float)

    mask_p1 = (ks1 > 0.0) & (r1 > 0.0) & (effective_mean_particle_radius > 0.0)
    if np.any(mask_p1):
        C1p[mask_p1] = (
            np.pi
            * ks1[mask_p1]
            * effective_mean_particle_radius[mask_p1] ** 2
            / r1[mask_p1]
        )

    mask_p2 = (ks2 > 0.0) & (r2 > 0.0) & (effective_mean_particle_radius > 0.0)
    if np.any(mask_p2):
        C2p[mask_p2] = (
            np.pi
            * ks2[mask_p2]
            * effective_mean_particle_radius[mask_p2] ** 2
            / r2[mask_p2]
        )

    # ---------------------------------------------------------------------
    # Series combination
    # 1 / G_total = 1 / C1p + 1 / Cc + 1 / C2p
    # All three terms must be present for a valid series path.
    # ---------------------------------------------------------------------
    conductance = np.zeros(Nt, dtype=float)

    valid_series = (C1p > 0.0) & (Cc > 0.0) & (C2p > 0.0)
    if np.any(valid_series):
        total_resistance = (
            1.0 / C1p[valid_series]
            + 1.0 / Cc[valid_series]
            + 1.0 / C2p[valid_series]
        )
        conductance[valid_series] = 1.0 / total_resistance

    return np.maximum(conductance, 0.0)



def kunii_smith(
    solid_p,
    throat_solid_conductivity="throat.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    effective_radius="throat.effective_radius",
):
    r"""
    Calculates conductance of the bridge based on the Kunii and Smith (1960)
    contact model, adapted here as a throat-level conductance model for OpenPNM.

    Notes
    -----
    This implementation keeps the existing OpenPNM interpretation in which:
      - the contact-count quantity is approximated from local network coordination
      - the particle radius appearing in the paper is represented by
        ``throat.effective_radius``

    The original paper presents Eq. (11) as the heat flow through one contact
    region between particles, and then uses it in a packed-bed-scale derivation.
    Here that expression is used directly as a throat conductance model.

    Returns
    -------
    ndarray
        Conductance from sphere to sphere [W/kappa].
    """
    net = solid_p.network

    # Approximate contact-count surrogate from local coordination
    n = np.mean(net.num_neighbors(pores=net.Ps, flatten=False)[net.conns], axis=1) / 2.0

    # Guard against invalid/degenerate coordination values
    n = np.asarray(n, dtype=float)
    n = np.maximum(n, 1.0 + 1e-12)

    # In the paper, sin^2(theta_0) = 1 / n  ->  cos(theta_0) = sqrt(1 - 1/n)
    cos_theta = np.sqrt(np.maximum(1.0 - 1.0 / n, 0.0))

    # Conductivity ratio kappa = k_s / k_f
    k_s = np.asarray(solid_p[throat_solid_conductivity], dtype=float)
    k_f = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)
    R_eff = np.asarray(net[effective_radius], dtype=float)

    conductance = np.zeros_like(k_f, dtype=float)

    # Valid positive entries only
    valid = (k_s > 0.0) & (k_f > 0.0) & (R_eff > 0.0)
    if not np.any(valid):
        return conductance

    kappa = np.zeros_like(k_f, dtype=float)
    kappa[valid] = k_s[valid] / k_f[valid]

    # Handle kappa ~= 1 with the analytical limit:
    # lim_{kappa->1} (kappa/(kappa-1))^2 * [ ln(kappa - (kappa-1)cos(theta))
    #                           - ((kappa-1)/kappa)(1-cos(theta)) ]
    # = (1 - cos(theta)^2) / 2
    equal_mask = valid & np.isclose(kappa, 1.0, rtol=1e-8, atol=1e-12)
    if np.any(equal_mask):
        bracket = 0.5 * (1.0 - cos_theta[equal_mask] ** 2)
        conductance[equal_mask] = np.pi * R_eff[equal_mask] * k_f[equal_mask] * bracket

    # General case
    diff_mask = valid & ~np.isclose(kappa, 1.0, rtol=1e-8, atol=1e-12)
    if np.any(diff_mask):
        Kd = kappa[diff_mask]
        c = cos_theta[diff_mask]

        bracket = (Kd / (Kd - 1.0)) ** 2 * (
            np.log(Kd - (Kd - 1.0) * c)
            - ((Kd - 1.0) / Kd) * (1.0 - c)
        )

        conductance[diff_mask] = (
            np.pi * R_eff[diff_mask] * k_f[diff_mask] * bracket
        )

    # Small negative values can appear from floating-point noise
    return np.maximum(conductance, 0.0)



def tsotsas_bob(solid_p,
                throat_solid_conductivity="throat.thermal_solid_conductivity",
                relative_contact_radius="throat.relative_contact_throat_radius",
                effective_radius="throat.effective_radius"):
    r"""
    Calculate thermal conductance using the Tsotsas-Bob simplified contact model.

    This model estimates the thermal conductance through a particle contact based
    on the throat solid thermal conductivity, the effective particle radius, and
    the relative contact radius. The conductance is evaluated from the contact
    cross-sectional area and a characteristic conduction length proportional to
    the effective radius.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing throat-scale thermal conductivity data and a
        reference to the associated network.
    throat_solid_conductivity : str, optional
        Dictionary key of the throat solid thermal conductivity values [W/m.K].
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-]. This quantity scales
        the contact conductance and represents the characteristic contact size
        relative to the particle size.
    effective_radius : str, optional
        Dictionary key of the effective particle radius [m].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    If ``relative_contact_radius`` is not present in the network, a default
    value of ``0.009`` is assigned and a warning is issued.
    """

    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    particle_cross_section = np.pi * np.square(net[effective_radius])

    conductance = (net[relative_contact_radius]) * solid_p[throat_solid_conductivity] / (
            2 * np.pi * net[effective_radius]) * particle_cross_section

    return conductance


def argento(solid_p,
            throat_solid_conductivity="throat.thermal_solid_conductivity",
            relative_contact_radius="throat.relative_contact_throat_radius",
            effective_radius="throat.effective_radius",
            throat_lenght="throat.length"):
    r"""
    Calculate thermal conductance using the Argento contact-bridge model.

    This model estimates the thermal conductance through a particle bridge based
    on the throat solid thermal conductivity, the effective particle radius, the
    relative contact radius, and the total bridge length. The bridge length is
    defined as the throat length plus one half of each connected particle
    diameter.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing throat-scale thermal conductivity data and a
        reference to the associated network.
    throat_solid_conductivity : str, optional
        Dictionary key of the throat solid thermal conductivity values [W/m.K].
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-]. This quantity scales
        the conductive bridge area.
    effective_radius : str, optional
        Dictionary key of the effective particle radius [m].
    throat_lenght : str, optional
        Dictionary key of the throat length [m].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    If ``relative_contact_radius`` is not present in the network, a default
    value of ``0.009`` is assigned and a warning is issued.
    """

    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    bridge_len = net[throat_lenght] + np.sum((net['pore.diameter'][net.conns] / 2), axis=1)

    particle_cross_section = np.pi * np.square(net[effective_radius])
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
                 effective_radius="throat.effective_radius",
                 boundary_throats="throat.boundary",
                 diameter="pore.diameter"):
    r"""
    Calculate thermal conductance using the Fei-Narsilio bridge-contact model.

    This model combines three conductive contributions for each throat:

    1. particle-side conductance through pore 1,
    2. bridge/contact-region conductance,
    3. particle-side conductance through pore 2.

    The bridge/contact-region contribution is composed of:

    - a solid contact conductance through the contact area, and
    - a fluid-gap conductance through the bridge region.

    For internal throats, the fluid-gap conductance is evaluated on both particle
    sides and combined in series. For boundary throats, only one particle-side gap
    contribution is used and the second particle conductance is evaluated with unit
    shape factor.

    This implementation uses a vectorized analytical expression for the gap
    conductance integral and avoids per-throat numerical quadrature.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing pore-scale and throat-scale thermal conductivity
        data, together with a reference to the associated network.
    pore_thermal_conductivity : str, optional
        Dictionary key of the pore thermal conductivity values [W/m.K].
    throat_solid_conductivity : str, optional
        Dictionary key of the throat solid thermal conductivity values [W/m.K].
    throat_fluid_conductivity : str, optional
        Dictionary key of the throat fluid thermal conductivity values [W/m.K].
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-].
    relative_bridge_radius : str, optional
        Dictionary key of the relative bridge radius [-].
    effective_radius : str, optional
        Dictionary key of the effective throat radius [m].
    boundary_throats : str, optional
        Dictionary key of the boolean mask identifying boundary throats.
    diameter : str, optional
        Dictionary key of the pore diameter values [m].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    If ``relative_contact_radius`` is not available, a default value of ``0.009``
    is used. If ``relative_bridge_radius`` is not available, a default value of
    ``0.1`` is used.
    """

    def particle_conductance(_solid_conductivity, _particle_volume, _distance_to_contact, _shape_factor):
        return _solid_conductivity * _shape_factor * _particle_volume / _distance_to_contact**2

    def gap_conductance_closed_form(lower_radius, upper_radius, particle_radius, _fluid_conductivity):
        """
        Analytical value of:
            ∫ 2*pi*k_f*r / (R - sqrt(R^2 - r^2)) dr
        from lower_radius to upper_radius.
        """
        lower_radius = np.asarray(lower_radius, dtype=float)
        upper_radius = np.asarray(upper_radius, dtype=float)
        particle_radius = np.asarray(particle_radius, dtype=float)
        _fluid_conductivity = np.asarray(_fluid_conductivity, dtype=float)

        value = np.zeros_like(particle_radius, dtype=float)

        # Valid physical interval for the integral
        valid = (
            (_fluid_conductivity > 0.0)
            & (particle_radius > 0.0)
            & (upper_radius > lower_radius)
            & (lower_radius > 0.0)
            & (upper_radius <= particle_radius)
        )

        if not np.any(valid):
            return value

        R = particle_radius[valid]
        a = lower_radius[valid]
        b = upper_radius[valid]
        kf = _fluid_conductivity[valid]

        ua = np.sqrt(np.maximum(R**2 - a**2, 0.0))
        ub = np.sqrt(np.maximum(R**2 - b**2, 0.0))

        value[valid] = (
            2.0
            * np.pi
            * kf
            * (
                ub
                - ua
                + R * np.log((R - ub) / (R - ua))
            )
        )

        return value

    def contact_conductance(_contact_radius, _contact_length, _roughness_coef, _solid_conductivity):
        _contact_area = np.pi * _contact_radius**2
        return _solid_conductivity * _roughness_coef * _contact_area / _contact_length

    net = solid_p.network
    Nt = net.Nt

    # Relative radii with safe defaults
    if relative_contact_radius in net.keys():
        rel_contact = np.asarray(net[relative_contact_radius], dtype=float)
    else:
        rel_contact = np.full(Nt, 0.009, dtype=float)
        warnings.warn(
            f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009"
        )

    if relative_bridge_radius in net.keys():
        rel_bridge = np.asarray(net[relative_bridge_radius], dtype=float)
    else:
        rel_bridge = np.full(Nt, 0.1, dtype=float)
        warnings.warn(
            f"No {relative_bridge_radius} provided in solid phase. Using default value of 0.1"
        )

    fluid_conductivity = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)
    solid_conductivities = np.asarray(solid_p[pore_thermal_conductivity], dtype=float)[net.conns]

    shape_factor = 1.0 / net.num_neighbors(pores=net.Ps, flatten=False)
    pore1 = net.conns[:, 0]
    pore2 = net.conns[:, 1]

    sf1 = shape_factor[pore1]
    sf2 = shape_factor[pore2]

    r1, r2 = (np.asarray(net[diameter], dtype=float)[net.conns] / 2.0).T
    bridge_radius = np.asarray(net[effective_radius], dtype=float) * rel_bridge
    contact_radius = np.asarray(net[effective_radius], dtype=float) * rel_contact

    particle_volume1 = (4.0 / 3.0) * np.pi * r1**3
    particle_volume2 = (4.0 / 3.0) * np.pi * r2**3

    # Solid contact conductance
    # Preserves original behavior: contact_length = contact_radius, roughness_coef = 0.9
    Cc = contact_conductance(
        _contact_radius=contact_radius,
        _contact_length=contact_radius,
        _roughness_coef=0.9,
        _solid_conductivity=np.asarray(solid_p[throat_solid_conductivity], dtype=float),
    )

    # Fluid-gap conductance
    Cg1 = gap_conductance_closed_form(
        lower_radius=contact_radius,
        upper_radius=bridge_radius,
        particle_radius=r1,
        _fluid_conductivity=fluid_conductivity,
    )

    Cg2 = gap_conductance_closed_form(
        lower_radius=contact_radius,
        upper_radius=bridge_radius,
        particle_radius=r2,
        _fluid_conductivity=fluid_conductivity,
    )

    boundary_mask = np.asarray(net[boundary_throats], dtype=bool)
    internal_mask = ~boundary_mask

    Cg = np.zeros_like(r1, dtype=float)

    # Internal throats: two gap sides in series
    valid_internal = internal_mask & (Cg1 > 0.0) & (Cg2 > 0.0)
    Cg[valid_internal] = 1.0 / (
        1.0 / Cg1[valid_internal] + 1.0 / Cg2[valid_internal]
    )

    # Boundary throats: only the first side gap is used
    valid_boundary = boundary_mask & (Cg1 > 0.0)
    Cg[valid_boundary] = Cg1[valid_boundary]

    # Particle conductances
    C1p = particle_conductance(
        _solid_conductivity=solid_conductivities[:, 0],
        _particle_volume=particle_volume1,
        _distance_to_contact=r1,
        _shape_factor=sf1,
    )

    sf2_eff = sf2.copy()
    sf2_eff[boundary_mask] = 1.0

    C2p = particle_conductance(
        _solid_conductivity=solid_conductivities[:, 1],
        _particle_volume=particle_volume2,
        _distance_to_contact=r2,
        _shape_factor=sf2_eff,
    )

    # Bridge/contact path = fluid gap + solid contact in parallel
    bridge_path = Cg + Cc

    # Final series combination:
    # 1 / G = 1 / C1p + 1 / (Cg + Cc) + 1 / C2p
    R_total = np.zeros_like(r1, dtype=float)

    mask1 = C1p > 0.0
    maskb = bridge_path > 0.0
    mask2 = C2p > 0.0

    R_total[mask1] += 1.0 / C1p[mask1]
    R_total[maskb] += 1.0 / bridge_path[maskb]
    R_total[mask2] += 1.0 / C2p[mask2]

    conductance = np.zeros_like(r1, dtype=float)
    valid_total = R_total > 0.0
    conductance[valid_total] = 1.0 / R_total[valid_total]

    return conductance


def birkholz(solid_p,
             relative_contact_radius="throat.relative_contact_throat_radius",
             effective_radius="throat.effective_radius",
             pore_thermal_conductivity="pore.thermal_conductivity",
             ):

    r"""
    Calculate thermal conductance using the Birkholz contact model.

    This model estimates the thermal conductance between two connected particles
    from the effective radius, the relative contact radius, and the thermal
    conductivities of the two connected pores. The two pore conductivities are
    combined through the harmonic mean form appearing in the denominator.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing pore-scale thermal conductivity data and a
        reference to the associated network.
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-]. This quantity scales
        the effective conductive contact size.
    effective_radius : str, optional
        Dictionary key of the effective radius [m].
    pore_thermal_conductivity : str, optional
        Dictionary key of the pore thermal conductivity values [W/m.K].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    If ``relative_contact_radius`` is not present in the network, a default
    value of ``0.009`` is assigned and a warning is issued.
    """

    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    solid_conductivities = solid_p[pore_thermal_conductivity][net.conns].T
    return 4 * net[effective_radius] * net[relative_contact_radius] / (
            1 / solid_conductivities[0] + 1 / solid_conductivities[1])


def zehner_bauer_schlunder(solid_p,
                           solid_volume="throat.solid_volume",
                           fluid_volume="throat.fluid_volume",
                           throat_solid_conductivity="throat.thermal_solid_conductivity",
                           throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                           relative_contact_radius="throat.relative_contact_throat_radius",
                           diameter="pore.diameter",
                           return_conductance=True
                           ):

    r"""
    Calculate effective throat conductivity or conductance using the
    Zehner-Bauer-Schlünder (ZBS) model.

    This model estimates the effective thermal transport through a two-phase
    throat region composed of solid and fluid, based on the local porosity,
    the ratio of solid to fluid thermal conductivity, and a flattening/contact
    correction derived from the relative contact radius.

    The function can return either:

    - the thermal conductance of the throat [W/K], or
    - the effective throat thermal conductivity [W/m.K] together with porosity.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing throat-scale thermal conductivity data and a
        reference to the associated network.
    solid_volume : str, optional
        Dictionary key of the solid volume assigned to each throat [m³].
    fluid_volume : str, optional
        Dictionary key of the fluid volume assigned to each throat [m³].
    throat_solid_conductivity : str, optional
        Dictionary key of the throat solid thermal conductivity values [W/m.K].
    throat_fluid_conductivity : str, optional
        Dictionary key of the throat fluid thermal conductivity values [W/m.K].
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-]. This quantity is used
        to define the flattening coefficient in the model.
    diameter : str, optional
        Dictionary key of the pore diameter values [m].
    return_conductance : bool, optional
        If ``True``, return throat conductance [W/K]. If ``False``, return a
        tuple containing effective throat conductivity [W/m.K] and porosity [-].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K], if ``return_conductance=True``.

    tuple of ndarray
        Effective throat conductivity [W/m.K] and porosity [-], if
        ``return_conductance=False``.

    Notes
    -----
    If ``solid_volume`` is not available, it is estimated as the sum of two
    hemispherical particle volumes:

    .. math::

        V_s = \frac{2}{3}\pi r_1^3 + \frac{2}{3}\pi r_2^3

    If ``fluid_volume`` is not available, the total unit-cell volume is
    estimated from the larger connected particle diameter and the throat length
    :math:`(r_1 + r_2)`, and the fluid volume is obtained by subtraction.

    If ``return_conductance=True``, the throat conductance is obtained from the
    effective throat conductivity, the cross-sectional area of the unit cell,
    and the throat length.

    If ``relative_contact_radius`` is not present in the network, a default
    value of ``0.009`` is assigned and a warning is issued.
    """

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
        net[relative_contact_radius] = 0.009
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
                effective_radius="throat.effective_radius"):
    r"""
    Calculate thermal conductance using the Tsotsas-ZBS model.

    This model combines the effective throat conductivity obtained from the
    Zehner-Bauer-Schlünder (ZBS) formulation with a geometric scaling based on
    the solid throat volume, local porosity, and effective radius.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing throat-scale thermal conductivity data and a
        reference to the associated network.
    solid_volume : str, optional
        Dictionary key of the solid volume assigned to each throat [m³].
    fluid_volume : str, optional
        Dictionary key of the fluid volume assigned to each throat [m³].
    throat_solid_conductivity : str, optional
        Dictionary key of the throat solid thermal conductivity values [W/m.K].
    throat_fluid_conductivity : str, optional
        Dictionary key of the throat fluid thermal conductivity values [W/m.K].
    relative_contact_radius : str, optional
        Dictionary key of the relative contact radius [-].
    diameter : str, optional
        Dictionary key of the pore diameter values [m].
    effective_radius : str, optional
        Dictionary key of the effective radius [m].

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].

    Notes
    -------
    If ``solid_volume`` is not available, it is estimated from the connected
    particle diameters as the sum of two hemispherical volumes.

    If ``fluid_volume`` is not available, it is estimated from the unit-cell
    volume of the larger connected particle diameter minus the solid volume.

    If ``relative_contact_radius`` is not present in the network, a default
    value of ``0.009`` is assigned and a warning is issued.

    See Also
    --------
    zehner_bauer_schlunder
        Calculates the effective throat conductivity or conductance used by this model.
    """
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
        net[relative_contact_radius] = 0.009
        warnings.warn(f"No {relative_contact_radius} provided in solid phase. Using default value of 0.009")

    throat_conductivity, porosity = zehner_bauer_schlunder(solid_p,
                                                           solid_volume,
                                                           fluid_volume,
                                                           throat_solid_conductivity,
                                                           throat_fluid_conductivity,
                                                           relative_contact_radius,
                                                           diameter,
                                                           False)

    return net[solid_volume]/(4 * (1-porosity) * net[effective_radius] ** 2)*throat_conductivity


def bahrami_rough_joint(
    solid_p,
    pore_solid_conductivity="pore.thermal_solid_conductivity",
    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
    effective_radius="throat.effective_radius",
    diameter="pore.diameter",
    normal_force="throat.normal_force",
    effective_elastic_modulus="throat.effective_elastic_modulus",
    roughness="throat.roughness",
    asperity_slope="throat.asperity_slope",
    microhardness="throat.microhardness",
    vickers_c1="throat.vickers_c1",
    vickers_c2="throat.vickers_c2",
    gas_pressure="throat.gas_pressure",
    gas_temperature="throat.gas_temperature",
    accommodation_coefficient="throat.accommodation_coefficient",
    gas_specific_heat_ratio="throat.gas_specific_heat_ratio",
    gas_prandtl="throat.gas_prandtl",
    gas_mean_free_path_ref="throat.gas_mean_free_path_ref",
    reference_pressure=101325.0,
    reference_temperature=273.15,
    macro_gap_outer_radius=None,
    integration_points=200,
):
    r"""
    Calculate throat thermal conductance using the rough-sphere joint model
    of Bahrami, Yovanovich, and Culham (IJHMT 49, 2006, 3691-3701).

    This implementation returns the conductance of a *single contact region*
    between two rough spheres (or a rough-sphere approximation for a throat).
    It follows the resistance network:

        R_j = [ 1 / ( R_L + (R_s || R_g) ) + 1 / R_G ]^(-1)

    where:
      - R_s : microcontact solid conduction resistance
      - R_L : macrocontact constriction/spreading resistance
      - R_g : gas conduction in the microgap
      - R_G : gas conduction in the macrogap

    Notes
    -----
    1. If `throat.microhardness` is not provided, the function falls back to
       `throat.vickers_c1` and `throat.vickers_c2` using the correlation
       form reported in the paper.
    2. If `macro_gap_outer_radius` is not supplied, the model uses the smaller
       connected pore radius as the outer limit of the macrogap integration.
       This is the closest throat-scale analogue of the SC basic-cell geometry.
    4. The paper gives a closed-form expression for R_G, but to make the
       implementation robust and easier to maintain, this version evaluates
       the same macrogap contribution by direct radial integration.

    Parameters
    ----------
    solid_p : OpenPNM phase-like object
        Phase object containing thermal properties and a reference to the
        associated network.
    pore_solid_conductivity : str
        Key to pore-scale solid thermal conductivity [W/m.K].
    throat_fluid_conductivity : str
        Key to throat-scale gas/fluid thermal conductivity [W/m.K].
    effective_radius : str
        Key to effective radius of curvature [m].
    diameter : str
        Key to pore diameter [m].
    normal_force : str
        Key to normal contact force per throat [N].
    effective_elastic_modulus : str
        Key to effective elastic modulus E' [Pa].
    roughness : str
        Key to combined RMS roughness sigma [m].
    asperity_slope : str
        Key to combined asperity slope m [-].
    microhardness : str
        Key to effective microhardness H_mic [Pa].
    vickers_c1 : str
        Key to Vickers correlation coefficient c1 [Pa], if microhardness is
        not supplied.
    vickers_c2 : str
        Key to Vickers correlation coefficient c2 [-], if microhardness is
        not supplied.
    gas_pressure : str
        Key to gas pressure Pg [Pa].
    gas_temperature : str
        Key to gas temperature Tg [K].
    accommodation_coefficient : str
        Key to thermal accommodation coefficient alpha_T [-].
    gas_specific_heat_ratio : str
        Key to gas specific heat ratio gamma_g [-].
    gas_prandtl : str
        Key to gas Prandtl number Pr [-].
    gas_mean_free_path_ref : str
        Key to reference mean free path Lambda_0 [m].
    reference_pressure : float
        Reference pressure P0 for the mean-free-path correction [Pa].
    reference_temperature : float
        Reference temperature T0 for the mean-free-path correction [K].
    macro_gap_outer_radius : str or None
        Optional key for the outer radius of the macrogap integration [m].
        If None, the smaller connected pore radius is used.
    integration_points : int
        Number of points used for radial integration of R_G.

    Returns
    -------
    ndarray
        Thermal conductance of each throat [W/K].
    """
    net = solid_p.network
    conns = net.conns

    # ---------- helper: inverse complementary error function approximation ----------
    def _erfcinv_approx(x):
        """
        Approximation used in the paper for erfc^{-1}(x).
        Valid roughly for 1e-9 <= x <= 1.9
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 1e-9, 1.9)
        out = np.empty_like(x)

        mask1 = (x >= 1e-9) & (x <= 0.02)
        mask2 = (x > 0.02) & (x <= 0.5)
        mask3 = (x > 0.5) & (x <= 1.9)

        out[mask1] = 1.0 / (0.218 + 0.735 * x[mask1]**0.173)
        out[mask2] = 1.05 * (0.175 - x[mask2]) / (x[mask2] - 0.12)
        out[mask3] = (1.0 - x[mask3]) / (
            0.707 + 0.862 * x[mask3] - 0.431 * x[mask3]**2
        )
        return out

    # ---------- helper: harmonic mean conductivity between connected pores ----------
    k1 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 0]], dtype=float)
    k2 = np.asarray(solid_p[pore_solid_conductivity][conns[:, 1]], dtype=float)
    k_s = np.zeros_like(k1, dtype=float)
    valid_k = (k1 > 0.0) & (k2 > 0.0)
    k_s[valid_k] = 2.0 * k1[valid_k] * k2[valid_k] / (k1[valid_k] + k2[valid_k])

    # ---------- required throat / network data ----------
    k_g = np.asarray(solid_p[throat_fluid_conductivity], dtype=float)
    q_eff = np.asarray(net[effective_radius], dtype=float)              # effective radius of curvature
    F = np.asarray(net[normal_force], dtype=float)
    Eeff = np.asarray(net[effective_elastic_modulus], dtype=float)
    sigma = np.asarray(net[roughness], dtype=float)
    slope = np.asarray(net[asperity_slope], dtype=float)

    r1 = np.asarray(net[diameter][conns[:, 0]], dtype=float) / 2.0
    r2 = np.asarray(net[diameter][conns[:, 1]], dtype=float) / 2.0
    q_local = np.minimum(r1, r2)  # geometric throat-scale closure for outer gap size

    # ---------- microhardness ----------
    if microhardness in net.keys():
        Hmic = np.asarray(net[microhardness], dtype=float)
        H0 = Hmic.copy()
    elif microhardness in solid_p.keys():
        Hmic = np.asarray(solid_p[microhardness], dtype=float)
        H0 = Hmic.copy()
    else:
        if vickers_c1 not in net.keys() or vickers_c2 not in net.keys():
            raise KeyError(
                "Provide either 'throat.microhardness' or both "
                "'throat.vickers_c1' and 'throat.vickers_c2'."
            )
        c1 = np.asarray(net[vickers_c1], dtype=float)
        c2 = np.asarray(net[vickers_c2], dtype=float)
        # paper uses r0' = sigma / (1 micron)
        sigma_um = sigma / 1e-6
        ratio = np.maximum(sigma_um / np.maximum(slope, 1e-30), 1e-30)
        Hmic = c1 * ratio**c2
        H0 = c1 * np.maximum(1.62 * sigma_um / np.maximum(slope, 1e-30), 1e-30)**c2

    # ---------- gas rarefaction parameter ----------
    Pg = np.asarray(net[gas_pressure], dtype=float)
    Tg = np.asarray(net[gas_temperature], dtype=float)
    alpha = np.asarray(net[accommodation_coefficient], dtype=float)
    gamma_g = np.asarray(net[gas_specific_heat_ratio], dtype=float)
    Pr = np.asarray(net[gas_prandtl], dtype=float)
    Lambda0 = np.asarray(net[gas_mean_free_path_ref], dtype=float)

    # mean free path, paper Eq. (9)
    Lambda = (reference_pressure / np.maximum(Pg, 1e-30)) * (Tg / reference_temperature) * Lambda0

    # gas parameter M, assuming same accommodation coefficient on both sides
    # M = [ (2-alpha)/alpha + (2-alpha)/alpha ] * [2*gamma/(gamma+1)] * (1/Pr) * Lambda
    M = 2.0 * ((2.0 - alpha) / np.maximum(alpha, 1e-30)) \
        * (2.0 * gamma_g / (gamma_g + 1.0)) \
        * (1.0 / np.maximum(Pr, 1e-30)) \
        * Lambda

    # ---------- contact mechanics ----------
    # Smooth-sphere Hertz radius
    a_H = np.zeros_like(F, dtype=float)
    valid_hertz = (F > 0.0) & (q_eff > 0.0) & (Eeff > 0.0)
    a_H[valid_hertz] = (0.75 * F[valid_hertz] * q_eff[valid_hertz] / Eeff[valid_hertz])**(1.0 / 3.0)

    P0_H = np.zeros_like(F, dtype=float)
    mask_PH = valid_hertz & (a_H > 0.0)
    P0_H[mask_PH] = 1.5 * F[mask_PH] / (np.pi * a_H[mask_PH]**2)

    # paper Eq. (7)
    alpha_cm = np.zeros_like(F, dtype=float)
    j = np.zeros_like(F, dtype=float)
    mask_cm = valid_hertz & (a_H > 0.0) & (Hmic > 0.0) & (sigma >= 0.0)
    alpha_cm[mask_cm] = sigma[mask_cm] * q_eff[mask_cm] / np.maximum(a_H[mask_cm]**2, 1e-30)
    j[mask_cm] = (Eeff[mask_cm] / np.maximum(Hmic[mask_cm], 1e-30)) \
                 * np.sqrt(np.maximum(sigma[mask_cm], 0.0) / np.maximum(q_eff[mask_cm], 1e-30))

    # paper Eq. (5)
    P0_ratio = np.ones_like(F, dtype=float)
    mask_ratio = mask_cm & (j > 0.0)
    P0_ratio[mask_ratio] = 1.0 / (1.0 + 1.22 * alpha_cm[mask_ratio] * j[mask_ratio]**(-0.16))
    P0_ratio = np.clip(P0_ratio, 0.01, 1.0)

    # paper Eq. (6)
    aL_over_aH = np.ones_like(F, dtype=float)
    mask_low = P0_ratio <= 0.47
    mask_high = P0_ratio > 0.47
    aL_over_aH[mask_low] = 1.605 / np.sqrt(P0_ratio[mask_low])
    aL_over_aH[mask_high] = 3.51 - 2.51 * P0_ratio[mask_high]

    a_L = aL_over_aH * a_H
    P0 = P0_ratio * P0_H

    # ---------- resistances ----------
    # (1) microcontact solid resistance, paper Eq. (2)
    R_s = np.full_like(F, np.inf, dtype=float)
    mask_Rs = (k_s > 0.0) & (F > 0.0) & (slope > 0.0)

    # Smooth-surface limit: sigma -> 0 => Rs -> 0
    smooth = mask_Rs & (sigma <= 0.0)
    R_s[smooth] = 0.0

    rough = mask_Rs & (sigma > 0.0)
    R_s[rough] = 0.565 * Hmic[rough] * (sigma[rough] / slope[rough]) / (
        np.maximum(k_s[rough], 1e-30) * F[rough]
    )

    # (2) macrocontact spreading resistance, paper Eq. (8)
    R_L = np.full_like(F, np.inf, dtype=float)
    mask_RL = (k_s > 0.0) & (a_L > 0.0)
    R_L[mask_RL] = 1.0 / (2.0 * k_s[mask_RL] * a_L[mask_RL])

    # (3) microgap gas resistance, paper Eq. (13)
    R_g = np.full_like(F, np.inf, dtype=float)

    # smooth limit from paper Eq. (21)
    mask_Rg_smooth = (sigma <= 0.0) & (k_g > 0.0) & (a_H > 0.0)
    R_g[mask_Rg_smooth] = M[mask_Rg_smooth] / (
        np.pi * k_g[mask_Rg_smooth] * a_H[mask_Rg_smooth]**2
    )

    # rough case
    mask_Rg = (sigma > 0.0) & (k_g > 0.0) & (a_L > 0.0) & (H0 > 0.0) & (P0 > 0.0)
    if np.any(mask_Rg):
        x1 = np.clip(2.0 * P0[mask_Rg] / H0[mask_Rg], 1e-9, 1.9)
        x2 = np.clip(0.03 * P0[mask_Rg] / H0[mask_Rg], 1e-9, 1.9)

        a1 = _erfcinv_approx(x1)
        a2 = _erfcinv_approx(x2) - a1

        denom = a1 + M[mask_Rg] / np.maximum(2.0 * np.sqrt(2.0) * sigma[mask_Rg], 1e-30)
        logterm = np.log(1.0 + a2 / np.maximum(denom, 1e-30))

        R_g[mask_Rg] = (
            2.0 * np.sqrt(2.0) * sigma[mask_Rg]
            / (np.pi * k_g[mask_Rg] * a_L[mask_Rg]**2)
        ) * logterm

    # (4) macrogap gas resistance, paper Eq. (14)
    # We evaluate the same macrogap contribution numerically:
    # G_G = ∫[a_L -> b_L] 2*pi*k_g*r / (gap(r) + M) dr
    # with gap(r) = 2*(q - x0) - 2*sqrt(q^2 - r^2)
    # and R_G = 1 / G_G
    R_G = np.full_like(F, np.inf, dtype=float)

    if macro_gap_outer_radius is None:
        b_L = q_local.copy()
    else:
        b_L = np.asarray(net[macro_gap_outer_radius], dtype=float)

    x0 = np.zeros_like(F, dtype=float)
    mask_x0 = q_local > 0.0
    x0[mask_x0] = a_L[mask_x0]**2 / (2.0 * q_local[mask_x0])

    for i in range(net.Nt):
        if not (
            np.isfinite(a_L[i])
            and np.isfinite(b_L[i])
            and np.isfinite(q_local[i])
            and np.isfinite(k_g[i])
            and np.isfinite(M[i])
            and (k_g[i] > 0.0)
            and (q_local[i] > 0.0)
            and (b_L[i] > a_L[i])
        ):
            continue

        r = np.linspace(a_L[i], b_L[i], integration_points)
        root = np.sqrt(np.maximum(q_local[i]**2 - r**2, 0.0))
        gap = 2.0 * (q_local[i] - x0[i]) - 2.0 * root
        denom = np.maximum(gap + M[i], 1e-30)

        integrand = 2.0 * np.pi * k_g[i] * r / denom
        Gg = np.trapezoid(integrand, r)

        if Gg > 0.0:
            R_G[i] = 1.0 / Gg

    # ---------- combine resistances ----------
    # Rs || Rg
    R_parallel_micro = np.full_like(F, np.inf, dtype=float)
    inv_micro = np.zeros_like(F, dtype=float)

    mask_rs = np.isfinite(R_s) & (R_s > 0.0)
    inv_micro[mask_rs] += 1.0 / R_s[mask_rs]

    # If Rs == 0, the parallel branch is zero resistance
    mask_rs_zero = (R_s == 0.0)
    R_parallel_micro[mask_rs_zero] = 0.0

    mask_rg = np.isfinite(R_g) & (R_g > 0.0)
    inv_micro[mask_rg] += 1.0 / R_g[mask_rg]

    mask_micro = (R_parallel_micro != 0.0) & (inv_micro > 0.0)
    R_parallel_micro[mask_micro] = 1.0 / inv_micro[mask_micro]

    # series branch: R_L + (R_s || R_g)
    R_series = np.full_like(F, np.inf, dtype=float)
    finite_series = np.isfinite(R_L) & np.isfinite(R_parallel_micro)
    R_series[finite_series] = R_L[finite_series] + R_parallel_micro[finite_series]

    # total: [1/R_series + 1/R_G]^{-1}
    conductance = np.zeros_like(F, dtype=float)
    inv_total = np.zeros_like(F, dtype=float)

    mask_series = np.isfinite(R_series) & (R_series > 0.0)
    inv_total[mask_series] += 1.0 / R_series[mask_series]

    mask_RG = np.isfinite(R_G) & (R_G > 0.0)
    inv_total[mask_RG] += 1.0 / R_G[mask_RG]

    valid_total = inv_total > 0.0
    conductance[valid_total] = 1.0 / (1.0 / inv_total[valid_total])

    # Clean up tiny numerical noise
    conductance = np.maximum(conductance, 0.0)
    return conductance