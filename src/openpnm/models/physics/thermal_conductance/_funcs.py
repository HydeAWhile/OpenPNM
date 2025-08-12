from openpnm.models.physics._utils import _poisson_conductance
from openpnm.models import _doctxt
import warnings
import numpy as np
from scipy import integrate

__all__ = ["generic_thermal",
           "series_resistors",
           "yovanovitch",
           "dixon_bridge_model",
           "extended_dixon_bridge_model",
           "batchelor",
           "kunii_smith",
           "tsotsas",
           "argento",
           "fei_narsilio"
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


def yovanovitch(solid_p,
                throat_solid_conductivity="throat.thermal_solid_conductivity",
                relative_contact_radius="throat.relative_contact_throat_radius",
                mean_curvature="throat.mean_curvature",
                relative_gas_conductivity_radius=1e12,
                radiation_exchange_factor=0
                ):
    r"""
    Calculates conductance based on Yovanovitch 1967 contact model. DOI: 10.2514/3.28821
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
        warnings.warn("No relative contact radius provided in solid phase. Using default value of 0.009")

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


def dixon_bridge_model(solid_p,
                       pore_thermal_conductivity="pore.thermal_conductivity",
                       throat_solid_conductivity="throat.thermal_solid_conductivity",
                       throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                       relative_bridge_radius="throat.relative_bridge_radius",
                       diameter="pore.diameter",
                       mean_curvature="throat.mean_curvature",
                       throat_length="throat.length"
                       ):
    r"""
    Calculates conductance based on Bridge model published by Dixon et al. 2013. DOI: 10.1016/j.compchemeng.2012.08.011
    Problem of the model is that it requires the particles to be in very near contact,
    as it does not include a term which would compensate for a larger gap.

    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    throat_solid_conductivity : str
        %(dict_burb)s throat thermal conductivity
    relative_bridge_radius : str
        %(dict_burb)s relative contact radius
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    throat_length : str
        %(dict_burb)s throat length

    diameter : str
        %(dict_burb)s diameter

    Returns
    -------
    %(return_arr)s

    """

    def h_f(r: float) -> float:
        """
        Calculates length of the fluid section of the bridge at given radius
        :param r: _h_f = _h_f(r), float
        :return: Distance _h_f, float
        """
        return r_particle[i] - np.sqrt(r_particle[i] ** 2 - r ** 2)

    def h_s(length_fluid: float) -> float:
        """
        Calculates length of the solid section of the bridge at given radius
        :param length_fluid: Size of the fluid part, float
        :return: Distance _h_s, float
        """
        return length_bridge[i] - length_fluid

    def integral(r) -> float:
        """
        Integrated function of the bridge model
        :param length_bridge: Size of the bridge, float
        :param r: f = f(r), float

        :return: Integration function of the bridge model, float
        """
        _h_f = h_f(r)
        _h_s = h_s(_h_f)
        return r / (_h_f * solid_p[throat_solid_conductivity][i] + _h_s * solid_p[throat_fluid_conductivity][i]) # Thermal conductivities factored ot before integral

    net = solid_p.network

    # Checks, whether a contact radius value was inserted, if not, than
    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn("No relative bridge radius provided in solid phase. Using default value of 0.1")

    # Sets the particle radius as the smaller of the two nodes.
    r_particle = np.min(net[diameter][net.conns], 1)

    # Uses throar radius based on Batch O'Brien definition
    # Calculates the bridge radius as fraction of the particle radius
    r_bridge = r_particle * net[relative_bridge_radius]

    # Calculates bridge length to the symmetry plane
    i = range(0, len(r_bridge))
    length_bridge = h_f(r_bridge)

    # Integates heat flux at different radia to get a total heat flow through the bridge
    integral_value = np.empty_like(r_bridge)
    for i, radius in enumerate(r_bridge):
        _integral = integrate.quad(lambda r: integral(r), 0, radius)
        integral_value[i] = _integral[0]

    # Calculates effective conductivity based of the bridge (the model utilises half symmetry)
    effective_conductivity = 2 * length_bridge / (r_bridge ** 2) * integral_value[:] * (
            solid_p[throat_solid_conductivity] * solid_p[throat_fluid_conductivity])
    # LAST BRACKET IS A CONSTANT FACTORED OUT FROM THE INTEGRAL

    # Sets the total length of the bridge as 2x the distance to the symmetry plane (only half in article!)
    # + gap between particles
    bridge_length = 2 * length_bridge + net[throat_length]

    # Calculates the cross section of the bridge
    bridge_crossection = np.pi * r_bridge ** 2

    # Calculates the thermal conductacnce of the bridge in W/K
    conductance = (effective_conductivity * bridge_crossection) / bridge_length
    return conductance


def extended_dixon_bridge_model(solid_p,
                                pore_thermal_conductivity="pore.thermal_conductivity",
                                throat_solid_conductivity="throat.thermal_solid_conductivity",
                                throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                                relative_bridge_radius="throat.relative_bridge_radius",
                                diameter="pore.diameter",
                                mean_curvature="throat.mean_curvature",
                                throat_length="throat.length"):
    r"""
    Parameters
    ----------
    %(solid_p)s
    pore_thermal_conductivity : str
        %(dict_burb)s pore thermal conductivity
    throat_fluid_conductivity : str
        %(dict_burb)s throat thermal conductivity
    relative_bridge_radius : str
        %(dict_burb)s relative contact radius
    mean_curvature : str
        %(dict_burb)s mean curvature
        Average curvature of the two particles in the proximity point of their contact point. Usually calculated
        using formula:

        .. math::

            R = \frac{2 * R1 * R2}{R1 + R2}

    throat_length : str
        %(dict_burb)s throat length

    diameter : str
        %(dict_burb)s diameter

    %(fluid_p)s
    throat_fluid_conductivity : str
        %(dict_burb)s thermal conductivity

    Returns
    -------
    %(return_arr)s

    """

    def h_f(r: float) -> float:
        """
        Calculates fluid part of the bridge at given radius
        :param r: _h_f = _h_f(r), float
        :return: Distance _h_f, float
        """
        return r_particle[i] - np.sqrt(r_particle[i] ** 2 - r ** 2)

    def h_s(length_fluid: float) -> float:
        """
        Calculates solid part of the bridge at given radius
        :param length_fluid: Size of the fluid part, float
        :return: Distance _h_s, float
        """
        return length_bridge[i] - length_fluid

    def integral(r) -> float:
        """
        Integrated function of the bridge model
        :param r: f = f(r), float

        :return: Integration function of the bridge model, float
        """
        _h_f = h_f(r)
        _h_s = h_s(_h_f)
        return r / (_h_f * solid_p[throat_solid_conductivity][i] + _h_s * solid_p[throat_fluid_conductivity][i])

    net = solid_p.network

    if relative_bridge_radius not in net.keys():
        net[relative_bridge_radius] = 0.1
        warnings.warn("No relative bridge radius provided in solid phase. Using default value of 0.1")

    # Sets the particle radius as the smaller of the two nodes.
    r_particle = np.min(net[diameter][net.conns], 1)

    # Uses throar radius based on Batch O'Brien definition
    # Calculates the bridge radius as fraction of the particle radius
    r_bridge = r_particle * net[relative_bridge_radius]

    # Calculates bridge length to the symmetry plane
    i = range(0, len(r_bridge))
    length_bridge = h_f(r_bridge)

    # Integates heat flux at different radia to get a total heat flow through the bridge
    integral_value = np.empty_like(r_bridge)
    for i, radius in enumerate(r_bridge):
        _integral = integrate.quad(lambda r: integral(r), 0, radius)
        integral_value[i] = _integral[0]

    # Calculates effective conductivity based of the bridge (the model utilises half symmetry)
    effective_conductivity = 2 * length_bridge / (r_bridge ** 2) * integral_value[:] * (
            solid_p[throat_solid_conductivity] * solid_p[throat_fluid_conductivity])
    # LAST BRACKET IS A CONSTANT FACTORED OUT FROM THE INTEGRAL

    # Sets the total length of the bridge as 2x the distance to the symmetry plane (only half in article!)
    # + gap between particles
    bridge_length = 2 * length_bridge + net[throat_length]

    # Calculates the cross section of the bridge
    bridge_crossection = np.pi * r_bridge ** 2

    # Calculates the thermal conductacnce of the bridge in W/K
    conductance = (effective_conductivity * bridge_crossection) / bridge_length
    return conductance

def batchelor(solid_p,
              pore_thermal_conductivity="pore.thermal_conductivity",
              throat_solid_conductivity="throat.thermal_solid_conductivity",
              throat_fluid_conductivity="throat.thermal_fluid_conductivity",
              mean_curvature="throat.mean_curvature",
              relative_contact_radius="throat.relative_contact_throat_radius",
              throat_length="throat.length",
              effective_mean_radius_curvature_fraction="throat.relative_mean_curvature_radius"
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
        warnings.warn("No relative curvature fraction provided in solid phase. Using default value of 0.5")

    if relative_contact_radius not in net.keys():
        no_contact = True
#    elif solid_p[relative_contact_radius] <= 0:
#       no_contact = True DODELAT - potreba kontroly pro kazdy element
    else:
        no_contact = False

    conductivity_ratios = (solid_p[throat_solid_conductivity] /
                           solid_p[throat_fluid_conductivity])  # alpha in text
    effective_mean_particle_radius = net[mean_curvature]*net[effective_mean_radius_curvature_fraction]

    Cc = np.zeros_like(conductivity_ratios)
    C1p = np.zeros_like(conductivity_ratios)
    C2p = np.zeros_like(conductivity_ratios)
    if no_contact:
        separation_parameters = np.square(conductivity_ratios) * net[throat_length] / net[mean_curvature] # lambda in text
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

    conductance = 1/(1/C1p + 1/Cc + 1/C2p)

    return conductance


def batchelor_old(solid_p,
                  pore_thermal_conductivity="pore.thermal_conductivity",
                  throat_solid_conductivity="throat.thermal_conductivity",
                  throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                  mean_curvature="throat.mean_curvature",
                  throat_length="throat.length",
                  mean_radius_curvature_fraction="throat.relative_mean_curvature_radius"
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
        warnings.warn("No relative curvature fraction provided in solid phase. Using default value of 0.5")


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
                boundary_throats="throat.boundary",
                n=1.45  # The number of contacts per hemisphere should be between 1.42 and 1.5
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

    costheta = np.sqrt(1-1/n)

    contact_area = np.pi * net[mean_curvature] ** 2 * 1 / n
    # Area influenced by the contacts
    # sin2(theta) substituted by 1/n.
    # Original paper presumes particles of same size, Batchelors mean curvature was thought to
    # better represent this parameter.

    kappa = solid_p[throat_solid_conductivity] / solid_p[throat_fluid_conductivity]
    # Original paper presumes same material, modified with the presumption, that lower conductivity influences the
    # system more.

    fluid_film_thicknes = ((net[mean_curvature] * 2 * 1 / 2 * ((kappa - 1) / kappa) ** 2 * 1 / n) /
                           (np.log(kappa - (kappa - 1) * costheta) - (kappa - 1) / kappa * (1 - costheta))
                           * (2 / 3 * 1 / kappa))

    def solid_resistance(particle_radius, solid_cond, ind):
        return 2 * particle_radius * (2 / 3) / (solid_cond * contact_area[ind])

    def fluid_resistance(ind):
        return fluid_film_thicknes[ind] / (solid_p[throat_fluid_conductivity][ind] * contact_area[ind])

    def boundary_resistance(boundary_dimension, radius_of_influence, solid_cond) -> float:
        """
            Modifying function not in the original publication.
            Calculates resistance of the bidge, if one of the nodes is a boundary node (representing a wall)
            :return: Resistance of the fluid gap, float, K/W
            """
        return boundary_dimension / (solid_cond * np.pi * radius_of_influence ** 2)
        # return boundary_dimension/(boundary_conductivity*np.pi*r_bridge**2) # Overestimates resistance due to very thin cylinder

    r1, r2 = (net[diameter][net.conns] / 2).T
    resistance = np.zeros_like(contact_area)
    for i, conns in enumerate(net.conns):
        particle_res = solid_resistance(r1[i], particle_conductivities[i][0], i)
        resistance[i] = particle_res
        if not net[boundary_throats][i]:
            solid_res = solid_resistance(r2[i], particle_conductivities[i][1], i)
            resistance[i] += solid_res
        else:
            boundary_res = boundary_resistance(r1[i], r2[i], particle_conductivities[i][1])
            resistance[i] += boundary_res

        fluid_res = fluid_resistance(i)
        resistance[i] += fluid_res

    return 1 / resistance

def tsotsas(solid_p,
            pore_thermal_conductivity="pore.thermal_conductivity",
            throat_solid_conductivity="throat.thermal_solid_conductivity",
            relative_contact_radius="throat.relative_contact_throat_radius",
            mean_curvature="throat.mean_curvature"):
    net = solid_p.network

    if relative_contact_radius not in net.keys():
        net[relative_contact_radius] = 0.009
        warnings.warn("No relative contact radius provided in solid phase. Using default value of 0.009")

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
        warnings.warn("No relative contact radius provided in solid phase. Using default value of 0.009")

    bridge_len = net[throat_lenght] + np.sum((net['pore.diameter'][net.conns] / 2), axis=1)

    particle_cross_section = np.pi * np.square(net[mean_curvature])
    resistance = 0.889 / (net[relative_contact_radius] * solid_p[throat_solid_conductivity] * particle_cross_section) * bridge_len

    conductance = 1 / resistance

    return conductance


def fei_narsilio(solid_p,
                 pore_thermal_conductivity="pore.thermal_conductivity",
                 throat_solid_conductivity="throat.thermal_solid_conductivity",
                 throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                 relative_contact_radius="throat.relative_contact_throat_radius",
                 relative_bridge_radius="throat.relative_bridge_radius",
                 mean_curvature="throat.mean_curvature",
                 throat_lenght="throat.length",
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
        return 1 / _conductance

    net = solid_p.network

    if relative_contact_radius not in net.keys():
        solid_p[relative_contact_radius] = 0.009
        warnings.warn("No relative contact radius provided in solid phase. Using default value of 0.009")

    if relative_bridge_radius not in net.keys():
        solid_p[relative_bridge_radius] = 0.1
        warnings.warn("No relative bridge radius provided in solid phase. Using default value of 0.1")

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
        Cc = contact_conductance(contact_radius, net[mean_curvature] / 20, 0.9, solid_p[throat_solid_conductivity])

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
