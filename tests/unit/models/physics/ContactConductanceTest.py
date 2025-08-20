import numpy as np
from numpy.testing import assert_allclose

import openpnm as op


def mean_throat_curvature(pn):
    r1, r2 = (pn['pore.diameter'][pn.conns] / 2).T
    return 2 * r1 * r2 / (r1 + r2)


def throat_length(pn):
    r1, r2 = (pn['pore.diameter'][pn.conns] / 2).T
    l_total = np.sqrt(np.sum(np.diff(pn.coords[pn.conns], axis=1).squeeze() ** 2, axis=1))
    lt = l_total - r1 - r2
    return lt


class ThermalConductanceTest:

    def setup_class(self):
        self.net = op.network.Cubic((4, 4, 4), spacing=1e-2)
        self.net["pore.diameter"] = 1e-2
        self.net["throat.mean_curvature"] = mean_throat_curvature(self.net)
        self.net["throat.length"] = throat_length(self.net)
        self.net["throat.boundary"] = False
        self.net["pore.boundary"] = False
        self.net["throat.relative_mean_curvature_radius"] = 0.5
        self.net["throat.relative_bridge_radius"] = 0.1
        self.net["throat.relative_contact_throat_radius"] = 0.01
        self.solid = op.phase.Phase(network=self.net)
        self.solid['pore.thermal_conductivity'] = 42
        self.solid['throat.thermal_solid_conductivity'] = np.min(
            self.solid['pore.thermal_conductivity'][self.net.conns], 1)
        self.solid['throat.thermal_fluid_conductivity'] = 0.0243

    def test_yovanovitch(self):
        mod = op.models.physics.thermal_conductance.yovanovitch
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Yovanovitch is: {actual} W/K")

    def test_dixon(self):
        mod = op.models.physics.thermal_conductance.dixon_bridge_model
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Dixon is: {actual} W/K")

    def test_batchelor(self):
        mod = op.models.physics.thermal_conductance.batchelor
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Batchelor is: {actual} W/K")

    def test_kunii_smith(self):
        mod = op.models.physics.thermal_conductance.kunii_smith
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Kunii is: {actual} W/K")

    def test_tsotsas(self):
        mod = op.models.physics.thermal_conductance.tsotsas
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Tsotsas is: {actual} W/K")

    def test_argento(self):
        mod = op.models.physics.thermal_conductance.argento
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Argento is: {actual} W/K")

    def test_fei(self):
        mod = op.models.physics.thermal_conductance.fei_narsilio
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Fei is: {actual} W/K")

    def test_birkholz(self):
        mod = op.models.physics.thermal_conductance.birkholz
        self.solid.add_model(propname='throat.thermal_conductance', model=mod)
        self.solid.regenerate_models()
        actual = self.solid['throat.thermal_conductance'].mean()
        print(f" Average conductance Birkholz is: {actual} W/K")


if __name__ == '__main__':

    t = ThermalConductanceTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
