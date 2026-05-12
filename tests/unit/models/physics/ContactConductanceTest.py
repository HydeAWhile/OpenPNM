import argparse
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

import openpnm as op


# ---Main chapter---
def pore_boundary_mask(pn):
    """
    Mark some pores as boundary pores.

    Here, all pores on the z-min and z-max faces are marked as boundary pores.
    This gives a realistic subset of boundary pores without making every pore
    a boundary pore.
    """
    z = pn.coords[:, 2]
    zmin = np.min(z)
    zmax = np.max(z)
    return np.isclose(z, zmin) | np.isclose(z, zmax)


def throat_boundary_mask(pn):
    """
    A throat is a boundary throat if at least one connected pore is a boundary pore.
    """
    p_boundary = np.asarray(pn["pore.boundary"], dtype=bool)
    return np.any(p_boundary[pn.conns], axis=1)


def pore_diameter(pn):
    """
    Spatially varying, physically sound pore diameters.

    Range is chosen so that all pore diameters are safely below the network
    spacing (1e-2 m), which guarantees positive throat lengths after
    subtracting the two pore radii from the center-to-center distance.
    """
    xyz = pn.coords
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Normalize coordinates to [0, 1]
    x = x / np.max(x) if np.max(x) > 0 else x
    y = y / np.max(y) if np.max(y) > 0 else y
    z = z / np.max(z) if np.max(z) > 0 else z

    # Diameter range: 2.8 mm to 4.4 mm
    d = 2.8e-3 + 0.5e-3 * x + 0.5e-3 * y + 0.6e-3 * z
    return d


def effective_radius(pn):
    """
    Harmonic-mean-like effective radius from the two connected pore radii.
    """
    r1, r2 = (pn["pore.diameter"][pn.conns] / 2).T
    return 2.0 * r1 * r2 / (r1 + r2)


def throat_length(pn):
    """
    Geometric throat length = center distance - pore radii.
    """
    r1, r2 = (pn["pore.diameter"][pn.conns] / 2).T
    l_total = np.sqrt(
        np.sum(np.diff(pn.coords[pn.conns], axis=1).squeeze() ** 2, axis=1)
    )
    lt = l_total - r1 - r2
    return lt


def relative_bridge_radius(pn):
    """
    Spatially varying bridge radius fraction.

    Chosen in the range ~0.08 to 0.18, i.e. positive and well below 1.
    """
    c = pn.coords[pn.conns].mean(axis=1)
    x = c[:, 0] / np.max(pn.coords[:, 0]) if np.max(pn.coords[:, 0]) > 0 else c[:, 0]
    y = c[:, 1] / np.max(pn.coords[:, 1]) if np.max(pn.coords[:, 1]) > 0 else c[:, 1]
    z = c[:, 2] / np.max(pn.coords[:, 2]) if np.max(pn.coords[:, 2]) > 0 else c[:, 2]
    rb = 0.08 + 0.03 * x + 0.03 * y + 0.04 * z
    return rb


def relative_contact_throat_radius(pn):
    """
    Spatially varying contact radius fraction.

    Chosen smaller than bridge radius and in a realistic small range.
    """
    rb = relative_bridge_radius(pn)
    rc = 0.003 + 0.05 * rb
    return np.minimum(rc, 0.012)


def relative_effective_radius(pn):
    """
    Spatially varying effective radius fraction for batchelor.

    Positive and within a plausible range discussed in the model comments.
    """
    c = pn.coords[pn.conns].mean(axis=1)
    x = c[:, 0] / np.max(pn.coords[:, 0]) if np.max(pn.coords[:, 0]) > 0 else c[:, 0]
    y = c[:, 1] / np.max(pn.coords[:, 1]) if np.max(pn.coords[:, 1]) > 0 else c[:, 1]
    z = c[:, 2] / np.max(pn.coords[:, 2]) if np.max(pn.coords[:, 2]) > 0 else c[:, 2]
    return 0.45 + 0.08 * x + 0.10 * y + 0.12 * z


def solid_and_fluid_volumes(pn):
    """
    Physically sound unit-cell-like solid and fluid volumes used by ZBS models.

    solid_volume:
        sum of two hemispherical particle volumes

    total_volume:
        conservative unit cell based on the larger connected particle diameter
        and the connection length (r1 + r2)

    fluid_volume:
        clipped to remain strictly positive
    """
    r1, r2 = (pn["pore.diameter"][pn.conns] / 2.0).T
    solid_volume = 2.0 / 3.0 * np.pi * r1**3 + 2.0 / 3.0 * np.pi * r2**3
    total_volume = (2.0 * np.maximum(r1, r2)) ** 2 * (r1 + r2)
    fluid_volume = np.maximum(total_volume - solid_volume, 1e-30)
    return solid_volume, fluid_volume


def throat_thermal_conductivity(pn):
    """
    A positive, spatially varying generic throat thermal conductivity
    for generic_thermal and series_resistors.
    """
    c = pn.coords[pn.conns].mean(axis=1)
    x = c[:, 0] / np.max(pn.coords[:, 0]) if np.max(pn.coords[:, 0]) > 0 else c[:, 0]
    y = c[:, 1] / np.max(pn.coords[:, 1]) if np.max(pn.coords[:, 1]) > 0 else c[:, 1]
    z = c[:, 2] / np.max(pn.coords[:, 2]) if np.max(pn.coords[:, 2]) > 0 else c[:, 2]
    return 0.22 + 0.04 * x + 0.05 * y + 0.06 * z


def pore_thermal_conductivity(pn):
    """
    Positive, spatially varying pore solid conductivity.
    """
    xyz = pn.coords
    x = xyz[:, 0] / np.max(xyz[:, 0]) if np.max(xyz[:, 0]) > 0 else xyz[:, 0]
    y = xyz[:, 1] / np.max(xyz[:, 1]) if np.max(xyz[:, 1]) > 0 else xyz[:, 1]
    z = xyz[:, 2] / np.max(xyz[:, 2]) if np.max(xyz[:, 2]) > 0 else xyz[:, 2]
    return 18.0 + 8.0 * x + 10.0 * y + 12.0 * z


def throat_thermal_solid_conductivity(pn, phase):
    """
    Throat solid conductivity from the smaller connected pore conductivity.
    """
    return np.min(phase["pore.thermal_conductivity"][pn.conns], axis=1)


def throat_thermal_fluid_conductivity(pn):
    """
    Positive, spatially varying throat fluid conductivity in a gas-like range.
    """
    c = pn.coords[pn.conns].mean(axis=1)
    x = c[:, 0] / np.max(pn.coords[:, 0]) if np.max(pn.coords[:, 0]) > 0 else c[:, 0]
    y = c[:, 1] / np.max(pn.coords[:, 1]) if np.max(pn.coords[:, 1]) > 0 else c[:, 1]
    z = c[:, 2] / np.max(pn.coords[:, 2]) if np.max(pn.coords[:, 2]) > 0 else c[:, 2]
    return 0.022 + 0.003 * x + 0.004 * y + 0.005 * z


def diffusive_size_factors(pn):
    """
    Physically sound, positive conduit diffusive size factors.

    These are simple A/L-like factors for:
        pore 1 half,
        throat,
        pore 2 half

    They are deterministic and vary with local geometry.
    """
    eps = 1e-30
    r1, r2 = (pn["pore.diameter"][pn.conns] / 2.0).T
    Lt = pn["throat.length"]

    # Effective local radii for pore halves and throat section
    rp1 = 0.55 * r1
    rp2 = 0.55 * r2
    rt = 0.35 * np.minimum(r1, r2)

    # Characteristic lengths
    L1 = np.maximum(0.50 * r1, eps)
    L2 = np.maximum(0.50 * r2, eps)
    Lt = np.maximum(Lt, eps)

    F1 = np.pi * rp1**2 / L1
    Ft = np.pi * rt**2 / Lt
    F2 = np.pi * rp2**2 / L2

    return np.vstack([F1, Ft, F2]).T


# ---Main chapter---
class TestThermalConductanceGoldenMaster:
    BASELINE_FILE = Path(__file__).with_name("thermal_conductance_golden_master.npz")

    def setup_class(self):
        self.net = op.network.Cubic((4, 4, 4), spacing=1e-2)

        # Spatially varying, physically sound network properties
        self.net["pore.diameter"] = pore_diameter(self.net)
        self.net["pore.boundary"] = pore_boundary_mask(self.net)
        self.net["throat.boundary"] = throat_boundary_mask(self.net)

        self.net["throat.effective_radius"] = effective_radius(self.net)
        self.net["throat.length"] = throat_length(self.net)
        self.net["throat.relative_bridge_radius"] = relative_bridge_radius(self.net)
        self.net["throat.relative_contact_throat_radius"] = relative_contact_throat_radius(self.net)
        self.net["throat.relative_effective_radius"] = relative_effective_radius(self.net)

        solid_volume, fluid_volume = solid_and_fluid_volumes(self.net)
        self.net["throat.solid_volume"] = solid_volume
        self.net["throat.fluid_volume"] = fluid_volume

        # Phase properties
        self.solid = op.phase.Phase(network=self.net)
        self.solid["pore.thermal_conductivity"] = pore_thermal_conductivity(self.net)
        self.solid["throat.thermal_solid_conductivity"] = throat_thermal_solid_conductivity(
            self.net, self.solid
        )
        self.solid["throat.thermal_fluid_conductivity"] = throat_thermal_fluid_conductivity(self.net)
        self.solid["throat.thermal_conductivity"] = throat_thermal_conductivity(self.net)
        self.net["throat.diffusive_size_factors"] = diffusive_size_factors(self.net)

        self._validate_fixture()
        self.model_specs = self._build_model_specs()

    def _validate_fixture(self):
        """
        Sanity checks to ensure all generated values are physically sound.
        """
        assert np.all(self.net["pore.diameter"] > 0.0)
        assert np.all(self.net["throat.effective_radius"] > 0.0)
        assert np.all(self.net["throat.length"] > 0.0)
        assert np.all(self.net["throat.relative_bridge_radius"] > 0.0)
        assert np.all(self.net["throat.relative_bridge_radius"] < 1.0)
        assert np.all(self.net["throat.relative_contact_throat_radius"] > 0.0)
        assert np.all(
            self.net["throat.relative_contact_throat_radius"]
            < self.net["throat.relative_bridge_radius"]
        )
        assert np.all(self.net["throat.relative_effective_radius"] > 0.0)
        assert np.all(self.net["throat.solid_volume"] > 0.0)
        assert np.all(self.net["throat.fluid_volume"] > 0.0)

        assert np.any(self.net["pore.boundary"])
        assert np.any(self.net["throat.boundary"])
        assert np.any(~self.net["pore.boundary"])
        assert np.any(~self.net["throat.boundary"])

        assert np.all(self.solid["pore.thermal_conductivity"] > 0.0)
        assert np.all(self.solid["throat.thermal_solid_conductivity"] > 0.0)
        assert np.all(self.solid["throat.thermal_fluid_conductivity"] > 0.0)
        assert np.all(self.solid["throat.thermal_conductivity"] > 0.0)

        # IMPORTANT: size factors belong on the network for generic_thermal and series_resistors
        assert np.all(self.net["throat.diffusive_size_factors"] > 0.0)

    def _copy_with_alias(self, model_name, kw_name, propname):
        """
        Create a model-specific alias property so every input argument is
        explicitly exercised for each model.

        Important:
        Some properties must live on the network (e.g. throat.diffusive_size_factors
        for generic_thermal / series_resistors), while others belong on the phase.
        """
        alias = f"{propname}__{model_name}__{kw_name}"

        # Properties that must be stored on the network
        network_props = {
            "pore.diameter",
            "pore.boundary",
            "throat.boundary",
            "throat.effective_radius",
            "throat.length",
            "throat.relative_bridge_radius",
            "throat.relative_contact_throat_radius",
            "throat.relative_effective_radius",
            "throat.solid_volume",
            "throat.fluid_volume",
            "throat.diffusive_size_factors",
        }

        if propname in network_props:
            if propname not in self.net:
                raise KeyError(
                    f"Expected network property '{propname}' for model='{model_name}', "
                    f"kwarg='{kw_name}', but it was not found on the network."
                )
            self.net[alias] = np.array(self.net[propname], copy=True)
            return alias

        # Otherwise assume phase property
        if propname in self.solid:
            self.solid[alias] = np.array(self.solid[propname], copy=True)
            return alias

        # Fallback: if user placed it on network, still allow it
        if propname in self.net:
            self.net[alias] = np.array(self.net[propname], copy=True)
            return alias

        raise KeyError(
            f"Property '{propname}' not found while preparing alias for "
            f"model='{model_name}', kwarg='{kw_name}'"
        )

    # ---Main chapter---
    def _alias_kwargs(self, model_name, **kwargs):
        out = {}
        for kw_name, value in kwargs.items():
            if isinstance(value, str):
                out[kw_name] = self._copy_with_alias(model_name, kw_name, value)
            else:
                out[kw_name] = value
        return out

    # ---Main chapter---
    def _build_model_specs(self):
        tc = op.models.physics.thermal_conductance
        return {
            "generic_thermal": {
                "model": tc.generic_thermal,
                "kwargs": self._alias_kwargs(
                    "generic_thermal",
                    pore_conductivity="pore.thermal_conductivity",
                    throat_conductivity="throat.thermal_conductivity",
                    size_factors="throat.diffusive_size_factors",
                ),
            },
            "series_resistors": {
                "model": tc.series_resistors,
                "kwargs": self._alias_kwargs(
                    "series_resistors",
                    pore_thermal_conductivity="pore.thermal_conductivity",
                    throat_thermal_conductivity="throat.thermal_conductivity",
                    size_factors="throat.diffusive_size_factors",
                ),
            },
            "yovanovich": {
                "model": tc.yovanovich,
                "kwargs": self._alias_kwargs(
                    "yovanovich",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    effective_radius="throat.effective_radius",
                    relative_gas_conductivity_radius=1e12,
                    radiation_exchange_factor=0.0,
                ),
            },
            "dixon_bridge_model": {
                "model": tc.dixon_bridge_model,
                "kwargs": self._alias_kwargs(
                    "dixon_bridge_model",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    relative_bridge_radius="throat.relative_bridge_radius",
                    diameter="pore.diameter",
                    throat_length="throat.length",
                ),
            },
            "extended_dixon_bridge_model": {
                "model": tc.extended_dixon_bridge_model,
                "kwargs": self._alias_kwargs(
                    "extended_dixon_bridge_model",
                    pore_solid_conductivity="pore.thermal_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    relative_bridge_radius="throat.relative_bridge_radius",
                    diameter="pore.diameter",
                    throat_length="throat.length",
                ),
            },
            "batchelor": {
                "model": tc.batchelor,
                "kwargs": self._alias_kwargs(
                    "batchelor",
                    pore_thermal_conductivity="pore.thermal_conductivity",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    effective_radius="throat.effective_radius",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    throat_length="throat.length",
                    effective_radius_fraction="throat.relative_effective_radius",
                ),
            },
            "kunii_smith": {
                "model": tc.kunii_smith,
                "kwargs": self._alias_kwargs(
                    "kunii_smith",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    effective_radius="throat.effective_radius",
                ),
            },
            "tsotsas_bob": {
                "model": tc.tsotsas_bob,
                "kwargs": self._alias_kwargs(
                    "tsotsas_bob",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    effective_radius="throat.effective_radius",
                ),
            },
            "argento": {
                "model": tc.argento,
                "kwargs": self._alias_kwargs(
                    "argento",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    effective_radius="throat.effective_radius",
                    throat_lenght="throat.length",
                ),
            },
            "fei_narsilio": {
                "model": tc.fei_narsilio,
                "kwargs": self._alias_kwargs(
                    "fei_narsilio",
                    pore_thermal_conductivity="pore.thermal_conductivity",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    relative_bridge_radius="throat.relative_bridge_radius",
                    effective_radius="throat.effective_radius",
                    boundary_throats="throat.boundary",
                    diameter="pore.diameter",
                ),
            },
            "birkholz": {
                "model": tc.birkholz,
                "kwargs": self._alias_kwargs(
                    "birkholz",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    effective_radius="throat.effective_radius",
                    pore_thermal_conductivity="pore.thermal_conductivity",
                ),
            },
            "zehner_bauer_schlunder": {
                "model": tc.zehner_bauer_schlunder,
                "kwargs": self._alias_kwargs(
                    "zehner_bauer_schlunder",
                    solid_volume="throat.solid_volume",
                    fluid_volume="throat.fluid_volume",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    diameter="pore.diameter",
                    return_conductance=True,
                ),
            },
            "tsotsas_zbs": {
                "model": tc.tsotsas_zbs,
                "kwargs": self._alias_kwargs(
                    "tsotsas_zbs",
                    solid_volume="throat.solid_volume",
                    fluid_volume="throat.fluid_volume",
                    throat_solid_conductivity="throat.thermal_solid_conductivity",
                    throat_fluid_conductivity="throat.thermal_fluid_conductivity",
                    relative_contact_radius="throat.relative_contact_throat_radius",
                    diameter="pore.diameter",
                    effective_radius="throat.effective_radius",
                ),
            },
            "bahrami_rough_joint": {
                "model": tc.bahrami_rough_joint,
                "kwargs": self._alias_kwargs(
                    "bahrami_rough_joint",
                    pore_solid_conductivity="pore.thermal_conductivity",
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
                ),
            },
        }

    # TODO: Fully implement bahrami golden test.

    # ---Main chapter---
    def _run_model(self, model_name):
        spec = self.model_specs[model_name]
        propname = f"throat._golden_master_{model_name}"

        self.solid.add_model(
            propname=propname,
            model=spec["model"],
            **spec["kwargs"],
        )
        self.solid.regenerate_models(propnames=[propname])

        values = np.asarray(self.solid[propname], dtype=float)
        if values.ndim != 1:
            raise AssertionError(
                f"Model '{model_name}' returned shape {values.shape}, expected 1D throat array"
            )
        return values

    # ---Main chapter---
    def collect_results(self):
        return {model_name: self._run_model(model_name) for model_name in self.model_specs}

    # ---Main chapter---
    def write_baseline(self, path=None):
        if path is None:
            path = self.BASELINE_FILE

        results = self.collect_results()
        np.savez_compressed(path, **results)

        print(f"Golden master written to: {path}")
        for name, values in results.items():
            print(
                f"{name:28s} "
                f"mean={values.mean():.16e} "
                f"min={values.min():.16e} "
                f"max={values.max():.16e}"
            )

    # ---Main chapter---
    def assert_model_matches_baseline(self, model_name, path=None, rtol=1e-12, atol=1e-15):
        if path is None:
            path = self.BASELINE_FILE

        if not path.exists():
            raise FileNotFoundError(
                f"Golden master file not found: {path}\n"
                f"Create it first with:\n"
                f"    python {Path(__file__).name} --write-baseline"
            )

        actual = self._run_model(model_name)

        with np.load(path, allow_pickle=False) as baseline:
            if model_name not in baseline.files:
                raise AssertionError(
                    f"Model '{model_name}' not found in baseline. "
                    f"Available keys: {sorted(baseline.files)}"
                )
            expected = baseline[model_name]

        assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"Golden master mismatch in model '{model_name}'",
        )

    # ---Main chapter---
    def test_generic_thermal(self):
        self.assert_model_matches_baseline("generic_thermal")

    def test_series_resistors(self):
        self.assert_model_matches_baseline("series_resistors")

    def test_yovanovich(self):
        self.assert_model_matches_baseline("yovanovich")

    def test_dixon_bridge_model(self):
        self.assert_model_matches_baseline("dixon_bridge_model")

    def test_extended_dixon_bridge_model(self):
        self.assert_model_matches_baseline("extended_dixon_bridge_model")

    def test_batchelor(self):
        self.assert_model_matches_baseline("batchelor")

    def test_kunii_smith(self):
        self.assert_model_matches_baseline("kunii_smith")

    def test_tsotsas_bob(self):
        self.assert_model_matches_baseline("tsotsas_bob")

    def test_argento(self):
        self.assert_model_matches_baseline("argento")

    def test_fei_narsilio(self):
        self.assert_model_matches_baseline("fei_narsilio")

    def test_birkholz(self):
        self.assert_model_matches_baseline("birkholz")

    def test_zehner_bauer_schlunder(self):
        self.assert_model_matches_baseline("zehner_bauer_schlunder")

    def test_tsotsas_zbs(self):
        self.assert_model_matches_baseline("tsotsas_zbs")


# ---Main chapter---
def main():
    parser = argparse.ArgumentParser(
        description="Golden master tests for OpenPNM thermal conductance models"
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current model outputs to the golden master baseline",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=TestThermalConductanceGoldenMaster.BASELINE_FILE,
        help="Path to .npz golden master baseline file",
    )
    parser.add_argument("--rtol", type=float, default=1e-12, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-15, help="Absolute tolerance")
    args = parser.parse_args()

    t = TestThermalConductanceGoldenMaster()
    t.setup_class()
    t.BASELINE_FILE = args.baseline

    if args.write_baseline:
        t.write_baseline(path=args.baseline)
        return

    for item in dir(t):
        if item.startswith("test_"):
            print(f"Running test: {item}")
            getattr(t, item)()


if __name__ == "__main__":
    main()
