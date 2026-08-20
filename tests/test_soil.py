import pytest

from core.models import SoilLayer, SoilProfile, SoilType


def test_stresses_with_gwl():
    profile = SoilProfile(
        layers=[
            SoilLayer(
                name="As",
                soil_type=SoilType.SAND,
                thickness=10.0,
                n_value=10,
                gamma_t=18.0,
                gamma_sat=19.0,
            )
        ],
        gwl=1.0,
    )
    sigma_v, sigma_v_eff = profile.stresses_at(5.0)
    # σv = 1×18 + 4×19 = 94, u = 4×9.8 = 39.2
    assert sigma_v == pytest.approx(94.0)
    assert sigma_v_eff == pytest.approx(54.8)


def test_stresses_multilayer():
    profile = SoilProfile(
        layers=[
            SoilLayer(
                soil_type=SoilType.CLAY,
                thickness=3.0,
                n_value=2,
                gamma_t=16.0,
                gamma_sat=16.0,
            ),
            SoilLayer(
                soil_type=SoilType.SAND,
                thickness=7.0,
                n_value=15,
                gamma_t=18.0,
                gamma_sat=19.0,
            ),
        ],
        gwl=3.0,
    )
    sigma_v, sigma_v_eff = profile.stresses_at(8.0)
    # σv = 3×16 + 5×19 = 143, u = 5×9.8 = 49
    assert sigma_v == pytest.approx(143.0)
    assert sigma_v_eff == pytest.approx(94.0)


def test_stress_out_of_range():
    profile = SoilProfile(
        layers=[
            SoilLayer(
                soil_type=SoilType.SAND,
                thickness=5.0,
                n_value=10,
                gamma_t=18.0,
                gamma_sat=19.0,
            )
        ],
        gwl=1.0,
    )
    with pytest.raises(ValueError):
        profile.stresses_at(6.0)
