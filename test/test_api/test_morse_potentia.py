"""Unit test for Morse potential specific API."""
from __future__ import annotations

import pytest

from epseon_nn.api.morse_potential import estimate_morse_potential_params

LEVEL_DATA_0 = [
    -5446.902056983451,
    -5366.912545092755,
    -5288.884129967908,
    -5208.871125278474,
    -5126.403283600707,
    -5042.451461168255,
    -4957.855727174555,
    -4873.409012269292,
    -4789.586442312533,
    -4706.659727697996,
    -4624.714770964255,
    -4543.546871648250,
    -4463.002768586786,
    -4383.030433208333,
    -4303.564644428669,
    -4224.597730762711,
    -4146.176852761439,
    -4068.340652394370,
    -3991.125383758283,
    -3914.579704253993,
    -3838.689804772547,
    -3763.433669739079,
    -3688.795533906819,
    -3614.798955355019,
    -3541.459932482626,
    -3468.793136156479,
    -3396.799710735992,
    -3325.487827290825,
    -3254.878786240879,
    -3185.002480609012,
    -3115.880869705394,
    -3047.520959166172,
    -2979.929835951191,
    -2913.121763009104,
    -2847.122386236198,
    -2781.951652999776,
    -2717.622551259853,
    -2654.143829755050,
    -2591.526078444105,
    -2529.787713687999,
    -2468.949011894107,
]

NULL = 0.0


@pytest.mark.parametrize(
    "level",
    [
        LEVEL_DATA_0,
        [*([0.0] * 10), *LEVEL_DATA_0[10:]],
        [*LEVEL_DATA_0[:-10], *([0.0] * 10)],
    ],
    ids=lambda e: f"Case {len(e)}: {e}",
)
def test_morse_potential_estimation_from_n_values(level: list[float]) -> None:
    """Test quality of morse potential estimation for different input level energies counts."""
    params = estimate_morse_potential_params(level)
    assert params.dissociation_energy != NULL
    assert params.equilibrium_bond_distance != NULL
    assert params.alpha != NULL
