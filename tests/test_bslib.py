import pytest
import src.bslib as bsl


def test_ac_batmod():
    ac_bat = bsl.ACBatMod('S2')
    result = ac_bat.simulate(p_load=500, soc=0.0, dt=1)
    assert result.p_bat == 454.698031908842
    assert result.p_bs == 497.5
    assert result.soc == 1.382787776396303e-05


def test_dc_batmod():
    dc_battery = bsl.DCBatMod("S3")
    result = dc_battery.simulation(p_load=500, p_pv=350, soc=0, dt=1)
    assert result.p_pvbs == 312.78314769227916
    assert result.p_bat == 0.0
    assert result.soc == 0.0