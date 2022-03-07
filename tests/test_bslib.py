import pytest
import src.bslib as bsl

# ac_bat object veralgemeinern Werte selber setzen und nicht vom import abhängig machen

def test_acbatmod_avoid_overcharging_charging():
    ac_bat = bsl.ACBatMod('S2')
    ac_bat._E_BAT = 1.0
    assert ac_bat.avoid_overcharging(p_bs=3600, e_b0=1, dt=1) == 0.0

def test_acbatmod_avoid_overcharging_discharging():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.avoid_overcharging(p_bs=-2*3600, e_b0=1.0, dt=1) == -3240.0

def test_acbatmod_adjust_to_stationary_deviations():
    ac_bat = bsl.ACBatMod('S2')
    ac_bat._P_AC2BAT_MIN = 0
    ac_bat._P_BAT2AC_MIN = 0
    assert ac_bat.adjust_to_stationary_deviations(p_bs=1) == 0
    assert ac_bat.adjust_to_stationary_deviations(p_bs=-1) == 0


def test_acbatmod_get_battery_state():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.get_battery_state(1.0, 0.0) == bsl.BatteryState.CHARGING
    assert ac_bat.get_battery_state(-1.0, 1.0) == bsl.BatteryState.DISCHARGING
    assert ac_bat.get_battery_state(0.0, 0.0) is None

def test_acbatmod_charge_battery():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.charge_battery(p_bs=1500.0) == 1430.458346868159


def test_acbatmod_discharge_battery():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.discharge_battery(p_bs=-1500.0) == -1574.7343399810065

def test_acbatmod_standy_mode():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.standby_mode(soc=0) == (12.1, 0)
    assert ac_bat.standby_mode(soc=1) == (14.9, -0.1)

def test_acbatmod_change_battery_content():
    ac_bat = bsl.ACBatMod('S2')
    assert ac_bat.change_battery_content(p_bat=1.0, e_b0=1.0, dt=1) == 1.0002733959955272
    assert ac_bat.change_battery_content(p_bat=-1.0, e_b0=1.0, dt=1) == 0.9997177702121118
    assert ac_bat.change_battery_content(p_bat=0.0, e_b0=1.0, dt=1) == 1.0

def test_acbatmod_simulate():
    ac_bat = bsl.ACBatMod('S2')
    result = ac_bat.simulate(p_load=1500.0, soc=0.5, dt=1)
    assert result.p_bs == 1497.5
    assert result.p_bat == 1428.0461947033716
    assert result.soc == 0.5000434284884382

    result = ac_bat.simulate(p_load=-1500.0, soc=0.5, dt=1)
    assert result.p_bs == -1499.9
    assert result.p_bat == -1574.630013291337
    assert result.soc == 0.49995056646333114


def test_dc_batmod():
    dc_battery = bsl.DCBatMod("S3")
    result = dc_battery.simulation(p_load=500, p_pv=350, soc=0, dt=1)
    assert result.p_pvbs == 312.78314769227916
    assert result.p_bat == 0.0
    assert result.soc == 0.0