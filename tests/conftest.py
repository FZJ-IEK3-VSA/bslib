import pytest
import bslib.bslib as bsl


@pytest.fixture(scope="session")
def ac_bat():
    yield bsl.ACBatMod('S2')
