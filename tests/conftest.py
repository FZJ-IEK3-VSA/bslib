import pytest
import src.bslib as bsl


@pytest.fixture(scope="session")
def ac_bat():
    yield bsl.ACBatMod('S2')
