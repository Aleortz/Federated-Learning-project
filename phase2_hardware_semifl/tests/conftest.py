import pytest

from common import data


@pytest.fixture(scope="session")
def ds():
    return data.cargar()


@pytest.fixture
def cfg_rapida(tmp_path):
    """Configuracion pequena y rapida para tests de integracion."""
    from common import fl_config
    return fl_config.cargar(rounds=4, round_timeout=5, results_dir=str(tmp_path / "results"),
                            connect_timeout=30, partition_timeout=30, log_level="WARNING")
