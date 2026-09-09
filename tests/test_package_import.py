import pytest

pytestmark = pytest.mark.no_backend

import dd_nm_rom


def test_package_import_exposes_common_modules():
  assert isinstance(dd_nm_rom.__version__, str)
  assert dd_nm_rom.config.__name__ == "dd_nm_rom.config"
  assert dd_nm_rom.env.__name__ == "dd_nm_rom.env"
