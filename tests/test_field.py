import gc
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from dd_nm_rom import backend as bkd
from dd_nm_rom import ops
from dd_nm_rom.field import (
  Burgers2DExact,
  ElasticityForce,
  MultiPeak,
  MultiPeakGen,
  MultiPeakGen2,
  MultiPeakSgn,
  PoissonForce,
  SinMultiPeak,
  SinPeak,
)


FIELD_CASES = (
  (SinMultiPeak, 16),
  (MultiPeak, 32),
  (MultiPeakSgn, 32),
  (MultiPeakGen, 32),
  (MultiPeakGen2, 32),
  (PoissonForce, 18),
  (ElasticityForce, 34),
)

ALL_FIELD_CASES = FIELD_CASES + ((Burgers2DExact, 2),)
ALL_FIELD_TYPES = (
  Burgers2DExact,
  SinMultiPeak,
  SinPeak,
  MultiPeak,
  MultiPeakSgn,
  MultiPeakGen,
  MultiPeakGen2,
  PoissonForce,
  ElasticityForce,
)


def _field(field_type, n_sub, use_qmc=True):
  return field_type(SimpleNamespace(n_sub=n_sub), use_qmc=use_qmc)


@pytest.mark.no_backend
@pytest.mark.parametrize("field_type", ALL_FIELD_TYPES)
def test_every_field_defaults_to_qmc(field_type):
  assert inspect.signature(field_type).parameters["use_qmc"].default is True
  assert _field(field_type, n_sub=4).use_qmc is True


@pytest.mark.no_backend
@pytest.mark.parametrize("field_type, n_parameters", ALL_FIELD_CASES)
def test_qmc_is_the_default_and_preserves_parameter_shape(
  field_type, n_parameters, monkeypatch
):
  def fail_if_called(*args, **kwargs):
    raise AssertionError("the default path constructed all configurations")

  monkeypatch.setattr(ops, "generate_combs", fail_if_called)
  field = _field(field_type, n_sub=16)

  assert field.use_qmc is True
  samples = field.construct_design_mat(n_samples=3)
  assert samples.shape == (3, n_parameters)


@pytest.mark.no_backend
@pytest.mark.parametrize("field_type, _", FIELD_CASES)
def test_legacy_and_qmc_paths_return_compatible_samples(field_type, _):
  qmc_samples = _field(field_type, n_sub=4).construct_design_mat(5)
  legacy_samples = _field(
    field_type, n_sub=4, use_qmc=False
  ).construct_design_mat(5)

  assert qmc_samples.shape == legacy_samples.shape
  assert np.all(np.isfinite(qmc_samples))
  assert np.all(np.isfinite(legacy_samples))


def _peak_allocation(field_type, use_qmc):
  # n_sub=16 is large enough to expose the old 2**n configuration allocation,
  # while remaining small enough for a normal unit-test process.
  gc.collect()
  field = _field(field_type, n_sub=16, use_qmc=use_qmc)
  start = bkd._start_mem_trace(reset=True)
  field.construct_design_mat(n_samples=2)
  end = bkd._end_mem_trace(print_stats=False)
  return bkd._get_mem_trace_stats(start, end)[1]


@pytest.mark.no_backend
@pytest.mark.parametrize("field_type, _", FIELD_CASES)
def test_default_qmc_avoids_configuration_memory_allocation(field_type, _):
  qmc_peak = _peak_allocation(field_type, use_qmc=True)
  legacy_peak = _peak_allocation(field_type, use_qmc=False)

  # The legacy path materializes the 2**n binary configuration table.  Keep
  # this as a relative assertion so the test is independent of the allocator.
  assert legacy_peak > 1.0
  assert qmc_peak < legacy_peak
