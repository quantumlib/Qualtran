#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from contextlib import contextmanager
from unittest import mock

import pytest

from qualtran.conftest import get_available_cpu_count, pytest_configure


@contextmanager
def temporary_delete_attr(obj, attr):
    """Context manager to temporarily delete an attribute from an object.

    This is useful for mocking Python environment detection logic where we want
    to simulate standard library features or system-level attributes being absent.
    """
    has_attr = hasattr(obj, attr)
    if has_attr:
        val = getattr(obj, attr)
        delattr(obj, attr)
        try:
            yield
        finally:
            setattr(obj, attr, val)
    else:
        try:
            yield
        finally:
            pass


@contextmanager
def temporary_set_attr(obj, attr, val):
    """Context manager to temporarily set or override an attribute on an object.

    This allows us to simulate the presence of standard library features or OS-level
    functions that might not exist on the current host system running the tests.
    """
    has_attr = hasattr(obj, attr)
    if has_attr:
        old_val = getattr(obj, attr)
        setattr(obj, attr, val)
        try:
            yield
        finally:
            setattr(obj, attr, old_val)
    else:
        setattr(obj, attr, val)
        try:
            yield
        finally:
            delattr(obj, attr)


def test_get_available_cpu_count_process_cpu_count():
    # Test when process_cpu_count returns a valid count.
    mock_func = mock.Mock(return_value=4)
    with temporary_set_attr(os, 'process_cpu_count', mock_func):
        assert get_available_cpu_count() == 4

    # Test when process_cpu_count returns 0 or None (forces fallback to or 1).
    mock_func = mock.Mock(return_value=0)
    with temporary_set_attr(os, 'process_cpu_count', mock_func):
        assert get_available_cpu_count() == 1


def test_get_available_cpu_count_sched_getaffinity():
    # Test when process_cpu_count is not available, but sched_getaffinity is.
    with temporary_delete_attr(os, 'process_cpu_count'):
        mock_affinity = mock.Mock(return_value={0, 1})
        with temporary_set_attr(os, 'sched_getaffinity', mock_affinity):
            assert get_available_cpu_count() == 2


def test_get_available_cpu_count_sched_getaffinity_os_error():
    # Test when process_cpu_count is not available, and sched_getaffinity raises OSError.
    with temporary_delete_attr(os, 'process_cpu_count'):
        mock_affinity = mock.Mock(side_effect=OSError)
        with temporary_set_attr(os, 'sched_getaffinity', mock_affinity):
            with mock.patch('os.cpu_count', return_value=8):
                assert get_available_cpu_count() == 8

            with mock.patch('os.cpu_count', return_value=0):
                assert get_available_cpu_count() == 1


def test_get_available_cpu_count_fallback():
    # Test fallback path when neither process_cpu_count nor sched_getaffinity is available.
    with (
        temporary_delete_attr(os, 'process_cpu_count'),
        temporary_delete_attr(os, 'sched_getaffinity'),
    ):
        with mock.patch('os.cpu_count', return_value=16):
            assert get_available_cpu_count() == 16

        with mock.patch('os.cpu_count', return_value=None):
            assert get_available_cpu_count() == 1


def test_pytest_configure_worker_early_return():
    config = mock.MagicMock(spec=pytest.Config)
    config.workerinput = {}
    with mock.patch('qualtran.conftest.get_available_cpu_count') as mock_get_cpus:
        pytest_configure(config)
        mock_get_cpus.assert_not_called()


def test_pytest_configure_getoption_value_error():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput  # Ensure hasattr returns False
    config.getoption.side_effect = ValueError('numprocesses is not configured')

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=4),
    ):
        pytest_configure(config)
        # Value error sets num_workers to "1".
        # When num_workers <= 1, env variables are not set.
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert var not in os.environ


def test_pytest_configure_auto_workers():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.return_value = 'auto'

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=8),
    ):
        pytest_configure(config)
        # num_workers is 8, limit is max(1, 8 // 8) = 1
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert os.environ[var] == '1'


def test_pytest_configure_invalid_workers():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.return_value = 'invalid'

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=4),
    ):
        pytest_configure(config)
        # raises ValueError in int(), num_workers is set to 1.
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert var not in os.environ


def test_pytest_configure_type_error_workers():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.return_value = []  # List raises TypeError in int()

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=4),
    ):
        pytest_configure(config)
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert var not in os.environ


def test_pytest_configure_set_env_vars():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.return_value = '4'

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=12),
    ):
        pytest_configure(config)
        # num_workers = 4, num_cpus = 12, limit = max(1, 12 // 4) = 3
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert os.environ[var] == '3'


def test_pytest_configure_cpus_non_positive():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.return_value = '4'

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch('qualtran.conftest.get_available_cpu_count', return_value=0),
    ):
        pytest_configure(config)
        # num_cpus <= 0, so thread limit is not set.
        env_vars = [
            'MKL_NUM_THREADS',
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
        ]
        for var in env_vars:
            assert var not in os.environ


def test_pytest_configure_sets_dist_worksteal():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.side_effect = lambda name: '4' if name == 'numprocesses' else 'load'
    config.option = mock.MagicMock()
    config.option.dist = 'load'
    config.invocation_params = mock.MagicMock()
    config.invocation_params.args = ['-n', '4']

    with mock.patch('qualtran.conftest.get_available_cpu_count', return_value=8):
        pytest_configure(config)
        assert config.option.dist == 'worksteal'


def test_pytest_configure_preserves_user_dist():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.side_effect = lambda name: '4' if name == 'numprocesses' else 'loadscope'
    config.option = mock.MagicMock()
    config.option.dist = 'loadscope'
    config.invocation_params = mock.MagicMock()
    config.invocation_params.args = ['-n', '4', '--dist', 'loadscope']

    with mock.patch('qualtran.conftest.get_available_cpu_count', return_value=8):
        pytest_configure(config)
        assert config.option.dist == 'loadscope'


def test_pytest_configure_preserves_user_dist_equals():
    config = mock.MagicMock(spec=pytest.Config)
    del config.workerinput
    config.getoption.side_effect = lambda name: '4' if name == 'numprocesses' else 'each'
    config.option = mock.MagicMock()
    config.option.dist = 'each'
    config.invocation_params = mock.MagicMock()
    config.invocation_params.args = ['-n', '4', '--dist=each']

    with mock.patch('qualtran.conftest.get_available_cpu_count', return_value=8):
        pytest_configure(config)
        assert config.option.dist == 'each'
