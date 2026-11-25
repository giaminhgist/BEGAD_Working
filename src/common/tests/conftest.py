import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests (TabPFN fits); deselect with -m 'not slow'")
