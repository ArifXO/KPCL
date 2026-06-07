"""pytest configuration: slow-test marker and --slow flag."""
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run slow integration tests (require network download)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires network; skipped by default)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="pass --slow to run integration tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
