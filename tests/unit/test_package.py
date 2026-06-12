import mltrade


def test_package_exposes_version() -> None:
    assert mltrade.__version__ == "0.1.0"
