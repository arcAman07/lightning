from tests_lite.helpers.runif import RunIf

from lightning_lite.lite import LightningLite  # noqa: F401


def test_placeholder(tmpdir):
    assert True


@RunIf(min_cuda_gpus=2, standalone=True)
def test_placeholder_standalone(tmpdir):
    assert True
