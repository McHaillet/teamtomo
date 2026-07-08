import warnings

import pytest
import torch

warnings.filterwarnings(action="ignore", category=UserWarning, module="tiler")
from torch_segment_fiducials_2d.model import ResidualUNet18


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    """Pin torch to a single CPU thread for this module.

    CI runners hit pathologically slow multi-threaded CPU convolutions here
    (test_model.py went from ~5s locally to ~105-176s in CI); single-threaded
    execution avoids that without affecting other test modules.
    """
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_model_instantiation():
    """Instantiation test."""
    model = ResidualUNet18()
    assert isinstance(model, ResidualUNet18)


def test_model_call():
    """Model should return 2-class logits of same shape."""
    model = ResidualUNet18()
    image = torch.rand(size=(2, 1, 128, 128))
    out = model(image)
    assert out.shape == (2, 2, 128, 128)


def test_model_predict_step():
    """Test auto tiled prediction of larger single images.

    model.predict_step() should yield probabilities of same shape for class 1.
    """
    model = ResidualUNet18(batch_size=1)
    image = torch.rand(size=(512, 512))
    out = model.predict_step(image)
    assert out.shape == image.shape
    assert torch.min(out) >= 0
    assert torch.max(out) <= 1
