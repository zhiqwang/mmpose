import pytest


def test_old_fashion_registry_importing():
    with pytest.warns(DeprecationWarning):
        from mmpose.models.registry import (
            BACKBONES,
            HEADS,  # noqa: F401
            LOSSES,
            NECKS,
            POSENETS)
    with pytest.warns(DeprecationWarning):
        from mmpose.datasets.registry import DATASETS, PIPELINES  # noqa: F401
