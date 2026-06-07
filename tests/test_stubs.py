"""
Stage 0 smoke test: verify all stub modules import without error.

Real unit tests (R3: pos/neg/FN cases, R7: dict keys) are added per stage.
"""


def test_data_imports():
    from src.data.loader import DataSplit, load_dataset  # noqa: F401


def test_model_imports():
    from src.models.encoder import KANEncoder, MLPEncoder  # noqa: F401
    from src.models.heads import KANHead, MLPHead  # noqa: F401


def test_loss_imports():
    from src.losses.infonce import info_nce_loss  # noqa: F401
    from src.losses.kpcl import kpcl_loss  # noqa: F401
    from src.losses.kurc import kurc_loss  # noqa: F401


def test_metrics_imports():
    from src.metrics.auroc import macro_auroc, fn_ranking_auc  # noqa: F401


def test_train_imports():
    import src.train  # noqa: F401
