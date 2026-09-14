import pandas as pd

from jfunction.rocktype import classify_by_permeability


def test_classify_by_permeability_bins():
    perm = pd.Series([0.5, 1.0, 5.0, 10.0, 50.0, 200.0])
    labels = classify_by_permeability(perm, [1, 10, 100])

    assert list(labels.astype(str)) == ["< 1", "1-10", "1-10", "10-100", "10-100", ">= 100"]
