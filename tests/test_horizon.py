import pandas as pd

from jfunction.horizon import assign_horizon_by_depth


def test_assign_horizon_by_depth_matches_correct_interval():
    horizon_map = pd.DataFrame(
        {
            "well": ["300", "300", "301"],
            "depth_from": [300.0, 700.0, 500.0],
            "depth_to": [600.0, 1000.0, 600.0],
            "horizon": ["Мел", "Юра", "Юра"],
        }
    )
    df = pd.DataFrame(
        {
            "well": ["300", "300", "301", "300"],
            "depth": [400.0, 800.0, 550.0, 2000.0],  # последняя точка вне всех интервалов
        }
    )

    out = assign_horizon_by_depth(df, horizon_map)

    assert list(out["horizon"][:3]) == ["Мел", "Юра", "Юра"]
    assert pd.isna(out["horizon"].iloc[3])
