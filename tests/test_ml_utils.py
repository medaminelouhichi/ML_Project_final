"""Tests unitaires de src/ml_utils.py (données synthétiques minuscules, structure Arrange-Act-Assert)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import ml_utils as U

NUM, CAT = ["a", "b"], ["c"]
ORDER = ["none", "class_weight", "ros", "smote"]


@pytest.fixture
def raw_flows() -> pd.DataFrame:
    return pd.DataFrame({
        "src_ip": ["192.168.1.5", "10.0.0.2"], "dst_ip": ["192.168.1.9", "8.8.8.8"],
        "src_port": [50000, 80], "dst_port": [80, 80],
        "src_bytes": [100, 0], "dst_bytes": [50, 10], "src_pkts": [3, 1], "dst_pkts": [2, 1],
        "src_ip_bytes": [220, 40], "dst_ip_bytes": [150, 40], "duration": [1.5, 0.0],
        "dns_query": ["x.com", np.nan], "http_method": [np.nan, "GET"],
        "ssl_version": [np.nan, np.nan], "weird_name": [np.nan, np.nan],
    })


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(0)
    n = 240
    y = np.repeat([0, 1, 2], [150, 70, 20])
    X = pd.DataFrame({"a": rng.normal(y, 1.0), "b": rng.exponential(1 + y), "c": rng.choice(["u", "v", "w"], n)})
    X.loc[::17, "a"] = np.nan
    return X, pd.Series(y)


def test_engineer_features_has_no_ip_derived_flags(raw_flows):
    out = U.engineer_features(raw_flows)
    assert "src_internal" not in out.columns and "dst_internal" not in out.columns
    assert out["has_dns"].tolist() == [1, 0] and out["has_http"].tolist() == [0, 1]
    assert out["src_port_bucket"].tolist() == ["ephemeral", "well_known"]
    assert out["total_bytes"].tolist() == [150, 10]


def test_leaky_columns_documented():
    assert {"src_ip", "dst_ip", "src_port"} <= set(U.LEAKY_OR_ID_COLUMNS)


def test_type_maps_and_order_roundtrip():
    order = U.get_type_order({"type_order": ["b", "a", "c"]})
    to_code, to_type = U.type_maps(order)
    assert to_code == {"b": 0, "a": 1, "c": 2} and to_type[1] == "a"


def test_validate_raw_flags_invalid_values():
    df = pd.DataFrame({"src_bytes": [1, -5], "src_port": [10, 70000], "label": [0, 1], "type": ["normal", "normal"]})
    rep = U.validate_raw(df).set_index("contrôle")["n_invalides"]
    assert rep["src_bytes < 0"] == 1 and rep["src_port hors [0, 65535]"] == 1
    assert rep["label != (type != 'normal')"] == 1


def test_binary_metrics_specificity_and_macro_f1():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    m = U.binary_metrics(y, pred, scores=pred.astype(float), proba=pred.astype(float))
    assert m["specificity"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["f1_macro"] == pytest.approx(0.75) and m["f1_weighted"] == pytest.approx(0.75)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert {"roc_auc", "pr_auc", "brier", "fpr", "fnr"} <= set(m)


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    thr, val = U.best_threshold(y, s)
    assert 0.4 < thr <= 0.6 and val == pytest.approx(1.0)


def test_multiclass_metrics_and_per_class_report():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    m = U.multiclass_metrics(y, pred, labels=[0, 1, 2])
    rep = U.per_class_report(y, pred, [0, 1, 2], ["x", "y", "z"])
    assert m["accuracy"] == pytest.approx(4 / 6) and 0 < m["f1_macro"] < 1
    assert rep.loc["y", "recall"] == 1.0 and rep["support"].tolist() == [2, 2, 2]


def test_regression_metrics_only_continuous_scores():
    m = U.regression_metrics([0, 1, 2], [0.0, 1.0, 3.0])
    assert set(m) == {"mae", "rmse", "r2"} and m["mae"] == pytest.approx(1 / 3)


def test_decode_with_cuts_and_round_and_within_k():
    raw = np.array([-0.4, 0.6, 1.4, 9.0])
    assert U.decode_round(raw, 3).tolist() == [0, 1, 1, 2]
    assert U.decode_with_cuts(raw, [0.5, 1.5]).tolist() == [0, 1, 1, 2]
    assert U.within_k([0, 1, 2], [1, 1, 0], k=1) == pytest.approx(2 / 3)


def test_optimize_cutpoints_not_worse_than_rounding_and_increasing():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 600)
    raw = y * 0.8 + rng.normal(0, 0.25, 600)  # sortie biaisée : l'arrondi se trompe
    base = U._macro_f1_fast(y, U.decode_round(raw, 4), 4)
    cuts, best = U.optimize_cutpoints(raw, y, 4)
    assert best >= base and np.all(np.diff(cuts) > 0)
    assert U._macro_f1_fast(y, U.decode_with_cuts(raw, cuts), 4) == pytest.approx(best)


def test_hash_overlap_and_drop_rows_present_in():
    ref = pd.DataFrame({"x": [1.0, 2.0, np.nan], "k": ["a", "b", None]})
    other = pd.DataFrame({"x": [2.0, 9.0, np.nan, 7.0], "k": ["b", "z", None, "a"]})
    y = pd.Series([0, 1, 2, 3])
    assert U.cross_split_overlap(ref, other) == pytest.approx(0.5)
    X2, y2, n = U.drop_rows_present_in(ref, other, y)
    assert n == 2 and y2.tolist() == [1, 3] and U.cross_split_overlap(ref, X2) == 0.0
    assert len(U.hash_rows(ref)) == 3


@pytest.mark.parametrize("strategy", list(U.STRATEGIES))
def test_pipeline_every_strategy_fits_and_predicts_classifier(toy_xy, strategy):
    X, y = toy_xy
    cw = "balanced" if strategy == "class_weight" else None
    est = RandomForestClassifier(n_estimators=10, random_state=0, class_weight=cw)
    pipe = U.make_model_pipeline(est, NUM, CAT, strategy, scale=False, random_state=0).fit(X, y)
    assert pipe.named_steps["model"] is est and "pre" in pipe.named_steps
    assert set(pipe.predict(X)) <= {0, 1, 2}


@pytest.mark.parametrize("strategy", list(U.STRATEGIES))
def test_pipeline_every_strategy_fits_regressor_on_integer_codes(toy_xy, strategy):
    X, y = toy_xy
    pipe = U.make_model_pipeline(RandomForestRegressor(n_estimators=10, random_state=0), NUM, CAT, strategy,
                                 random_state=0).fit(X, y.astype(int))
    pred = pipe.predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


def test_pipeline_unknown_strategy_raises():
    with pytest.raises(ValueError):
        U.make_model_pipeline(LogisticRegression(), NUM, CAT, "bogus")


def test_oversampling_pipeline_balances_training_fit_only(toy_xy):
    X, y = toy_xy
    pipe = U.make_model_pipeline(RandomForestClassifier(n_estimators=5, random_state=0), NUM, CAT, "smotenc",
                                 random_state=0)
    _, ys = pipe[:2].fit_resample(X, y)
    assert pd.Series(ys).value_counts().nunique() == 1 and len(ys) > len(y)
    assert len(X) == 240  # les données d'origine ne sont pas modifiées


def test_select_by_parsimony_prefers_simplest_within_one_std():
    t = pd.DataFrame({"f1": [0.90, 0.905, 0.95, 0.80], "std": [0.01] * 4}, index=ORDER)
    assert U.select_by_parsimony(t, "f1", "std", ORDER) == "ros"
    t.loc["ros", "f1"] = 0.915
    assert U.select_by_parsimony(t, "f1", "std", ORDER) == "class_weight"


def test_select_calibrator_returns_probabilities_from_validation_only():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    scores = (y * 2 - 1) * 1.5 + rng.normal(0, 1, 2000)  # décisions brutes (non probabilistes)
    name, calib, brier = U.select_calibrator(scores, y, random_state=0)
    p = calib(scores)
    assert name in {"platt", "isotonic"} and set(brier.index) == {"platt", "isotonic"}
    assert p.min() >= 0 and p.max() <= 1 and brier.min() < 0.25
    _, _, brier_p = U.select_calibrator(np.clip(0.5 + scores / 10, 0, 1), y, 0, is_probability=True)
    assert "raw" in brier_p.index


def test_select_calibrator_is_deterministic_given_inputs():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 0.7, 500)
    a = U.select_calibrator(s, y, 7)
    b = U.select_calibrator(s.copy(), y.copy(), 7)
    assert a[0] == b[0] and a[2].equals(b[2])


def test_measure_predict_time_positive():
    assert U.measure_predict_time(np.sort, np.random.rand(2000), repeats=2) > 0


def test_save_result_writes_valid_json_with_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "RESULT_DIR", tmp_path)
    U.save_result("t", {"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.5, np.inf]),
                        "d": {"e": pd.Series({"x": 1})}, "f": ["lim"]})
    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert data == {"a": None, "b": 3, "c": [1.5, None], "d": {"e": {"x": 1}}, "f": ["lim"]}
