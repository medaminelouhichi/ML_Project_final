"""Utilitaires partagés par les notebooks (TON_IoT network : détection d'anomalies + prédiction du type).

Objectif : un seul endroit pour le chargement, le feature engineering, les pipelines, les métriques et les
garde-fous anti-fuite, afin que tous les notebooks travaillent sur exactement les mêmes données et règles.
Les graphiques sont dans `ml_plots.py` et ré-exportés ici (`U.plot_*`).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_recall_fscore_support,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_plots import (  # noqa: F401  (ré-export : U.plot_* reste disponible)
    plot_balancing_effect, plot_calibration, plot_class_distribution, plot_confusion, plot_cv_comparison,
    plot_learning_curve, plot_roc_pr,
)

# ----------------------------------------------------------------------------- chemins
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train_test_network.csv"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
RESULT_DIR = REPORT_DIR / "results"
MODEL_DIR = ROOT / "models"
for _d in (PROC_DIR, FIG_DIR, RESULT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Fixe les graines des générateurs Python et NumPy (les modèles reçoivent aussi random_state)."""
    random.seed(seed)
    np.random.seed(seed)


def set_style() -> None:
    """Style graphique commun (seaborn si disponible) et affichage pandas lisible."""
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook")
    except ImportError:  # pragma: no cover
        plt.style.use("ggplot")
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", "{:.4f}".format)


# Ordre DOCUMENTÉ des types (progression d'un scénario d'attaque). C'est une hypothèse : le notebook 01 la
# met à l'épreuve (étude de l'encodage ordinal) et enregistre l'ordre retenu dans config.json.
DOCUMENTED_TYPE_ORDER = ["normal", "scanning", "dos", "ddos", "password", "injection", "xss",
                         "backdoor", "ransomware", "mitm"]


def get_type_order(cfg: dict) -> list[str]:
    """Ordre des types retenu (config.json), à utiliser pour tout codage `type` -> `type_code`."""
    return list(cfg["type_order"])


def type_maps(order: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    """(type -> code, code -> type) pour un ordre donné."""
    to_code = {t: i for i, t in enumerate(order)}
    return to_code, {i: t for t, i in to_code.items()}


# ----------------------------------------------------------------------------- chargement
# Colonnes exclues des variables explicatives, avec la raison (affichée dans 01_prep).
LEAKY_OR_ID_COLUMNS = {
    "src_ip": "identifiant d'hôte : mémorisation de l'environnement du banc d'essai",
    "dst_ip": "identifiant d'hôte : mémorisation de l'environnement du banc d'essai",
    "src_port": "port source éphémère : rend les flux uniques, aucune valeur sémantique "
                "(la catégorie src_port_bucket est conservée)",
}
COUNT_COLUMNS = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts",
                 "src_ip_bytes", "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut. Les '-' (valeur 'absent' de Zeek) sont convertis en NaN : sans cela,
    le contrôle qualité affiche 0 valeur manquante alors que ~20 colonnes sont presque vides."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    obj = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df[obj] = df[obj].apply(lambda s: s.str.strip()).replace("-", np.nan)
    df["type"] = df["type"].astype(str).str.lower()
    df["label"] = df["label"].astype("int8")
    return df


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rapport des valeurs invalides : comptes/octets/durée négatifs, ports hors [0, 65535],
    incohérence label/type, types inconnus. Une ligne par contrôle (n_invalides = 0 -> OK)."""
    rows = []
    for c in (c for c in COUNT_COLUMNS if c in df.columns):
        rows.append((f"{c} < 0", c, int((df[c] < 0).sum())))
    for c in ("src_port", "dst_port"):
        if c in df.columns:
            rows.append((f"{c} hors [0, 65535]", c, int(((df[c] < 0) | (df[c] > 65535)).sum())))
    if {"label", "type"} <= set(df.columns):
        rows.append(("label != (type != 'normal')", "label/type",
                     int((df["label"].to_numpy() != (df["type"] != "normal").to_numpy()).sum())))
        rows.append(("type inconnu", "type", int((~df["type"].isin(DOCUMENTED_TYPE_ORDER)).sum())))
    out = pd.DataFrame(rows, columns=["contrôle", "colonne", "n_invalides"])
    out["part"] = out["n_invalides"] / max(len(df), 1)
    return out


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(colonnes numériques, colonnes catégorielles) selon le dtype (robuste object/str de pandas 3)."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X.columns if c not in num]


# ----------------------------------------------------------------------------- feature engineering
def _port_bucket(p: pd.Series) -> pd.Series:
    return pd.cut(p, bins=[-1, 1023, 49151, 65535], labels=["well_known", "registered", "ephemeral"]).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées ligne par ligne (sans statistique globale -> aucune fuite possible).
    Aucun indicateur issu des adresses IP (topologie du banc d'essai : ne se transfère pas à un autre réseau)."""
    out = df.copy()
    # 1) indicateurs de présence de protocole (remplacent les colonnes http_/ssl_/weird_ quasi vides)
    out["has_dns"] = out["dns_query"].notna().astype("int8")
    out["has_http"] = out["http_method"].notna().astype("int8")
    out["has_ssl"] = out["ssl_version"].notna().astype("int8")
    out["has_weird"] = out["weird_name"].notna().astype("int8")
    # 2) volumes et débits
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["total_pkts"] = out["src_pkts"] + out["dst_pkts"]
    out["bytes_ratio"] = out["src_bytes"] / (out["dst_bytes"] + 1)
    out["pkts_ratio"] = out["src_pkts"] / (out["dst_pkts"] + 1)
    out["bytes_per_pkt_src"] = out["src_bytes"] / (out["src_pkts"] + 1)
    out["bytes_per_pkt_dst"] = out["dst_bytes"] / (out["dst_pkts"] + 1)
    out["bytes_per_sec"] = out["total_bytes"] / (out["duration"] + 1e-3)
    out["pkts_per_sec"] = out["total_pkts"] / (out["duration"] + 1e-3)
    out["ip_overhead_src"] = out["src_ip_bytes"] - out["src_bytes"]
    out["ip_overhead_dst"] = out["dst_ip_bytes"] - out["dst_bytes"]
    # 3) ports : catégories au lieu de 65 535 valeurs quasi-ordinales
    out["src_port_bucket"] = _port_bucket(out["src_port"])
    out["dst_port_bucket"] = _port_bucket(out["dst_port"])
    out["same_ports"] = (out["src_port"] == out["dst_port"]).astype("int8")
    return out


# colonnes dont le taux de NaN dépasse ce seuil sont supprimées après création des indicateurs has_*
MAX_MISSING_RATE = 0.99
LOG_COLUMNS = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
               "missed_bytes", "total_bytes", "total_pkts", "bytes_ratio", "pkts_ratio", "bytes_per_pkt_src",
               "bytes_per_pkt_dst", "bytes_per_sec", "pkts_per_sec", "ip_overhead_src", "ip_overhead_dst",
               "http_request_body_len", "http_response_body_len"]


# ----------------------------------------------------------------------------- préprocesseur et pipelines
def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def _numeric_blocks(numeric_cols: Sequence[str], scale: bool) -> list[tuple]:
    """Transformateurs numériques : imputation médiane, (log1p pour les queues lourdes), standardisation."""
    log_cols = [c for c in numeric_cols if c in LOG_COLUMNS]
    lin_cols = [c for c in numeric_cols if c not in LOG_COLUMNS]
    log_steps = [("imputer", SimpleImputer(strategy="median")),
                 ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one"))]
    lin_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        log_steps.append(("scale", StandardScaler()))
        lin_steps.append(("scale", StandardScaler()))
    blocks = []
    if log_cols:
        blocks.append(("num_log", Pipeline(log_steps), log_cols))
    if lin_cols:
        blocks.append(("num_lin", Pipeline(lin_steps), lin_cols))
    return blocks


def make_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str], scale: bool = True,
                      min_frequency: int = 30) -> ColumnTransformer:
    """ColumnTransformer sans fuite : imputation, log1p des variables à queue lourde, standardisation
    (SVM seulement), encodage one-hot. Un NaN catégoriel devient la modalité 'missing' (l'absence est
    informative en réseau : pas de requête DNS/HTTP), pas la modalité la plus fréquente."""
    transformers = _numeric_blocks(numeric_cols, scale)
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)),
        ]), list(categorical_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


STRATEGIES = ("none", "class_weight", "ros", "smote", "smotenc")
STRATEGY_LABELS = {"none": "Aucun", "class_weight": "Poids de classe", "ros": "Sur-échantillonnage aléatoire",
                   "smote": "SMOTE", "smotenc": "SMOTENC"}


def _smotenc_blocks(num_cols: Sequence[str], cat_cols: Sequence[str], scale: bool, random_state: int) -> list[tuple]:
    """Étapes 'pre' + 'sampler' + 'onehot' de SMOTENC : numériques + catégories ordinales -> SMOTENC -> one-hot."""
    n_num, n_cat = len(num_cols), len(cat_cols)
    ordinal = ColumnTransformer(
        _numeric_blocks(num_cols, scale) + [("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), list(cat_cols))],
        remainder="drop", verbose_feature_names_out=False)
    sampler = SMOTENC(categorical_features=list(range(n_num, n_num + n_cat)), random_state=random_state)
    onehot = ColumnTransformer(
        [("num", "passthrough", list(range(n_num))),
         ("cat", OneHotEncoder(handle_unknown="ignore"), list(range(n_num, n_num + n_cat)))],
        remainder="drop")
    return [("pre", ordinal), ("sampler", sampler), ("onehot", onehot)]


def make_model_pipeline(estimator, num_cols: Sequence[str], cat_cols: Sequence[str], strategy: str = "none",
                        scale: bool = False, random_state: int = RANDOM_STATE) -> ImbPipeline:
    """Pipeline imblearn : préprocesseur ('pre') -> [rééchantillonnage] -> estimateur ('model').

    strategy : 'none' | 'class_weight' (identique à none : le poids se règle sur l'estimateur) | 'ros' |
    'smote' (après one-hot) | 'smotenc' (catégories ordinales, SMOTENC, puis one-hot). Le rééchantillonnage
    n'est appliqué qu'aux plis d'apprentissage (imblearn), jamais à la validation. Fonctionne aussi pour un
    régresseur dont la cible est le code entier de la classe (les échantillonneurs la traitent comme classes)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"stratégie inconnue : {strategy!r} (attendu : {STRATEGIES})")
    if strategy == "smotenc" and cat_cols:
        steps = _smotenc_blocks(num_cols, cat_cols, scale, random_state)
    else:
        steps = [("pre", make_preprocessor(num_cols, cat_cols, scale=scale))]
        if strategy == "ros":
            steps.append(("sampler", RandomOverSampler(random_state=random_state)))
        elif strategy in ("smote", "smotenc"):  # smotenc sans catégorielle == smote
            steps.append(("sampler", SMOTE(random_state=random_state)))
    return ImbPipeline(steps + [("model", estimator)])


def select_by_parsimony(cv_table: pd.DataFrame, metric: str, std_col: str, simplicity_order: Sequence[str]) -> str:
    """Stratégie retenue : la plus simple dont la moyenne est >= (meilleure moyenne - son écart-type).
    Évite de choisir sur du bruit (différences < 1 écart-type de la validation croisée)."""
    best = cv_table[metric].idxmax()
    floor = cv_table.loc[best, metric] - cv_table.loc[best, std_col]
    for name in simplicity_order:
        if name in cv_table.index and cv_table.loc[name, metric] >= floor:
            return name
    return str(best)


# ----------------------------------------------------------------------------- données préparées
def load_split(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge X/y d'un split produit par 01_prep.ipynb. name in {'train','valid','test'}."""
    return pd.read_parquet(PROC_DIR / f"X_{name}.parquet"), pd.read_parquet(PROC_DIR / f"y_{name}.parquet")


def load_config() -> dict:
    return json.loads((PROC_DIR / "config.json").read_text(encoding="utf-8"))


def feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


# ----------------------------------------------------------------------------- garde-fous anti-fuite
def hash_rows(df: pd.DataFrame) -> pd.Series:
    """Empreinte 64 bits de chaque ligne (valeurs uniquement, NaN inclus) pour détecter les doublons."""
    return pd.util.hash_pandas_object(df.reset_index(drop=True), index=False)


def cross_split_overlap(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> float:
    """Part des lignes de `X_other` identiques à une ligne de `X_ref` (même colonnes, même espace de variables)."""
    if len(X_other) == 0:
        return 0.0
    cols = list(X_ref.columns)
    return float(hash_rows(X_other[cols]).isin(set(hash_rows(X_ref[cols]))).mean())


def drop_rows_present_in(X_ref: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
    """Retire de (X, y) les lignes déjà présentes dans X_ref. Renvoie (X, y, n_supprimées), index réinitialisés."""
    cols = list(X_ref.columns)
    dup = hash_rows(X[cols]).isin(set(hash_rows(X_ref[cols]))).to_numpy()
    return X.loc[~dup].reset_index(drop=True), y.loc[~dup].reset_index(drop=True), int(dup.sum())


# ----------------------------------------------------------------------------- métriques
def _binary_counts(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray):
    pos, neg = np.sort(scores[y == 1]), np.sort(scores[y == 0])
    fn = np.searchsorted(pos, thresholds, side="left")  # positifs < seuil -> prédits 0
    tn = np.searchsorted(neg, thresholds, side="left")
    return len(pos) - fn, len(neg) - tn, fn, tn  # tp, fp, fn, tn


def best_threshold(y_true, scores, metric: str = "f1_macro", n_grid: int = 400) -> tuple[float, float]:
    """Seuil de décision maximisant `metric` ('f1_macro' | 'f1' | 'balanced_accuracy'). VALIDATION uniquement.
    Prédiction positive si score >= seuil. Renvoie (seuil, valeur)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    qs = np.unique(np.quantile(s, np.linspace(0.001, 0.999, n_grid)))
    tp, fp, fn, tn = (a.astype(float) for a in _binary_counts(y, s, qs))
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn), where=(2 * tn + fn + fp) > 0)
    if metric == "f1_macro":
        vals = (f1_pos + f1_neg) / 2
    elif metric == "f1":
        vals = f1_pos
    elif metric == "balanced_accuracy":
        vals = (tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1)) / 2
    else:
        raise ValueError(f"métrique non gérée : {metric!r}")
    i = int(np.argmax(vals))
    return float(qs[i]), float(vals[i])


def binary_metrics(y_true, pred, scores=None, proba=None) -> dict:
    """Métriques de détection binaire (classe positive = attaque). `scores` -> ROC/PR AUC ; `proba` -> Brier."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, pred, labels=[0, 1]).ravel())
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
    }
    if scores is not None:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    if proba is not None:
        out["brier"] = brier_score_loss(y_true, proba)
    return out


def multiclass_metrics(y_true, y_pred, labels: Sequence) -> dict:
    """Métriques de classification multi-classe (moyennes macro = classes à poids égal, weighted = par effectif)."""
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "mcc": matthews_corrcoef(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    return out


def per_class_report(y_true, y_pred, labels: Sequence, names: Sequence[str]) -> pd.DataFrame:
    """Précision, rappel, F1 et support par classe (index = noms des classes)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(names))


def regression_metrics(y_true_code, pred_raw) -> dict:
    """Métriques de régression sur la sortie CONTINUE (avant décodage) : MAE, RMSE, R²."""
    y, p = np.asarray(y_true_code, dtype=float), np.asarray(pred_raw, dtype=float)
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": r2_score(y, p)}


def decode_round(pred_raw, n_classes: int) -> np.ndarray:
    """Décodage naïf : code entier le plus proche, borné à [0, n_classes-1]."""
    return np.clip(np.rint(np.asarray(pred_raw, dtype=float)), 0, n_classes - 1).astype(int)


def within_k(y, pred, k: int = 1) -> float:
    """Part des prédictions à au plus k codes de la vraie classe (n'a de sens que pour un ordre ordinal)."""
    return float(np.mean(np.abs(np.asarray(pred, dtype=int) - np.asarray(y, dtype=int)) <= k))


def _macro_f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    return float(np.mean(np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)))


def decode_with_cuts(pred_raw, cuts) -> np.ndarray:
    """Transforme une sortie continue en classe via des seuils croissants (np.digitize)."""
    return np.digitize(np.asarray(pred_raw, dtype=float), cuts)


def optimize_cutpoints(pred_raw, y_int, n_classes: int, n_pass: int = 3, n_grid: int = 41):
    """Cherche, par descente de coordonnées, les n_classes-1 seuils qui maximisent le F1 macro.
    A APPELER SUR LA VALIDATION. Point de départ : arrondi classique (0.5, 1.5, ...). Renvoie (seuils, F1 macro)."""
    pred_raw, y_int = np.asarray(pred_raw, dtype=float), np.asarray(y_int, dtype=int)
    cuts = np.arange(n_classes - 1) + 0.5
    best = _macro_f1_fast(y_int, decode_with_cuts(pred_raw, cuts), n_classes)
    for _ in range(n_pass):
        for i in range(n_classes - 1):
            lo = cuts[i - 1] + 1e-3 if i > 0 else cuts[i] - 1.0
            hi = cuts[i + 1] - 1e-3 if i < n_classes - 2 else cuts[i] + 1.0
            for c in np.linspace(lo, hi, n_grid):
                trial = cuts.copy()
                trial[i] = c
                s = _macro_f1_fast(y_int, decode_with_cuts(pred_raw, trial), n_classes)
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts, best


# ----------------------------------------------------------------------------- calibration
def select_calibrator(scores_valid, y_valid, random_state: int = RANDOM_STATE, is_probability: bool = False):
    """Choisit le calibrateur de probabilités SANS jamais regarder le test.

    Split stratifié 50/50 de la validation : Platt (régression logistique sur le score) et isotonique
    ajustés sur la moitié A, Brier mesuré sur la moitié B (+ score brut si c'est déjà une probabilité).
    Le gagnant est réajusté sur toute la validation. Renvoie (nom, fonction score->proba, table Brier)."""
    s, y = np.asarray(scores_valid, dtype=float), np.asarray(y_valid).astype(int)
    ia, ib = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=random_state)

    def fit(kind: str, idx) -> Callable:
        if kind == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(s[idx].reshape(-1, 1), y[idx])
            return lambda v: m.predict_proba(np.asarray(v, dtype=float).reshape(-1, 1))[:, 1]
        if kind == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[idx], y[idx])
            return lambda v: m.predict(np.asarray(v, dtype=float))
        return lambda v: np.clip(np.asarray(v, dtype=float), 0, 1)  # 'raw'

    kinds = ["platt", "isotonic"] + (["raw"] if is_probability else [])
    brier = pd.Series({k: brier_score_loss(y[ib], fit(k, ia)(s[ib])) for k in kinds}, name="brier")
    best = str(brier.idxmin())
    return best, fit(best, np.arange(len(y))), brier


def measure_predict_time(fn: Callable, X, repeats: int = 3) -> float:
    """Temps de prédiction en millisecondes pour 1000 lignes (meilleur de `repeats` essais)."""
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        durations.append(time.perf_counter() - t0)
    best = min(durations)
    return float(best * 1000.0 / (len(X) / 1000.0))


# ----------------------------------------------------------------------------- graphiques / résultats
def savefig(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=140, bbox_inches="tight")


def _jsonable(obj):
    """Convertit récursivement types numpy / pandas / Path en types JSON ; NaN et inf deviennent null."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict("index"))
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_result(name: str, payload: dict) -> None:
    """Enregistre `payload` dans reports/results/<name>.json (JSON valide : NaN -> null, numpy -> natifs)."""
    text = json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
    (RESULT_DIR / f"{name}.json").write_text(text, encoding="utf-8")


def load_results() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULT_DIR.glob("*.json"))}
