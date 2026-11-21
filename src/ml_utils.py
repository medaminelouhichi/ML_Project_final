"""Utilitaires partagés par les 6 notebooks (TON_IoT network : détection d'anomalies + prédiction du type).

Objectif : un seul endroit pour le chargement, le feature engineering, le préprocesseur et les métriques,
afin que les 5 membres de l'équipe travaillent sur exactement les mêmes données et les mêmes règles.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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

# Codage ordinal du type d'attaque (utilisé UNIQUEMENT pour les régresseurs).
# Ordre = progression d'un scénario d'attaque : normal -> reconnaissance -> déni de service ->
# attaques applicatives (web) -> compromission/malware -> interception.
# Des codes proches correspondent donc à des comportements réseau proches, ce qui donne un sens
# à une régression (contrairement à l'ordre alphabétique).
TYPE_ORDER = ["normal", "scanning", "dos", "ddos", "password", "injection", "xss",
              "backdoor", "ransomware", "mitm"]
TYPE_TO_CODE = {t: i for i, t in enumerate(TYPE_ORDER)}
CODE_TO_TYPE = {i: t for t, i in TYPE_TO_CODE.items()}

# ----------------------------------------------------------------------------- chargement
ID_COLUMNS = ["src_ip", "dst_ip"]  # identifiants d'hôtes : mémorisation de l'environnement -> exclus


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut. Les '-' (valeur 'absent' de Zeek) sont convertis en NaN : sans cela,
    le contrôle qualité affiche 0 valeur manquante alors que ~20 colonnes sont presque vides."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    obj = df.select_dtypes(include="object").columns
    df[obj] = df[obj].apply(lambda s: s.str.strip()).replace("-", np.nan)
    df["type"] = df["type"].astype(str).str.lower()
    df["label"] = df["label"].astype("int8")
    return df


# ----------------------------------------------------------------------------- feature engineering
def _port_bucket(p: pd.Series) -> pd.Series:
    return pd.cut(p, bins=[-1, 1023, 49151, 65535], labels=["well_known", "registered", "ephemeral"]).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées ligne par ligne (sans statistique globale -> aucune fuite possible)."""
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
    # 4) réseau interne : l'attaque vient-elle d'un hôte du sous-réseau 192.168.1.x ?
    out["src_internal"] = out["src_ip"].str.startswith("192.168.").astype("int8")
    out["dst_internal"] = out["dst_ip"].str.startswith("192.168.").astype("int8")
    return out


# colonnes dont le taux de NaN dépasse ce seuil sont supprimées après création des indicateurs has_*
MAX_MISSING_RATE = 0.99
LOG_COLUMNS = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes",
               "missed_bytes", "total_bytes", "total_pkts", "bytes_ratio", "pkts_ratio", "bytes_per_pkt_src",
               "bytes_per_pkt_dst", "bytes_per_sec", "pkts_per_sec", "ip_overhead_src", "ip_overhead_dst",
               "http_request_body_len", "http_response_body_len"]


# ----------------------------------------------------------------------------- préprocesseur
def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(numeric_cols, categorical_cols, scale: bool = True, min_frequency: int = 30):
    """ColumnTransformer sans fuite : imputation, log1p des variables à queue lourde, standardisation
    (SVM seulement), encodage one-hot. Un NaN catégoriel devient la modalité 'missing' (l'absence est
    informative en réseau : pas de requête DNS/HTTP), pas la modalité la plus fréquente."""
    log_cols = [c for c in numeric_cols if c in LOG_COLUMNS]
    lin_cols = [c for c in numeric_cols if c not in LOG_COLUMNS]
    log_steps = [("imputer", SimpleImputer(strategy="median")),
                 ("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one"))]
    lin_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        log_steps.append(("scale", StandardScaler()))
        lin_steps.append(("scale", StandardScaler()))
    transformers = []
    if log_cols:
        transformers.append(("num_log", Pipeline(log_steps), log_cols))
    if lin_cols:
        transformers.append(("num_lin", Pipeline(lin_steps), lin_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency)),
        ]), categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ----------------------------------------------------------------------------- données préparées
def load_split(name: str):
    """Charge X/y d'un split produit par 01_prep.ipynb. name in {'train','valid','test'}."""
    X = pd.read_parquet(PROC_DIR / f"X_{name}.parquet")
    y = pd.read_parquet(PROC_DIR / f"y_{name}.parquet")
    return X, y


def load_config() -> dict:
    return json.loads((PROC_DIR / "config.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- métriques
def best_f1_threshold(y_true, scores, n_grid: int = 400):
    """Seuil maximisant le F1 (à appeler sur la VALIDATION uniquement)."""
    qs = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, n_grid)))
    f1s = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in qs]
    i = int(np.argmax(f1s))
    return float(qs[i]), float(f1s[i])


def binary_metrics(y_true, pred, scores=None, proba=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1)})
    if scores is not None:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    if proba is not None:
        out["brier"] = brier_score_loss(y_true, proba)
    return out


def regression_metrics(y_true_code, pred_raw, n_classes: int) -> dict:
    """Métriques de régression + métriques de classification après arrondi au code le plus proche."""
    y_true_code = np.asarray(y_true_code, dtype=float)
    pred_raw = np.asarray(pred_raw, dtype=float)
    pred_cls = np.clip(np.rint(pred_raw), 0, n_classes - 1).astype(int)
    y_int = y_true_code.astype(int)
    return {
        "mae": mean_absolute_error(y_true_code, pred_raw),
        "rmse": float(np.sqrt(mean_squared_error(y_true_code, pred_raw))),
        "r2": r2_score(y_true_code, pred_raw),
        "accuracy": accuracy_score(y_int, pred_cls),
        "balanced_accuracy": balanced_accuracy_score(y_int, pred_cls),
        "f1_macro": f1_score(y_int, pred_cls, average="macro", zero_division=0),
        "within_1_class": float(np.mean(np.abs(pred_cls - y_int) <= 1)),
    }


# ----------------------------------------------------------------------------- noms de features
def feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


# ----------------------------------------------------------------------------- graphiques
def savefig(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=140, bbox_inches="tight")


def plot_confusion(cm, labels, title, ax=None, normalize=False):
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    thr = shown.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > thr else "black")
    return ax


def plot_roc_pr(y_true, scores, label, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = roc_curve(y_true, scores)
    p, r, _ = precision_recall_curve(y_true, scores)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, scores):.4f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC")
    axes[1].plot(r, p, label=f"{label} (AP={average_precision_score(y_true, scores):.4f})")
    axes[1].set(xlabel="Rappel", ylabel="Précision", title="Courbe précision-rappel")
    for a in axes:
        a.legend(loc="lower right")
        a.grid(alpha=.3)
    return axes


def plot_calibration(y_true, probas: dict, n_bins: int = 10, ax=None):
    """Diagramme de fiabilité : probabilité prédite vs fréquence observée."""
    from sklearn.calibration import calibration_curve
    ax = ax or plt.subplots(figsize=(5.5, 5))[1]
    ax.plot([0, 1], [0, 1], "k:", label="Calibration parfaite")
    for name, p in probas.items():
        frac, mean_p = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=f"{name} (Brier={brier_score_loss(y_true, p):.4f})")
    ax.set(xlabel="Probabilité prédite", ylabel="Fréquence observée", title="Diagramme de fiabilité")
    ax.legend()
    ax.grid(alpha=.3)
    return ax


# ----------------------------------------------------------------------------- résultats
def save_result(name: str, payload: dict):
    (RESULT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def load_results() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULT_DIR.glob("*.json"))}


# ----------------------------------------------------------------------------- décodage ordinal (régresseurs)
def _macro_f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    return float(np.mean(np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)))


def decode_with_cuts(pred_raw, cuts):
    """Transforme une sortie continue en classe via des seuils croissants (np.digitize)."""
    return np.digitize(np.asarray(pred_raw, dtype=float), cuts)


def optimize_cutpoints(pred_raw, y_int, n_classes, n_pass=3, n_grid=41):
    """Cherche, par descente de coordonnées, les n_classes-1 seuils qui maximisent le F1 macro.
    A APPELER SUR LA VALIDATION. Point de départ : arrondi classique (0.5, 1.5, ...)."""
    pred_raw, y_int = np.asarray(pred_raw, dtype=float), np.asarray(y_int, dtype=int)
    cuts = np.arange(n_classes - 1) + 0.5
    best = _macro_f1_fast(y_int, decode_with_cuts(pred_raw, cuts), n_classes)
    for _ in range(n_pass):
        for i in range(n_classes - 1):
            lo = cuts[i - 1] + 1e-3 if i > 0 else cuts[i] - 1.0
            hi = cuts[i + 1] - 1e-3 if i < n_classes - 2 else cuts[i] + 1.0
            for c in np.linspace(lo, hi, n_grid):
                trial = cuts.copy(); trial[i] = c
                s = _macro_f1_fast(y_int, decode_with_cuts(pred_raw, trial), n_classes)
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts, best


def regression_metrics_from_classes(y_true_code, pred_raw, pred_cls, n_classes: int) -> dict:
    """Comme regression_metrics mais avec un décodage de classes fourni (ex. seuils optimisés)."""
    y_int = np.asarray(y_true_code, dtype=int)
    return {
        "mae": mean_absolute_error(y_true_code, pred_raw),
        "rmse": float(np.sqrt(mean_squared_error(y_true_code, pred_raw))),
        "r2": r2_score(y_true_code, pred_raw),
        "accuracy": accuracy_score(y_int, pred_cls),
        "balanced_accuracy": balanced_accuracy_score(y_int, pred_cls),
        "f1_macro": f1_score(y_int, pred_cls, average="macro", zero_division=0),
        "within_1_class": float(np.mean(np.abs(np.asarray(pred_cls) - y_int) <= 1)),
    }
