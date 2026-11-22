"""Graphiques partagés (libellés en français). Ré-exportés par `ml_utils` : `U.plot_*` fonctionne.

Aucune dépendance vers `ml_utils` (évite les imports circulaires) : les fonctions reçoivent des
tableaux / Series et dessinent sur un `Axes` fourni (ou en créent un).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import learning_curve


def plot_confusion(cm, labels: Sequence[str], title: str, ax=None, normalize: bool = False):
    """Matrice de confusion (effectifs ou normalisée par ligne, c.-à-d. rappel par classe)."""
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    ax.imshow(shown, cmap="Blues")
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


def plot_roc_pr(y_true, scores, label: str, axes=None):
    """Courbes ROC et précision-rappel d'un score ; appeler plusieurs fois pour superposer."""
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


def plot_class_distribution(series: pd.Series, title: str, ax=None, order: Sequence | None = None):
    """Barres des effectifs par classe, annotées avec l'effectif et le pourcentage."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    counts = series.value_counts()
    counts = counts.reindex(order).fillna(0) if order is not None else counts.sort_index()
    bars = ax.bar([str(i) for i in counts.index], counts.values, color="#2878b5")
    total = max(counts.sum(), 1)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}\n{v / total:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set(title=title, xlabel="Classe", ylabel="Nombre de lignes")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 1)
    ax.grid(axis="y", alpha=.3)
    return ax


def plot_balancing_effect(before: pd.Series, after: dict[str, pd.Series], ax=None):
    """Effectifs par classe avant / après chaque rééchantillonnage (échelle log : classes très inégales)."""
    ax = ax or plt.subplots(figsize=(11, 4.5))[1]
    table = pd.DataFrame({"Avant (train)": before.value_counts(), **{k: v.value_counts() for k, v in after.items()}})
    table = table.reindex(before.value_counts().index).fillna(0)
    table.plot(kind="bar", ax=ax, width=.85)
    ax.set(title="Effet du rééchantillonnage sur la distribution des classes (train uniquement)",
           xlabel="Classe", ylabel="Nombre de lignes (échelle log)", yscale="log")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Stratégie")
    ax.grid(axis="y", alpha=.3)
    return ax


def plot_learning_curve(estimator, X, y, cv, scoring: str, title: str, ax=None, train_sizes=None, n_jobs: int = -1):
    """Courbe d'apprentissage (score train vs validation croisée, bande = écart-type)."""
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    sizes = np.linspace(0.1, 1.0, 5) if train_sizes is None else train_sizes
    n, tr, va = learning_curve(estimator, X, y, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=n_jobs)
    for scores, name, color in [(tr, "Entraînement", "#c0392b"), (va, "Validation croisée", "#27ae60")]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(n, m, marker="o", color=color, label=name)
        ax.fill_between(n, m - s, m + s, color=color, alpha=.15)
    ax.set(title=title, xlabel="Nombre de lignes d'entraînement", ylabel=scoring)
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    return ax


def plot_cv_comparison(df: pd.DataFrame, metrics: Sequence[str], std_cols: Sequence[str], title: str, ax=None):
    """Barres groupées (une couleur par métrique, une barre par stratégie = ligne de `df`) avec barres d'erreur ±1 écart-type."""
    ax = ax or plt.subplots(figsize=(11, 4.5))[1]
    n_m, x = len(metrics), np.arange(len(df))
    w = 0.8 / n_m
    for k, (m, s) in enumerate(zip(metrics, std_cols)):
        ax.bar(x + (k - (n_m - 1) / 2) * w, df[m], w, yerr=df[s], capsize=3, label=m)
    ax.set_xticks(x, [str(i) for i in df.index], rotation=20)
    ax.set(title=title, xlabel="Stratégie", ylabel="Score (moyenne ± écart-type, validation croisée)")
    lo = float((df[list(metrics)].values - df[list(std_cols)].values).min())
    ax.set_ylim(max(0.0, lo - 0.05), 1.0 if df[list(metrics)].values.max() <= 1 else None)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=.3)
    return ax
