# ABOUTME: Fits called-strike probability models on location, count and handedness,
# ABOUTME: cross-validated by game, and writes out-of-fold predictions for attribution.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

PROCESSED = Path("data/processed")
CALLED_PITCHES = PROCESSED / "called_pitches.parquet"
PREDICTIONS = PROCESSED / "predictions.parquet"
FIGURES = Path("figures")

# Folds are split on game_pk, never on pitches. Pitches within a game share an
# umpire, a park and a day's conditions, so a pitch-level split would leak all
# of that across the fold boundary and flatter the calibration.
N_SPLITS = 5
GROUP = "game_pk"

TARGET = "is_strike"
LOCATION = ["plate_x", "plate_z"]
CATEGORICAL = ["count", "stand"]
# The baseline sees location, count and handedness. The gradient booster adds
# what the pitch was, which the baseline is deliberately blind to.
BASELINE_FEATURES = LOCATION + CATEGORICAL
GBM_FEATURES = BASELINE_FEATURES + ["p_throws", "pitch_type", "release_speed"]
GBM_CATEGORICAL = CATEGORICAL + ["p_throws", "pitch_type"]

# Carried through so the attribution step can join without re-reading the pitches.
CARRY = [GROUP, "fielder_2", "pitcher"]

POLYNOMIAL_DEGREE = 3
CALIBRATION_EDGES = np.round(np.arange(0, 1.05, 0.05), 2)

# Pitches nobody argues about are most of the data and none of the interest.
# This is the band where the call could plausibly go either way.
CONTESTED = (0.1, 0.9)


def load():
    columns = sorted(set(GBM_FEATURES + CARRY + [TARGET]))
    df = pd.read_parquet(CALLED_PITCHES, columns=columns)
    for column in LOCATION:
        df[column] = df[column].astype("float32")
    for column in GBM_CATEGORICAL:
        df[column] = df[column].astype("category")
    return df


def baseline_model():
    """Logistic regression on polynomial location terms plus count and handedness.

    Deliberately weak: a smooth polynomial surface cannot reproduce the flat top
    and sharp shoulders of a real zone, which is the point of a baseline."""
    location = Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(POLYNOMIAL_DEGREE, include_bias=False)),
        ]
    )
    features = ColumnTransformer(
        [
            ("location", location, LOCATION),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("features", features), ("model", LogisticRegression(max_iter=1000))])


def gbm_model():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=200, random_state=0
    )


def out_of_fold_predictions(model, df, features):
    """Strike probability for every pitch, each predicted by a model that never
    saw its game. These double as the held-out scores and as the residual source
    for umpire and catcher attribution."""
    folds = GroupKFold(n_splits=N_SPLITS)
    probabilities = cross_val_predict(
        model,
        df[features],
        df[TARGET],
        groups=df[GROUP],
        cv=folds,
        method="predict_proba",
        n_jobs=1,  # ponytail: sequential; parallel folds copy 3.7M rows per worker
    )
    return probabilities[:, 1]


def evaluate(actual, predicted):
    return {
        "log_loss": log_loss(actual, predicted),
        "brier": brier_score_loss(actual, predicted),
        "auc": roc_auc_score(actual, predicted),
    }


def calibration_table(actual, predicted, edges=CALIBRATION_EDGES):
    """Mean predicted against mean observed, in fixed-width bins of predicted.

    Fixed width rather than equal count on purpose. Two thirds of all pitches
    are obvious takes, so equal-count bins spend most of their resolution on
    calls no umpire gets wrong and squeeze the contested band into one or two
    points, hiding the only miscalibration that would matter downstream."""
    frame = pd.DataFrame(
        {
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )
    frame["bin"] = pd.cut(frame["predicted"], edges, include_lowest=True)
    table = frame.groupby("bin", observed=True).agg(
        pitches=("actual", "size"),
        predicted=("predicted", "mean"),
        observed=("actual", "mean"),
    )
    table["gap"] = table["observed"] - table["predicted"]
    table = table.reset_index()
    table["band"] = [f"{b.left:.2f}-{b.right:.2f}" for b in table["bin"]]
    return table.drop(columns=["bin"])[["band", "pitches", "predicted", "observed", "gap"]]


def contested_scores(actual, predicted, band=CONTESTED):
    """Log loss restricted to pitches the model itself calls a coin flip.

    Overall log loss is dominated by the obvious takes, so two models can look
    close there while differing on the calls attribution is actually built on."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    inside = (predicted >= band[0]) & (predicted <= band[1])
    return {
        "pitches": int(inside.sum()),
        "share": float(inside.mean()),
        "log_loss": log_loss(actual[inside], predicted[inside]),
    }


def figure_calibration(tables, path):
    """Plotted as the gap from perfect, not as observed against predicted: on a
    diagonal plot a 2-point miss is invisible, which is how a badly calibrated
    model passes a calibration chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="black", ls="--", lw=1)
    for (name, table), color in zip(tables.items(), ["#c1442f", "#3b6ea5"]):
        ax.plot(table["predicted"], table["gap"], marker="o", color=color, label=name)
    ax.axvspan(*CONTESTED, color="#cccccc", alpha=0.35, zorder=0, label="contested band")
    ax.set_xlabel("predicted strike probability")
    ax.set_ylabel("observed − predicted")
    ax.set_title("Calibration error, held out by game")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


MODELS = {"baseline": "p_baseline", "gradient boosting": "p_gbm"}


def fit_and_save():
    """The expensive half: ten model fits over 3.7M pitches."""
    df = load()
    print(f"{len(df):,} called pitches across {df[GROUP].nunique():,} games")

    fits = {"baseline": (baseline_model(), BASELINE_FEATURES),
            "gradient boosting": (gbm_model(), GBM_FEATURES)}
    out = df[CARRY + [TARGET]].copy()
    for name, (model, features) in fits.items():
        print(f"fitting {name} over {N_SPLITS} game-grouped folds...")
        out[MODELS[name]] = out_of_fold_predictions(model, df, features).astype("float32")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS, index=False)
    print(f"wrote {PREDICTIONS}: {len(out):,} out-of-fold predictions\n")


def report():
    """The cheap half, reading back the saved predictions. Kept separate so that
    changing how a score is presented does not cost another ten model fits."""
    saved = pd.read_parquet(PREDICTIONS)
    actual = saved[TARGET]
    base_rate = actual.mean()
    print(f"base rate {base_rate:.4f}, log loss of always predicting it: "
          f"{log_loss(actual, np.full(len(actual), base_rate)):.4f}")

    scores = {name: evaluate(actual, saved[column]) for name, column in MODELS.items()}
    contested = {name: contested_scores(actual, saved[column]) for name, column in MODELS.items()}
    tables = {name: calibration_table(actual, saved[column]) for name, column in MODELS.items()}

    print("\nheld-out scores, all pitches (out of fold)")
    print(pd.DataFrame(scores).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nheld-out scores, contested band {CONTESTED[0]}-{CONTESTED[1]}")
    print(pd.DataFrame(contested).T.to_string(float_format=lambda v: f"{v:.4f}"))
    for name, table in tables.items():
        print(f"\ncalibration, {name}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    FIGURES.mkdir(exist_ok=True)
    figure_calibration(tables, FIGURES / "calibration.png")
    print(f"\nwrote {FIGURES / 'calibration.png'}")


def main(refit):
    if refit or not PREDICTIONS.exists():
        fit_and_save()
    else:
        print(f"reusing {PREDICTIONS}; pass --refit to fit again\n")
    report()


def _self_check():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "game_pk": rng.integers(0, 40, n),
            "fielder_2": rng.integers(0, 5, n),
            "plate_x": rng.uniform(-1.5, 1.5, n).astype("float32"),
            "plate_z": rng.uniform(1.0, 4.0, n).astype("float32"),
            "count": pd.Categorical(rng.choice(["0-0", "3-0", "0-2"], n)),
            "stand": pd.Categorical(rng.choice(["R", "L"], n)),
        }
    )
    df[TARGET] = (df["plate_x"].abs() < 0.7) & (df["plate_z"].between(1.6, 3.4))

    # The decision that matters: no game may appear on both sides of a fold, or
    # the model is scored on games it has already seen.
    for train, test in GroupKFold(n_splits=3).split(df, df[TARGET], groups=df[GROUP]):
        shared = set(df[GROUP].iloc[train]) & set(df[GROUP].iloc[test])
        assert not shared, f"game leaked across the fold boundary: {shared}"

    # The baseline's column wiring works end to end and yields real probabilities.
    model = baseline_model().fit(df[BASELINE_FEATURES], df[TARGET])
    p = model.predict_proba(df[BASELINE_FEATURES])[:, 1]
    assert p.shape == (n,) and ((p > 0) & (p < 1)).all()

    # Calibration of a perfectly calibrated predictor sits on the diagonal.
    truth = rng.uniform(0.05, 0.95, 200_000)
    calls = rng.random(200_000) < truth
    table = calibration_table(calls, truth)
    assert table["gap"].abs().max() < 0.02, table
    assert table["pitches"].sum() == 200_000, table["pitches"].sum()

    # A confidently wrong predictor must land off it, or the table proves nothing.
    skewed = calibration_table(calls, np.clip(truth + 0.2, 0, 1))
    assert skewed["gap"].abs().max() > 0.1, skewed

    # The contested band counts only the coin flips, not the obvious takes.
    obvious = np.concatenate([np.full(900, 0.001), np.full(100, 0.5)])
    band = contested_scores(rng.random(1000) < obvious, obvious)
    assert band["pitches"] == 100, band
    assert abs(band["share"] - 0.1) < 1e-9, band
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check"]:
        _self_check()
    else:
        main(refit="--refit" in args)
