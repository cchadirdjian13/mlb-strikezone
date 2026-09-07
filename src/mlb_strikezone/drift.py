# ABOUTME: Refits umpire and catcher effects one season at a time and measures
# ABOUTME: whether the spread between them has narrowed across 2015-2025.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlb_strikezone.attribution import (
    PREDICTIONS,
    ROLES,
    UMPIRES,
    bootstrap_standard_errors,
    build_design,
    fit_joint,
    games_as_row_blocks,
)

PROCESSED = Path("data/processed")
SEASON_EFFECTS = PROCESSED / "season_effects.parquet"
SEASON_SPREADS = PROCESSED / "season_spreads.parquet"
FIGURES = Path("figures")

# Far lighter than the pooled fit's 2000. Ridge shrinks an effect by roughly
# n/(n + alpha), and a season holds an eighth of an umpire's pooled workload, so
# the pooled penalty would shrink each season hard and 2020 — a third the length
# — hardest of all. That alone would manufacture a convergence trend out of
# nothing. At 100 the shrinkage is under 5% in every season including 2020.
SEASON_ALPHA = 100
SEASON_MIN_PITCHES = 1000
BOOTSTRAP_REPLICATES = 100

# The interval on a season's spread is nested: the outer loop resamples games,
# and each outer replicate needs its own standard errors before its spread can
# be corrected. Inner stays small because it only has to pin an average of
# squared errors over ~90 people, which is far easier than pinning any one.
#
# Only the width of that distribution is used. Its centre sits roughly 15% above
# the statistic, because an inner bootstrap drawn from an already-resampled set
# of games understates that replicate's noise, so every replicate under-corrects
# and lands high. That is an artefact of nesting rather than anything about the
# season, so percentile bounds would be misleading and the interval is built as
# the point estimate plus and minus two bootstrap standard errors.
OUTER_REPLICATES = 100
INNER_REPLICATES = 15

SHORT_SEASON = 2020


def load():
    pitches = pd.read_parquet(PREDICTIONS)
    umpires = pd.read_parquet(UMPIRES, columns=["game_pk", "game_date", "ump_id"])
    df = pitches.merge(umpires, on="game_pk", how="inner")
    df["season"] = df["game_date"].str.slice(0, 4).astype(int)
    df["residual"] = df["is_strike"].astype("float32") - df["p_gbm"]
    return df


def corrected_spread(effect, se):
    """Standard deviation of the true effects, with estimation noise removed.

    The observed spread of estimates carries its own standard errors on top of
    the real variation, and carries more of them in a season with less data.
    Subtracting the mean squared error is what stops the shortest season looking
    like the most variable one purely because it is the shortest."""
    variance = effect.var(ddof=1) - np.mean(np.asarray(se) ** 2)
    return float(np.sqrt(max(variance, 0.0)))


def season_effects(df, season):
    """Joint umpire and catcher fit on a single season, with bootstrap errors."""
    slice_ = df[df["season"] == season]
    effects = fit_joint(slice_, alpha=SEASON_ALPHA).merge(
        bootstrap_standard_errors(
            slice_, replicates=BOOTSTRAP_REPLICATES, alpha=SEASON_ALPHA
        ),
        on=["role", "id"],
    )

    counts = []
    for column, role in ROLES.items():
        sized = slice_.groupby(column).size()
        counts.append(pd.DataFrame({"role": role, "id": sized.index, "pitches": sized.to_numpy()}))
    return effects.merge(pd.concat(counts, ignore_index=True), on=["role", "id"]).assign(
        season=season
    )


def _resample(blocks, picked):
    return np.concatenate([blocks[game] for game in picked])


def spread_bootstrap(df, season, outer=OUTER_REPLICATES, inner=INNER_REPLICATES, seed=0):
    """Bootstrap distribution of each role's corrected spread for one season.

    Nested rather than single-level, because the spread is not a mean. It is a
    variance with the estimation noise subtracted, and that subtraction needs the
    standard errors of the effects, which are themselves a bootstrap. Every outer
    replicate therefore re-estimates them from its own resampled games. Reusing
    the full sample's standard errors would apply one estimator to the point
    estimate and a different one to the replicates, and the interval would
    describe neither."""
    slice_ = df[df["season"] == season]
    design, encoder = build_design(slice_)
    residual = slice_["residual"].to_numpy()
    blocks = games_as_row_blocks(slice_)
    columns = {
        "umpire": slice(0, len(encoder.categories_[0])),
        "catcher": slice(len(encoder.categories_[0]), None),
    }
    rng = np.random.default_rng(seed)

    draws = {role: np.empty(outer) for role in columns}
    for replicate in range(outer):
        picked = rng.integers(0, len(blocks), len(blocks))
        rows = _resample(blocks, picked)
        effect = Ridge(alpha=SEASON_ALPHA).fit(design[rows], residual[rows]).coef_
        # One-hot columns sum to each person's pitch count, so the qualification
        # floor can be reapplied per replicate without touching the frame.
        counts = np.asarray(design[rows].sum(axis=0)).ravel()

        inner_draws = np.empty((inner, design.shape[1]))
        for step in range(inner):
            repicked = picked[rng.integers(0, len(picked), len(picked))]
            inner_rows = _resample(blocks, repicked)
            inner_draws[step] = (
                Ridge(alpha=SEASON_ALPHA).fit(design[inner_rows], residual[inner_rows]).coef_
            )
        se = inner_draws.std(axis=0, ddof=1)

        for role, span in columns.items():
            keep = counts[span] >= SEASON_MIN_PITCHES
            draws[role][replicate] = corrected_spread(effect[span][keep], se[span][keep])
    return draws


def summarise_bootstrap(draws, season):
    """Width of the bootstrap distribution, and where it sits relative to the
    statistic so the shift stays visible rather than being quietly absorbed."""
    rows = []
    for role, values in draws.items():
        rows.append(
            {
                "season": season,
                "role": role,
                "spread_se": values.std(ddof=1) * 100,
                "draw_median": np.median(values) * 100,
            }
        )
    return pd.DataFrame(rows)


def spread_by_season(effects):
    rows = []
    for (season, role), group in effects.groupby(["season", "role"]):
        qualified = group[group["pitches"] >= SEASON_MIN_PITCHES]
        rows.append(
            {
                "season": season,
                "role": role,
                "people": len(qualified),
                "raw_sd": qualified["effect"].std(ddof=1) * 100,
                "spread": corrected_spread(qualified["effect"], qualified["se"]) * 100,
                "median_se": qualified["se"].median() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "season"])


def trend(subset):
    """Slope of spread against season, with its standard error.

    Seasons are weighted by their bootstrap precision when it is available, so a
    short season with a wide interval does not pull the line as hard as a full
    one. The standard error is scaled by the observed weighted scatter rather
    than taken from the weights alone, which keeps it honest if the spread
    really does move between seasons by more than estimation noise."""
    x, y = subset["season"].to_numpy(float), subset["spread"].to_numpy()
    weights = (
        1 / subset["spread_se"].to_numpy() ** 2
        if "spread_se" in subset and subset["spread_se"].notna().all()
        else np.ones(len(x))
    )

    centre = np.average(x, weights=weights)
    spread_x = (weights * (x - centre) ** 2).sum()
    slope = (weights * (x - centre) * (y - np.average(y, weights=weights))).sum() / spread_x
    intercept = np.average(y, weights=weights) - slope * centre

    residuals = y - (slope * x + intercept)
    scatter = (weights * residuals**2).sum() / (len(x) - 2)
    return slope, np.sqrt(scatter / spread_x)


def figure_spread(spreads, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for role, color in [("umpire", "#c1442f"), ("catcher", "#3b6ea5")]:
        subset = spreads[spreads["role"] == role]
        if "lo" in subset:
            ax.fill_between(subset["season"], subset["lo"], subset["hi"],
                            color=color, alpha=0.15, lw=0)
        ax.plot(subset["season"], subset["spread"], marker="o", color=color, label=role)
        short = subset[subset["season"] == SHORT_SEASON]
        ax.scatter(short["season"], short["spread"], s=90, facecolor="white",
                   edgecolor=color, zorder=3)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("season")
    ax.set_ylabel("spread between people, per 100 called pitches")
    ax.set_title("Noise-corrected spread by season, ±2 SE (hollow = 2020, 60 games)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fit_and_save():
    df = load()
    seasons = sorted(df["season"].unique())
    print(f"{len(df):,} called pitches over {len(seasons)} seasons")

    effects, intervals = [], []
    for season in seasons:
        print(f"fitting {season}: {BOOTSTRAP_REPLICATES} replicates for the effects, "
              f"then {OUTER_REPLICATES}x{INNER_REPLICATES} nested for the spread...")
        effects.append(season_effects(df, season))
        intervals.append(summarise_bootstrap(spread_bootstrap(df, season), season))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat(effects, ignore_index=True).to_parquet(SEASON_EFFECTS, index=False)
    pd.concat(intervals, ignore_index=True).to_parquet(SEASON_SPREADS, index=False)
    print(f"wrote {SEASON_EFFECTS} and {SEASON_SPREADS}\n")


def report():
    effects = pd.read_parquet(SEASON_EFFECTS)
    spreads = spread_by_season(effects).merge(
        pd.read_parquet(SEASON_SPREADS), on=["season", "role"], how="left"
    )
    spreads["lo"] = spreads["spread"] - 2 * spreads["spread_se"]
    spreads["hi"] = spreads["spread"] + 2 * spreads["spread_se"]
    for role in ROLES.values():
        print(f"\n{role}s, spread by season (per 100 called pitches, "
              f"n >= {SEASON_MIN_PITCHES:,})")
        print(
            spreads[spreads["role"] == role]
            .drop(columns=["role"])
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    lift = (spreads["draw_median"] - spreads["spread"]).mean()
    print(f"\nnested bootstrap draws sit {lift:+.3f} above the statistic on average, an "
          "artefact of\nresampling twice; intervals use the width of that distribution, "
          "not its position")

    print("\ntrend in spread per season, 2020 excluded as a 60-game season")
    full = spreads[spreads["season"] != SHORT_SEASON]
    for role in ROLES.values():
        subset = full[full["role"] == role]
        slope, error = trend(subset)
        verdict = "clear" if abs(slope) > 2 * error else "not distinguishable from flat"
        print(f"  {role}s: {subset['spread'].iloc[0]:.3f} -> {subset['spread'].iloc[-1]:.3f}, "
              f"slope {slope:+.4f} +/- {error:.4f} per season ({verdict})")

    FIGURES.mkdir(exist_ok=True)
    figure_spread(spreads, FIGURES / "spread_by_season.png")
    print(f"\nwrote {FIGURES / 'spread_by_season.png'}")


def main(refit):
    if refit or not (SEASON_EFFECTS.exists() and SEASON_SPREADS.exists()):
        fit_and_save()
    else:
        print(f"reusing {SEASON_EFFECTS}; pass --refit to fit again\n")
    report()


def _self_check():
    rng = np.random.default_rng(0)

    # A season's worth of estimates is the truth plus its own error. The raw SD
    # reads high by exactly that error; the correction has to take it back out.
    truth = rng.normal(0, 0.010, 4_000)
    for noise in (0.004, 0.012):
        estimates = truth + rng.normal(0, noise, len(truth))
        se = np.full(len(truth), noise)
        raw = estimates.std(ddof=1)
        assert raw > 0.0105, raw
        assert abs(corrected_spread(estimates, se) - 0.010) < 0.001, noise

    # Noisier seasons must not look more variable once corrected, which is the
    # whole trap: 2020 is a third the length of the others.
    quiet = truth + rng.normal(0, 0.004, len(truth))
    loud = truth + rng.normal(0, 0.012, len(truth))
    assert loud.std(ddof=1) > quiet.std(ddof=1) * 1.2
    assert abs(
        corrected_spread(loud, np.full(len(truth), 0.012))
        - corrected_spread(quiet, np.full(len(truth), 0.004))
    ) < 0.001

    # All noise and no signal clamps at zero rather than returning a NaN.
    assert corrected_spread(rng.normal(0, 0.005, 500), np.full(500, 0.05)) == 0.0

    # A flat series must not be reported as a trend, and a real slope must be.
    seasons = pd.DataFrame({"season": np.arange(2015, 2025)})
    flat = seasons.assign(spread=1.0 + rng.normal(0, 0.05, 10))
    sloped = seasons.assign(spread=1.0 - 0.05 * np.arange(10) + rng.normal(0, 0.05, 10))
    flat_slope, flat_error = trend(flat)
    real_slope, real_error = trend(sloped)
    assert abs(flat_slope) < 2 * flat_error, (flat_slope, flat_error)
    assert abs(real_slope) > 2 * real_error, (real_slope, real_error)

    # Weighting must actually bite: one wildly imprecise season pulls the
    # unweighted line and should barely move the weighted one.
    outlier = sloped.copy()
    outlier.loc[0, "spread"] = 3.0
    tight = np.full(10, 0.02)
    tight[0] = 5.0
    assert abs(trend(outlier)[0] - real_slope) > 0.05, "unweighted fit should be dragged"
    weighted = trend(outlier.assign(spread_se=tight))[0]
    assert abs(weighted - real_slope) < 0.02, (weighted, real_slope)

    # An all-NaN error column falls back to equal weights rather than failing.
    assert trend(sloped.assign(spread_se=np.nan))[0] == real_slope
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check"]:
        _self_check()
    else:
        main(refit="--refit" in args)
