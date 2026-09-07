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
    ROLE_COLOURS,
    ROLES,
    UMPIRES,
    bootstrap_standard_errors,
    build_design,
    cluster_robust_se,
    fit_joint,
    games_as_row_blocks,
    role_bounds,
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

# Per role, because the roles do not accumulate called pitches at the same rate.
# An umpire works a whole game and a catcher most of one; a starting pitcher is
# behind maybe a fifth of the taken pitches in his. Holding all three to the same
# floor empties the pitcher column in 2020 entirely.
SEASON_MIN_PITCHES = {"umpire": 1000, "catcher": 1000, "pitcher": 500}
BOOTSTRAP_REPLICATES = 100

# The interval on a season's spread resamples games once. Correcting a
# replicate's spread needs the standard errors of its effects, and those come
# from the closed-form clustered estimator rather than a second bootstrap, which
# is both faster and applies the identical estimator at both levels.
#
# The correction is doubled inside a replicate, and that is the part that
# matters. A replicate is noisy about the observed effects, which are themselves
# noisy about the truth, so it carries two helpings of estimation error. Taking
# out only one leaves the draws sitting on the raw standard deviation, about
# 0.11 above the statistic, far enough that every point estimate landed outside
# its own percentile bounds.
OUTER_REPLICATES = 300
INTERVAL = (5, 95)

SHORT_SEASON = 2020


def load():
    pitches = pd.read_parquet(PREDICTIONS)
    umpires = pd.read_parquet(UMPIRES, columns=["game_pk", "game_date", "ump_id"])
    df = pitches.merge(umpires, on="game_pk", how="inner")
    df["season"] = df["game_date"].str.slice(0, 4).astype(int)
    df["residual"] = df["is_strike"].astype("float32") - df["p_gbm"]
    return df


def corrected_spread(effect, se, doses=1):
    """Standard deviation of the true effects, with estimation noise removed.

    The observed spread of estimates carries its own standard errors on top of
    the real variation, and carries more of them in a season with less data.
    Subtracting the mean squared error is what stops the shortest season looking
    like the most variable one purely because it is the shortest.

    `doses` is how many helpings of that noise the input carries. Real estimates
    carry one. A bootstrap replicate carries two — the sampling noise already in
    the observed effects, plus the noise of resampling them — so subtracting one
    from a replicate leaves a whole helping behind, and the bootstrap converges
    on the raw standard deviation rather than on the statistic itself."""
    # A spread needs two people to exist. Without this guard a thin slice still
    # returns NaN, but through a pile of numpy warnings rather than by saying so.
    if len(effect) < 2:
        return float("nan")
    variance = effect.var(ddof=1) - doses * np.mean(np.asarray(se) ** 2)
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

    extra = []
    for column, role in ROLES.items():
        sized = slice_.groupby(column).size()
        # The spread correction uses this closed-form error, not the bootstrap
        # one above, so that the point estimate and the replicates behind its
        # interval are computed the same way. The bootstrap column stays as the
        # check on it.
        analytic = cluster_robust_se(
            slice_[column].to_numpy(), slice_["game_pk"].to_numpy(),
            slice_["residual"].to_numpy(),
        )
        extra.append(
            pd.DataFrame(
                {
                    "role": role,
                    "id": sized.index,
                    "pitches": sized.to_numpy(),
                    "se_analytic": analytic.reindex(sized.index).to_numpy(),
                }
            )
        )
    return effects.merge(pd.concat(extra, ignore_index=True), on=["role", "id"]).assign(
        season=season
    )


def _resample(blocks, picked):
    return np.concatenate([blocks[game] for game in picked])


def spread_bootstrap(df, season, outer=OUTER_REPLICATES, seed=0):
    """Bootstrap distribution of each role's corrected spread for one season.

    Each replicate resamples games, refits, and computes the standard errors it
    needs in closed form from that same resample — the identical estimator the
    point estimate uses, which is what keeps the distribution centred on it."""
    slice_ = df[df["season"] == season]
    design, encoder = build_design(slice_)
    residual = slice_["residual"].to_numpy()
    blocks = games_as_row_blocks(slice_)
    lengths = np.array([len(block) for block in blocks])
    bounds = role_bounds(encoder)
    roles = {
        role: (bounds[role], slice_[column].to_numpy(), categories)
        for (column, role), categories in zip(ROLES.items(), encoder.categories_)
    }
    rng = np.random.default_rng(seed)

    draws = {role: np.empty(outer) for role in roles}
    for replicate in range(outer):
        picked = rng.integers(0, len(blocks), len(blocks))
        rows = _resample(blocks, picked)
        effect = Ridge(alpha=SEASON_ALPHA).fit(design[rows], residual[rows]).coef_
        # One-hot columns sum to each person's pitch count, so the qualification
        # floor can be reapplied per replicate without touching the frame.
        counts = np.asarray(design[rows].sum(axis=0)).ravel()
        # A game drawn twice is two clusters, not one, so the cluster label is
        # the draw's position rather than the game it came from.
        cluster = np.repeat(np.arange(len(picked)), lengths[picked])

        for role, (span, people, categories) in roles.items():
            se = (
                cluster_robust_se(people[rows], cluster, residual[rows])
                .reindex(categories)
                .to_numpy()
            )
            keep = counts[span] >= SEASON_MIN_PITCHES[role]
            # Two doses: a replicate is noisy about the observed effects, which
            # are themselves noisy about the truth.
            draws[role][replicate] = corrected_spread(effect[span][keep], se[keep], doses=2)
    return draws


def summarise_bootstrap(draws, season):
    """Width and percentile bounds, plus where the distribution sits relative to
    the statistic so any residual shift stays visible rather than absorbed."""
    rows = []
    for role, values in draws.items():
        rows.append(
            {
                "season": season,
                "role": role,
                "spread_se": values.std(ddof=1) * 100,
                "draw_median": np.median(values) * 100,
                "lo": np.percentile(values, INTERVAL[0]) * 100,
                "hi": np.percentile(values, INTERVAL[1]) * 100,
            }
        )
    return pd.DataFrame(rows)


def spread_by_season(effects):
    rows = []
    for (season, role), group in effects.groupby(["season", "role"]):
        qualified = group[group["pitches"] >= SEASON_MIN_PITCHES[role]]
        rows.append(
            {
                "season": season,
                "role": role,
                "people": len(qualified),
                "raw_sd": qualified["effect"].std(ddof=1) * 100,
                "spread": corrected_spread(qualified["effect"], qualified["se_analytic"]) * 100,
                "median_se": qualified["se_analytic"].median() * 100,
                # Ratio of the closed-form error to the bootstrap one. Around
                # 1.07 is expected: the bootstrap sees a shrunken estimator and
                # the closed form describes a plain mean.
                "se_ratio": (qualified["se_analytic"] / qualified["se"]).median(),
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
    for role, color in ROLE_COLOURS.items():
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
    ax.set_title(
        f"Noise-corrected spread by season, {INTERVAL[1] - INTERVAL[0]}% interval "
        "(hollow = 2020, 60 games)"
    )
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
              f"then {OUTER_REPLICATES} for the spread interval...")
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
    for role in ROLES.values():
        print(f"\n{role}s, spread by season (per 100 called pitches, "
              f"n >= {SEASON_MIN_PITCHES[role]:,})")
        print(
            spreads[spreads["role"] == role]
            .drop(columns=["role"])
            .to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )

    spreads = spreads.dropna(subset=["spread", "lo", "hi"])
    lift = (spreads["draw_median"] - spreads["spread"]).mean()
    outside = ((spreads["spread"] < spreads["lo"]) | (spreads["spread"] > spreads["hi"])).sum()
    print(f"\nbootstrap draws sit {lift:+.3f} from the statistic on average; "
          f"{outside} of {len(spreads)} point estimates fall outside their own interval")
    print(f"closed-form errors run {spreads['se_ratio'].median():.2f}x the bootstrap ones, "
          "the gap being the shrinkage the bootstrap sees and a plain mean does not")

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

    # A slice too thin to have a spread says so quietly. 2020 has no pitcher
    # clearing a full season's floor, and numpy would otherwise warn per call.
    assert np.isnan(corrected_spread(np.array([]), np.array([])))
    assert np.isnan(corrected_spread(np.array([0.01]), np.array([0.001])))

    # A bootstrap replicate is noisy about the observed effects, which are
    # already noisy about the truth, so it carries two helpings of error. The
    # single correction leaves one behind and lands on the raw standard
    # deviation; only the double correction returns to the statistic.
    noise = 0.006
    se = np.full(len(truth), noise)
    observed = truth + rng.normal(0, noise, len(truth))
    replicate = observed + rng.normal(0, noise, len(truth))

    statistic = corrected_spread(observed, se)
    assert abs(corrected_spread(replicate, se, doses=2) - statistic) < 0.0005, statistic
    single = corrected_spread(replicate, se, doses=1)
    assert abs(single - observed.std(ddof=1)) < 0.0005, single
    # The leftover helping adds in quadrature, not linearly.
    assert single > statistic
    assert abs(single - np.hypot(statistic, noise)) < 0.0005, (single, statistic)

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
