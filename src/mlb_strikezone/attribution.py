# ABOUTME: Attributes out-of-fold called-strike residuals jointly to home-plate
# ABOUTME: umpires and catchers, and writes the leaderboards to data/processed/.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaseball import cache, playerid_reverse_lookup
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

cache.enable()

PROCESSED = Path("data/processed")
PREDICTIONS = PROCESSED / "predictions.parquet"
UMPIRES = PROCESSED / "umpires.parquet"
ATTRIBUTION = PROCESSED / "attribution.parquet"
FIGURES = Path("figures")

# Ridge shrinks each effect toward zero by roughly n/(n + ALPHA), so ALPHA is
# read in pitches: someone with ALPHA called pitches keeps about half of their
# raw residual. Set at the leaderboard sample floor from CLAUDE.md, which is the
# point below which a number is not worth believing on its own.
ALPHA = 2000
MIN_PITCHES = 2000

ROLES = {"ump_id": "umpire", "fielder_2": "catcher"}


def load():
    """One row per called pitch with its out-of-fold residual, umpire and catcher."""
    pitches = pd.read_parquet(PREDICTIONS)
    umpires = pd.read_parquet(UMPIRES, columns=["game_pk", "ump_id", "ump_name"])

    before = len(pitches)
    df = pitches.merge(umpires, on="game_pk", how="inner")
    if len(df) != before:
        raise RuntimeError(f"join changed the pitch count: {before:,} -> {len(df):,}")

    # Positive means called a strike more often than the pitch deserved.
    df["residual"] = df["is_strike"].astype("float32") - df["p_gbm"]
    return df


def fit_joint(df):
    """One ridge fit with umpire and catcher effects side by side.

    Fitting them separately would double-count: a catcher's framing sits in the
    residual of every umpire he works with, and every umpire's tendency sits in
    the residual of every catcher. Side by side they compete for the same
    variance, so each effect is net of the other.

    ponytail: linear on residuals, not logistic with an offset. The residual is
    heteroscedastic so this is not the efficient estimator, but it is the
    interpretable one — coefficients are already in called strikes per pitch —
    and it fits 3.7M rows in seconds. Revisit if the standard errors matter."""
    encoder = OneHotEncoder(dtype=np.float32)
    design = encoder.fit_transform(df[list(ROLES)])
    model = Ridge(alpha=ALPHA).fit(design, df["residual"])

    effects = []
    for (column, role), categories, coefficients in zip(
        ROLES.items(), encoder.categories_, np.split(model.coef_, [len(encoder.categories_[0])])
    ):
        effects.append(pd.DataFrame({"role": role, "id": categories, "effect": coefficients}))
    return pd.concat(effects, ignore_index=True)


def fit_marginal(df):
    """Each role fitted alone, with the same shrinkage as the joint fit.

    This is the honest comparison for what fitting jointly buys. Against the raw
    mean, most of the movement is just ridge pulling small samples toward zero,
    which has nothing to do with controlling for the partner; against this, the
    shrinkage is identical and the only difference left is the partner."""
    effects = []
    for column, role in ROLES.items():
        encoder = OneHotEncoder(dtype=np.float32)
        model = Ridge(alpha=ALPHA).fit(encoder.fit_transform(df[[column]]), df["residual"])
        effects.append(
            pd.DataFrame({"role": role, "id": encoder.categories_[0], "marginal": model.coef_})
        )
    return pd.concat(effects, ignore_index=True)


def naive_effects(df):
    """Mean residual per person, ignoring who they worked with. Only kept as the
    comparison that shows what the joint fit is buying."""
    rows = []
    for column, role in ROLES.items():
        grouped = df.groupby(column)["residual"].agg(["mean", "size"])
        rows.append(
            pd.DataFrame(
                {"role": role, "id": grouped.index, "naive": grouped["mean"].to_numpy(),
                 "pitches": grouped["size"].to_numpy()}
            )
        )
    return pd.concat(rows, ignore_index=True)


def add_names(effects, df):
    """Umpire names ride along with the umpire table; catchers are MLBAM ids."""
    umpire_names = (
        df[["ump_id", "ump_name"]].drop_duplicates().set_index("ump_id")["ump_name"]
    )
    catcher_ids = effects.loc[effects["role"] == "catcher", "id"].astype(int).tolist()
    people = playerid_reverse_lookup(catcher_ids, key_type="mlbam")
    catcher_names = (
        people.assign(name=people["name_first"].str.title() + " " + people["name_last"].str.title())
        .set_index("key_mlbam")["name"]
    )

    names = effects["id"].map(catcher_names).where(effects["role"] == "catcher")
    return effects.assign(name=names.fillna(effects["id"].map(umpire_names)).fillna(effects["id"]))


def leaderboard(effects, role, min_pitches=MIN_PITCHES):
    board = effects[(effects["role"] == role) & (effects["pitches"] >= min_pitches)].copy()
    # Per 100 called pitches is the readable unit; the total is the same number
    # expressed as how many calls it actually moved over the whole sample.
    board["per_100"] = board["effect"] * 100
    board["extra_strikes"] = board["effect"] * board["pitches"]
    board["alone_per_100"] = board["marginal"] * 100
    return board.sort_values("per_100", ascending=False)[
        ["name", "pitches", "per_100", "extra_strikes", "alone_per_100"]
    ]


def figure_joint_vs_marginal(effects, path):
    """How much of a ranking is really the partners you drew.

    Both axes carry the same ridge shrinkage, so distance from the diagonal is
    the partner correction alone, not a mix of that and regularisation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = 0
    for role, color in [("umpire", "#c1442f"), ("catcher", "#3b6ea5")]:
        subset = effects[(effects["role"] == role) & (effects["pitches"] >= MIN_PITCHES)]
        ax.scatter(subset["marginal"] * 100, subset["effect"] * 100, s=14, color=color,
                   alpha=0.7, label=f"{role}s (n={len(subset)})")
        limit = max(limit, subset["marginal"].abs().max() * 100, subset["effect"].abs().max() * 100)

    edge = limit * 1.15
    ax.plot([-edge, edge], [-edge, edge], color="black", ls="--", lw=1, label="unchanged")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)
    ax.set_xlabel("fitted alone, per 100 called pitches")
    ax.set_ylabel("fitted jointly, per 100 called pitches")
    ax.set_title("What controlling for the other party changes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = load()
    print(f"{len(df):,} called pitches, {df['ump_id'].nunique()} umpires, "
          f"{df['fielder_2'].nunique()} catchers")
    print(f"mean residual overall: {df['residual'].mean():+.5f}\n")

    effects = (
        fit_joint(df)
        .merge(fit_marginal(df), on=["role", "id"])
        .merge(naive_effects(df), on=["role", "id"])
    )
    effects = add_names(effects, df)

    for role in ROLES.values():
        board = leaderboard(effects, role)
        print(f"\n{role}s, most strike-friendly (n >= {MIN_PITCHES:,} called pitches, "
              f"{len(board)} qualify)")
        print(board.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n{role}s, least strike-friendly")
        print(board.tail(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nwhat each step does, per 100 called pitches, for qualifying people")
    qualified = effects[effects["pitches"] >= MIN_PITCHES]
    for role in ROLES.values():
        subset = qualified[qualified["role"] == role]
        shrinkage = (subset["naive"] - subset["marginal"]).abs().mean() * 100
        partner = (subset["marginal"] - subset["effect"]).abs().mean() * 100
        moved = (subset["effect"].abs() < subset["marginal"].abs()).mean()
        print(f"  {role}s: shrinkage moves {shrinkage:.3f}, controlling for the partner "
              f"moves {partner:.3f} ({moved:.0%} shrink further)")

    FIGURES.mkdir(exist_ok=True)
    figure_joint_vs_marginal(effects, FIGURES / "joint_vs_marginal_attribution.png")
    effects.to_parquet(ATTRIBUTION, index=False)
    print(f"\nwrote {ATTRIBUTION} and {FIGURES / 'joint_vs_marginal_attribution.png'}")


def _self_check():
    # Umpire A works catcher X most of the time, so a per-umpire mean cannot tell
    # A's strike-friendliness apart from X's framing. Both are +5 per 100 here,
    # and a naive mean should read roughly their sum for the pair.
    rng = np.random.default_rng(0)
    n = 400_000
    umps, catchers = np.array([0, 1, 2]), np.array([10, 11, 12])
    ump = rng.choice(umps, n)
    paired = rng.random(n) < 0.8
    catcher = np.where(paired, catchers[ump], rng.choice(catchers, n))

    ump_effect = np.array([0.05, 0.0, -0.05])
    catcher_effect = np.array([0.05, 0.0, -0.05])
    truth = ump_effect[ump] + catcher_effect[catchers.searchsorted(catcher)]
    df = pd.DataFrame(
        {
            "ump_id": ump,
            "fielder_2": catcher,
            "residual": (truth + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )

    joint = fit_joint(df).set_index(["role", "id"])["effect"]
    naive = naive_effects(df).set_index(["role", "id"])["naive"]
    marginal = fit_marginal(df).set_index(["role", "id"])["marginal"]

    # The joint fit recovers each effect; the naive mean inflates it by soaking
    # up the partner's, which is the entire reason for fitting them together.
    assert abs(joint[("umpire", 0)] - 0.05) < 0.01, joint[("umpire", 0)]
    assert abs(joint[("catcher", 10)] - 0.05) < 0.01, joint[("catcher", 10)]
    assert naive[("umpire", 0)] > 0.08, naive[("umpire", 0)]
    assert abs(joint[("umpire", 1)]) < 0.01, joint[("umpire", 1)]

    # Shrinkage pulls toward zero, never past it into the wrong sign.
    assert 0 < joint[("umpire", 0)] <= naive[("umpire", 0)]
    assert naive[("umpire", 2)] <= joint[("umpire", 2)] < 0

    # Fitting alone carries the same shrinkage but not the partner control, so it
    # keeps the contamination the joint fit removes. Without this the figure
    # comparing the two would be measuring regularisation, not confounding.
    assert marginal[("umpire", 0)] > joint[("umpire", 0)] + 0.02, marginal[("umpire", 0)]
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _self_check()
    else:
        main()
