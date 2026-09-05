"""How accurate could any model be on this compilation?

The compilation reports, for some systems, more than one measurement made under
the same stated conditions. Those repeats bound what a model can achieve: a
predictor that recovered the consensus value exactly would still be scored
against one noisy report and would still register an error. This module turns
the repeats into that bound and writes `noise_floor.csv`. No model is loaded,
trained or rescored, and nothing here is fitted.

Construction. Metal-complex records are grouped on the full condition key
(ligand, metal, oxidation state, temperature, ionic strength, background
electrolyte). Groups holding at least two records give a repeatability
variance in the usual pooled way, sum of within-group squared deviations over
the total degrees of freedom, which is distribution-free. The mean absolute
deviation is taken from the same deviations after scaling each group's by
sqrt(n/(n-1)), so that a single scaled deviation has the variance of one
report's error without assuming the errors are Gaussian; that matters here,
because they are emphatically not.

Two brackets are reported rather than one number. The inclusive estimate uses
every group. A handful of groups disagree by many log units and are more
plausibly transcription or assignment errors upstream than measurement scatter;
because a variance is dominated by them, the trimmed estimate repeats the
calculation on the groups whose full range is at most one log unit, which is
88% of them. The truth for a well-behaved system is near the trimmed value and
the truth for the compilation as a whole is near the inclusive one, so quoting
both is more honest than choosing.

Two directions of bias are worth stating and neither is corrected here.
Systems that were remeasured are not a random sample of the compilation, and
records that agree to the digit may be re-entries of one primary source rather
than independent determinations, which would make the floor an underestimate.
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

T = os.environ.get("BNN_TABLES", os.path.join("manuscripts", "derived", "tables")) + os.sep
DS1 = os.path.join("data", "processed", "ds1.csv")

TRIM_RANGE = 1.0     # a group wider than this is treated as a probable data error
MIN_REPLICATES = 2


def _pooled(groups):
    """Repeatability SD and mean absolute error of one report, from repeats."""
    n = np.array([len(g) for g in groups])
    ss = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sd = float(np.sqrt(ss / (n - 1).sum()))
    dev = np.concatenate([(g - g.mean()) * np.sqrt(len(g) / (len(g) - 1.0))
                          for g in groups])
    return sd, float(np.abs(dev).mean()), int(len(groups)), int(n.sum())


def noise_floor():
    d = pd.read_csv(DS1, usecols=["logK", "experiment_key", "is_proton", "cmi_source"])
    mc = d[~d.is_proton]
    groups = [g.values for _, g in mc.groupby("experiment_key").logK
              if len(g) >= MIN_REPLICATES]
    trimmed = [g for g in groups if g.max() - g.min() <= TRIM_RANGE]

    # Selectivity is a difference of two condition-cell consensus values, so its
    # noise variance is sigma^2 (1/n_x + 1/n_y). Most cells hold one record, and
    # the expectation below is taken over the empirical distribution of cell
    # sizes rather than assumed. Using sigma^2/n for the median of n records is
    # an approximation, exact only at n = 1 and n = 2, and n > 2 is rare.
    sizes = mc.groupby("experiment_key").size().values
    e_inv_n = float(np.mean(1.0 / sizes))
    pair_factor = float(np.sqrt(2.0 * e_inv_n))

    var_test = float(d[d.cmi_source == "test"].logK.var(ddof=1))
    rows = []
    for label, gs in (("inclusive, every repeated group", groups),
                      (f"trimmed, group range <= {TRIM_RANGE:g}", trimmed)):
        sd, mad, ng, nr = _pooled(gs)
        rows.append(dict(
            estimate=label, n_groups=ng, n_records=nr,
            share_of_groups=ng / len(groups),
            logk_rmse_floor=sd,
            logk_mae_floor=mad,
            logk_r2_ceiling=1.0 - sd ** 2 / var_test,
            selectivity_rmse_floor=sd * pair_factor,
            target_variance_published_test=var_test,
            pair_noise_factor=pair_factor,
            mean_inverse_cell_size=e_inv_n))
    return pd.DataFrame(rows)


def _provenance(name, note):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unavailable"
    json.dump(dict(table=name, inputs=[DS1], git_commit=commit, note=note,
                   trim_range=TRIM_RANGE, min_replicates=MIN_REPLICATES),
              open(T + name + ".provenance.json", "w"), indent=2)


if __name__ == "__main__":
    f = noise_floor()
    f.to_csv(T + "noise_floor.csv", index=False)
    _provenance("noise_floor.csv",
                "repeatability of the compilation, and the error floor it sets "
                "for absolute log K and for the pairwise selectivity target")
    for _, r in f.iterrows():
        print(f"{r.estimate}")
        print(f"   groups {r.n_groups} ({100 * r.share_of_groups:.1f}%), "
              f"records {r.n_records}")
        print(f"   log K   RMSE floor {r.logk_rmse_floor:.3f}   "
              f"MAE floor {r.logk_mae_floor:.3f}   "
              f"R2 ceiling {r.logk_r2_ceiling:.4f}")
        print(f"   dlog K  RMSE floor {r.selectivity_rmse_floor:.3f}  "
              f"(x{r.pair_noise_factor:.3f} for a difference of two cells)")
