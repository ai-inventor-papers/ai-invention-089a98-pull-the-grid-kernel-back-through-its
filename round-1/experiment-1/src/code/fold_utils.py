"""Fold-cell construction for the mechanism factorial and the controls, plus their
aggregation."""
from __future__ import annotations

import numpy as np

from folds import make_fold, fold_fingerprint, SEEDS, SCHEMES


def _tasks(metric_name, regime, ns, targets, operator, kind, extra):
    """Build one task per fold, CARRYING THE OPERATOR into the payload.

    The operator must travel with the task: the worker falls back to the primary
    operator when none is supplied, so omitting it here silently turns every metric
    cell into a repeat of the treatment.  That exact bug produced a full mechanism
    factorial in which every row was numerically identical -- the failure signature
    the artifact plan names ('a feature map is being reused by reference rather than
    rebuilt').  `mechanism_operator_present` is asserted by the runner.
    """
    if operator is None:
        raise ValueError(f"metric {metric_name!r} supplied no operator")
    op = np.ascontiguousarray(np.asarray(operator, dtype=float))
    if not np.isfinite(op).all():
        raise ValueError(f"metric {metric_name!r} operator is not finite")
    out = []
    for n in ns:
        for seed in SEEDS:
            for scheme in SCHEMES:
                try:
                    tr, te = make_fold(n, seed, scheme)
                except AssertionError:
                    continue
                t = dict(regime=regime, n=int(n), seed=int(seed), scheme=int(scheme),
                         train=tr, test=te, fingerprint=fold_fingerprint(tr, te),
                         metric=metric_name, target=targets[0], kind=kind,
                         operator=op)
                if extra:
                    t.update(extra)
                out.append(t)
    return out


def mechanism_cells(metric_name, regime, ns, targets, operator, kind="metric",
                    extra=None, n_draws: int = 220) -> list[dict]:
    if n_draws != 220:
        return _tasks_ndraws(metric_name, regime, ns, targets[0], operator, kind,
                             extra, n_draws)
    return _tasks(metric_name, regime, ns, targets, operator, kind, extra)


def _tasks_ndraws(metric_name, regime, ns, target, operator, kind, extra, n_draws):
    """Same recipe on a smaller draw pool (the near-end sheet has 120 rows, not 220).

    The near-end sheet is a separate 120-draw AC ensemble, so its folds are drawn from
    its own pool with the SAME seed/scheme recipe rather than by filtering the 220-row
    folds -- filtering would leave too few test rows at n = 10 and would silently change
    the scheme-1 fixed-block size.
    """
    if operator is None:
        raise ValueError(f"metric {metric_name!r} supplied no operator")
    op = np.ascontiguousarray(np.asarray(operator, dtype=float))
    out = []
    for n in ns:
        for seed in SEEDS:
            for scheme in SCHEMES:
                try:
                    tr, te = make_fold(n, seed, scheme, n_draws)
                except AssertionError:
                    continue
                t = dict(regime=regime, n=int(n), seed=int(seed), scheme=int(scheme),
                         train=tr, test=te, fingerprint=fold_fingerprint(tr, te),
                         metric=metric_name, target=target, kind=kind, operator=op,
                         n_draws=int(n_draws))
                if extra:
                    t.update(extra)
                out.append(t)
    return out


def assert_metrics_differ(tasks: list[dict]) -> dict:
    """Refuse to run a factorial whose metric cells carry the same operator.

    The check is on the OPERATOR, not on the resulting RMSE, so it fires before the
    expensive run rather than after it.  Two metrics may legitimately coincide (a
    rank truncation above the effective rank is inert); they may never be the SAME
    ARRAY, which is what a lost payload looks like.
    """
    seen = {}
    for t in tasks:
        op = t.get("operator")
        if op is None:
            raise AssertionError(f"{t['metric']} task carries no operator")
        key = (op.shape, float(np.linalg.norm(op)))
        seen.setdefault(key, set()).add(t["metric"])
    dupes = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    return dict(n_distinct_operator_shapes_and_norms=len(seen), collisions=dupes,
                ok=bool(not any(len(v) > 1 and len(set(v)) > 1 and
                                len({m for m in v}) > 1 for v in [list(x) for x in
                                                                  seen.values()])))


def control_cells(control, regime, ns) -> list[dict]:
    out = []
    for n in ns:
        for seed in SEEDS:
            for scheme in SCHEMES:
                try:
                    tr, te = make_fold(n, seed, scheme)
                except AssertionError:
                    continue
                out.append(dict(regime=regime, n=int(n), seed=int(seed),
                                scheme=int(scheme), train=tr, test=te,
                                fingerprint=fold_fingerprint(tr, te),
                                control=control, target="far",
                                metric="reduction"))
    return out


def _agg(recs) -> dict:
    ok = [r for r in recs if "error" not in r and "rmse" in r]
    if not ok:
        return dict(n_folds=0)
    rm = np.array([r["rmse"] for r in ok])
    rat = np.array([r.get("rmse_over_mean", np.nan) for r in ok])
    out = dict(n_folds=len(ok), mean_rmse=float(np.nanmean(rm)),
               worst_fold_rmse=float(np.nanmax(rm)),
               mean_rmse_over_mean=float(np.nanmean(rat)),
               mean_ell=float(np.mean([r.get("ell", np.nan) for r in ok])),
               per_fold_rmse=[float(v) for v in rm])
    if "nlpd" in ok[0]:
        out["mean_nlpd"] = float(np.mean([r["nlpd"] for r in ok]))
        out["mean_coverage68"] = float(np.mean([r["coverage68"] for r in ok]))
        out["mean_coverage95"] = float(np.mean([r["coverage95"] for r in ok]))
    return out


def aggregate_mechanism(recs) -> dict:
    """Nest every cell as metric -> target -> n -> summary."""
    buckets = {}
    for r in recs:
        # the worker writes BOTH `target` (the operator's target sheet name) and
        # `targets` (the same sheet's short tag).  Reading only one of them silently
        # merged the near-target cells into the far-target bucket, which corrupted the
        # mechanism verdict's far/full ratio.  Read either, preferring `target`.
        key = (r["metric"], r.get("target") or r.get("targets") or "far")
        buckets.setdefault(key, {}).setdefault(int(r["n"]), []).append(r)
    out = {}
    for (metric, target), by_n in buckets.items():
        out.setdefault(metric, {})[target] = {int(n): _agg(v) for n, v in sorted(by_n.items())}
    return out


def aggregate_controls(recs) -> dict:
    """Nest every control as control -> n -> summary, and form the ratio to the
    primary treatment's own error on the identical folds."""
    buckets = {}
    for r in recs:
        c = r.get("control") or r.get("metric")
        buckets.setdefault(c, {}).setdefault(int(r["n"]), []).append(r)
    out = {}
    for c, by_n in buckets.items():
        out[c] = {int(n): _agg(v) for n, v in sorted(by_n.items())}
    return out


def relative_to_treatment(control_summary: dict, treatment_summary: dict) -> dict:
    """How much of the treatment's advantage each control gives back, at matched n."""
    out = {}
    for c, by_n in control_summary.items():
        out[c] = {}
        for n, s in by_n.items():
            t = treatment_summary.get(n) or treatment_summary.get(str(n))
            if not t or not s.get("n_folds"):
                continue
            base = t.get("mean_far_end_rmse") or t.get("mean_rmse")
            if not base:
                continue
            out[c][n] = dict(control_rmse=s["mean_rmse"],
                             treatment_rmse=float(base),
                             ratio_to_treatment=float(s["mean_rmse"] / base))
    return out
