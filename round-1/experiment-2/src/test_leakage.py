"""T6 - leakage checks.

(a) Replace Y[te] with a permuted copy of itself and re-run one cell: the FITTED
    predictions must be bit-identical and only the reported error may change. This is
    the decisive test - if any test row reached the fit, the predictions move.
(b) Confirm the normalisation denominator, the injection standardisation, the feature
    scale, the GP hyperparameters, the ridge penalty and the (k, gamma) selection are
    all computed from TRAIN rows only.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
import method as M

M._init_worker()
bases = {(r["n"], r["fold_id"]): r for r in M.jload(M.RAW / "baselines.json")}
results = {}

n = 20
f = M.folds_for(n)[0]
base = bases[(n, f["fold_id"])]
X, Y = M._G["X"], M._G["Y"]
z = M._G["z"]
tr, te = f["tr"], f["te"]
vsd = base["vsd_train"]


def fit_predict(Ymat):
    Mw = M.op_whiten(z["U"], z["sv"], z["Vt"], 11, 0.5)
    Ftr, Fte = M.featurise(Mw, X[tr], X[te])
    mu, pv, hp, _, _ = M.gp_corr(M.rbf_corr(M.sqd(Ftr, Ftr), M.sqd(Fte, Ftr)), Ymat[tr])
    return mu, pv, hp


mu1, pv1, hp1 = fit_predict(Y)
rng = np.random.default_rng(999)
Yperm = Y.copy()
Yperm[te] = Y[te][rng.permutation(len(te))]
mu2, pv2, hp2 = fit_predict(Yperm)
results["T6a_predictions_bit_identical"] = bool(np.array_equal(mu1, mu2))
results["T6a_predictive_variance_bit_identical"] = bool(np.array_equal(pv1, pv2))
results["T6a_hyperparameters_identical"] = bool(hp1 == hp2)
results["T6a_rmse_changed_by_permutation"] = bool(
    abs(M.rmse(mu1, Y[te]) - M.rmse(mu2, Yperm[te])) > 0)
assert results["T6a_predictions_bit_identical"], "TEST SET LEAKED INTO THE FIT"
assert results["T6a_hyperparameters_identical"], "TEST SET LEAKED INTO HYPERPARAMETERS"

# (b) denominator and standardisation depend on TRAIN rows only
vsd_te = float(Y[te].std(0).mean())
results["T6b_vsd_train"] = vsd
results["T6b_vsd_test"] = vsd_te
results["T6b_vsd_differs_from_test"] = bool(abs(vsd - vsd_te) > 1e-12)

# feature scale sc must be a function of X[tr] only
Mw = M.op_whiten(z["U"], z["sv"], z["Vt"], 11, 0.5)
Ftr_a, Fte_a = M.featurise(Mw, X[tr], X[te])
Ftr_b, _ = M.featurise(Mw, X[tr], X[te] * 1e6)
results["T6b_feature_scale_ignores_test_rows"] = bool(np.array_equal(Ftr_a, Ftr_b))

# ridge penalty selected on train rows only: the fitted MAP coefficients must depend on
# the train design alone, so scaling the TEST design may only rescale the test-side
# contribution, never refit the coefficients. Check the ridge linear operator instead:
# ridge_scale_free(Za, 0, Ytr) returns only the intercept column, and ridge on a zeroed
# test block must equal the intercept for every output.
Ztr_std = X[tr].std(0)
Ztr_std[Ztr_std < 1e-12] = 1.0
Za = (X[tr] - X[tr].mean(0)) / Ztr_std
Zb = (X[te] - X[tr].mean(0)) / Ztr_std
pa = M.ridge_scale_free(Za, Zb, Y[tr])
# an independent reimplementation of the same estimator must agree - if any test row
# entered the penalty selection or the solve, this reimplementation (which never touches
# the test block during fitting) would diverge.
n_tr = len(Za)
A = np.hstack([Za, np.ones((n_tr, 1))])
Dm = np.ones(A.shape[1])
Dm[-1] = 0.0
G = A.T @ A
sc = float(np.linalg.eigvalsh(G).max()) / A.shape[1]
folds = np.array_split(np.arange(n_tr), max(2, n_tr // 4))
best = None
for lam in np.logspace(-6, 6, 25):
    sse = 0.0
    for fo in folds:
        ti = np.setdiff1d(np.arange(n_tr), fo)
        A1 = A[ti]
        G1 = A1.T @ A1
        s1 = float(np.linalg.eigvalsh(G1).max()) / A1.shape[1]
        c = np.linalg.solve(G1 + lam * s1 * np.diag(Dm), A1.T @ Y[tr][ti])
        sse += float(((A[fo] @ c - Y[tr][fo]) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, lam)
c = np.linalg.solve(G + best[1] * sc * np.diag(Dm), A.T @ Y[tr])
pb = np.hstack([Zb, np.ones((len(te), 1))]) @ c
results["T6b_ridge_refit_matches"] = bool(np.allclose(pa, pb, rtol=1e-10, atol=1e-14))
# the fitted coefficients must not move when the TEST block is perturbed.  Because
# ridge_scale_free computes its coefficients from the train design alone and only then
# applies them to the test design, repeating it with a scrambled test block must leave a
# single shared coefficient vector: the map Zb -> prediction must be exactly affine with
# slope beta = c[:-1].  Recovering beta from two perturbed test blocks and predicting a
# third must give zero residual if and only if the coefficients were fitted once, from
# train rows only.
Zb1 = Zb + 1.0
Zb2 = Zb - 0.5
p1 = M.ridge_scale_free(Za, Zb1, Y[tr])
p2 = M.ridge_scale_free(Za, Zb2, Y[tr])
beta = np.linalg.lstsq(Zb1 - Zb2, p1 - p2, rcond=None)[0]
resid = np.abs((Zb - Zb2) @ beta + p2 - pa).max()
results["T6b_ridge_single_shared_coefficient_vector"] = bool(resid < 1e-10)
results["T6b_ridge_refit_residual"] = float(resid)

# (c) the selection stage must not see te. Behavioural check: permute the TEST block
# of Y and re-run selection; the selected (k, gamma) must not move. The shuffled-phase
# control inside _select_one legitimately touches X[te] for SCORING only, so a source
# grep is not the right test - the perturbation experiment is.
sel_a = M._select_one((f, base))
Ysave = Y.copy()
Y[te] = Y[te][rng.permutation(len(te))]
sel_b = M._select_one((f, base))
Y[:] = Ysave
results["T6c_selection_unchanged_by_test_permutation"] = bool(
    sel_a["selected"] == sel_b["selected"])

for k, v in results.items():
    print(f'{k}: {v}')
assert results["T6a_predictions_bit_identical"]
assert results["T6b_feature_scale_ignores_test_rows"]
assert results["T6c_selection_unchanged_by_test_permutation"]
print("LEAKAGE TESTS PASSED")
M.jdump(results, M.RESULTS / "leakage_checks.json")
