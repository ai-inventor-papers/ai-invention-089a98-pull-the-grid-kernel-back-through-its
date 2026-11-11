"""Unit checks for gp_corr: posterior mean/variance and both selection criteria.

Verifies, against an independent dense reference:
  1. the posterior mean and predictive variance at the selected hyperparameters;
  2. that mll_mode="archived" selects the archive's criterion
     (-0.5 ||K^-1 Yc||_F^2 - 0.5 m log|K|) and mll_mode="quadratic" selects the
     true marginal likelihood (-0.5 tr(Yc^T K^-1 Yc) - 0.5 m log|K|);
  3. that the two criteria in fact differ, so the sensitivity row is meaningful.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from method import LGRID, N2G, S2G, gp_corr

rng = np.random.default_rng(0)
n, m, nte = 14, 5, 7
A = rng.standard_normal((n, 3))
B = rng.standard_normal((nte, 3))
Dtr = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
Dte = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
Y = rng.standard_normal((n, m))
corr = [(float(l), np.exp(-Dtr / (2 * l * l)), np.exp(-Dte / (2 * l * l))) for l in LGRID]

yb = Y.mean(0)
ys = Y.std(0) + 1e-12
Yc = (Y - yb) / ys

# --- independent dense scan for both criteria --------------------------------
rows = []
for lab, Ktr, Kte in corr:
    for s2 in S2G:
        for n2 in N2G:
            K = s2 * Ktr + n2 * np.eye(n)
            sign, ld = np.linalg.slogdet(K)
            al = np.linalg.solve(K, Yc)
            arch = -0.5 * float((al ** 2).sum()) - 0.5 * m * ld
            quad = -0.5 * float(np.einsum('ij,ij->j', Yc, al).sum()) - 0.5 * m * ld
            rows.append(dict(lab=lab, s2=s2, n2=n2, arch=arch, quad=quad,
                             Ktr=Ktr, Kte=Kte))

b_arch = max(rows, key=lambda r: r["arch"])
b_quad = max(rows, key=lambda r: r["quad"])

for mode, ref in (("archived", b_arch), ("quadratic", b_quad)):
    mu, pv, hp, Ktr_ret, Kte_ret = gp_corr(corr, Y, mll_mode=mode)
    assert abs(hp[0] - ref["lab"]) < 1e-12 and hp[1] == ref["s2"] and hp[2] == ref["n2"], (
        f"{mode} selection mismatch: got {hp}, dense reference "
        f"{(ref['lab'], ref['s2'], ref['n2'])}")
    assert np.array_equal(Ktr_ret, ref["Ktr"]), "returned Ktr must be the SELECTED one"
    assert np.array_equal(Kte_ret, ref["Kte"]), "returned Kte must be the SELECTED one"
    K = ref["s2"] * ref["Ktr"] + ref["n2"] * np.eye(n)
    Ks = ref["s2"] * ref["Kte"]
    a = np.linalg.solve(K, Yc)
    mu_ref = (Ks @ a) * ys + yb
    pv_ref = np.maximum(ref["s2"] - np.einsum('ij,jk,ik->i', Ks, np.linalg.inv(K), Ks),
                        1e-12)[:, None] * (ys ** 2)[None, :]
    print(f'[{mode}] selected {hp} | mu rel dev '
          f'{np.abs(mu - mu_ref).max() / np.abs(mu_ref).max():.2e} | pv rel dev '
          f'{np.abs(pv - pv_ref).max() / np.abs(pv_ref).max():.2e}')

print(f'archived criterion picks (l={b_arch["lab"]}, s2={b_arch["s2"]}, '
      f'n2={b_arch["n2"]}); quadratic picks (l={b_quad["lab"]}, s2={b_quad["s2"]}, '
      f'n2={b_quad["n2"]})')
print(f'the two criteria differ: quadratic minus archived score at the archive pick '
      f'{b_arch["quad"] - b_arch["arch"]:+.4f}')
print('TEST PASSED')
