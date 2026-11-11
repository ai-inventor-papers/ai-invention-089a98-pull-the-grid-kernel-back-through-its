import sys, json, time
sys.path.insert(0, 'code')
import numpy as np
from physics import Physics
from feeder import FeederOps

t0 = time.time()
ph = Physics()
ops = FeederOps.build(ph)
print('built %.1fs' % (time.time() - t0), flush=True)
c = dict(ops.checks)
for kk in ('re_lam_smallest_45', 'im_lam_frac_smallest_45', 'eig_argmax_entries_45'):
    c.pop(kk, None)
print(json.dumps(c, indent=1), flush=True)
for k, r in ops.routes.items():
    print(k, None if r.F is None else r.F.shape, '|', r.note[:110], flush=True)
ops.save_cache('results/spectra_cache.npz')
print('cache written %.1fs' % (time.time() - t0), flush=True)
