# Before/after for benchUsdGenRestructure: alternates runs between builds,
# reports the median of per-run medians.
#   python tests/perf/compareBench.py ROUNDS buildDirA buildDirB [...]
# A build dir holds benchUsdGenRestructure (or Release/benchUsdGenRestructure.exe)
# and usd/*/resources. fakeHdPrman and *Tools plugins are left off the plugin path.
import glob, os, re, statistics, subprocess, sys

def exe(b):
    for c in ('benchUsdGenRestructure', 'benchUsdGenRestructure.exe',
              'Release/benchUsdGenRestructure.exe'):
        if os.path.exists(os.path.join(b, c)): return os.path.join(b, c)
    sys.exit('no benchUsdGenRestructure in ' + b)

rounds = int(sys.argv[1]); builds = [os.path.abspath(b) for b in sys.argv[2:]]
res = {b: {} for b in builds}
for _ in range(rounds):
    for b in builds:
        env = dict(os.environ)
        env['PXR_PLUGINPATH_NAME'] = os.pathsep.join(
            d for d in sorted(glob.glob(os.path.join(b, 'usd', '*', 'resources')))
            if 'fakeHdPrman' not in d and 'Tools' not in d)
        out = subprocess.run([exe(b), '--threads', '4', '--runs', '5'], env=env,
                             capture_output=True, text=True, timeout=1800).stdout
        for m in re.finditer(r'^RESULT (\S+)\s+median=\s*([\d.]+)', out, re.M):
            res[b].setdefault(m.group(1), []).append(float(m.group(2)))
names = list(res[builds[0]])
print('metric'.ljust(24) + ''.join(os.path.basename(os.path.dirname(b) or b).rjust(14) for b in builds))
for n in names:
    print(n.ljust(24) + ''.join(('%.3f' % statistics.median(res[b][n])).rjust(14)
                                if res[b].get(n) else '-'.rjust(14) for b in builds))
