#!/usr/bin/env python3
# Copyright (c) 2026 Nick Burkard
# SPDX-License-Identifier: MIT
"""Step 12.2: rewrite includes of headers that moved into the api/ and
api/ext/ tiers, from tools/restructure/tiers12.txt.

usage: python tools/restructure/rewriteTiers12.py [--check] [dir ...]
Default dirs: libs tests testutils plugin tools. Re-run after a merge that
brings in old spellings. --check lists hits and exits 1 without writing.
Bytes in, bytes out: line endings and encodings are kept.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".h", ".hpp", ".cpp", ".cu", ".cuh", ".inl", ".c", ".cc", ".mm")


def load_map(path):
    pairs = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                old, new = line.split()
                pairs[old.encode()] = new.encode()
    return pairs


_MAP = load_map(os.path.join(HERE, "tiers12.txt"))
_RE = re.compile(
    rb'^([ \t]*#[ \t]*include[ \t]*[<"])(' +
    b"|".join(re.escape(o) for o in sorted(_MAP, key=len, reverse=True)) +
    rb')([>"])', re.M)


def rewrite(data):
    return _RE.subn(lambda m: m.group(1) + _MAP[m.group(2)] + m.group(3), data)


def main(argv):
    check = "--check" in argv
    dirs = [a for a in argv if not a.startswith("--")] or [
        "libs", "tests", "testutils", "plugin", "tools"]
    files = hits = 0
    for top in dirs:
        for root, subdirs, names in os.walk(top):
            subdirs[:] = sorted(d for d in subdirs if d not in (".git", "build"))
            for name in sorted(names):
                if not name.endswith(EXTS):
                    continue
                path = os.path.join(root, name)
                with open(path, "rb") as f:
                    data = f.read()
                out, n = rewrite(data)
                if not n:
                    continue
                files += 1
                hits += n
                print("%s %d" % (path.replace(os.sep, "/"), n))
                if not check:
                    with open(path, "wb") as f:
                        f.write(out)
    print("%d includes in %d files%s" % (hits, files, " (check)" if check else ""))
    return 1 if check and hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
