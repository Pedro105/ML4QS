"""
Patches the installed pybrain package to work with scipy >= 1.7.
scipy.random was removed in newer scipy; pybrain still imports it.
Since scipy.random was just an alias for numpy.random, the fix is to
replace those imports with numpy equivalents.

Run this script once after installing pybrain:
    python patch_pybrain.py
"""

import os
import re
import sys
import importlib.util


def find_pybrain():
    # Don't import pybrain directly — it fails before patching (scipy.random missing)
    spec = importlib.util.find_spec('pybrain')
    if spec is None:
        print("ERROR: pybrain is not installed in the current environment.")
        sys.exit(1)
    return os.path.dirname(spec.origin)


def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read()

    lines = original.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        # Standalone: from scipy import random
        if re.match(r'^(\s*)from scipy import random\s*$', line):
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}from numpy import random')
            changed = True
        # Multi-import: from scipy import ..., random, ... or random at start/end
        elif re.match(r'^\s*from scipy import .*\brandom\b', line):
            new_line = re.sub(r',\s*random\b', '', line)   # trailing: ..., random
            new_line = re.sub(r'\brandom\s*,\s*', '', new_line)  # leading: random, ...
            new_lines.append(new_line)
            indent = re.match(r'^(\s*)', line).group(1)
            new_lines.append(f'{indent}from numpy import random')
            changed = True
        else:
            new_lines.append(line)

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

    return changed


def main():
    pybrain_dir = find_pybrain()
    print(f"Found pybrain at: {pybrain_dir}")

    patched = []
    for root, dirs, files in os.walk(pybrain_dir):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                if patch_file(fpath):
                    patched.append(os.path.relpath(fpath, pybrain_dir))

    if patched:
        print(f"\nPatched {len(patched)} file(s):")
        for p in patched:
            print(f"  {p}")
    else:
        print("\nNothing to patch (already applied or not needed).")


if __name__ == '__main__':
    main()
