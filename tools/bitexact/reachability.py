"""Module-level reachability analysis starting from the FADA entry points.

Resolves both Python imports and the string references used in `config_values`
(of the form `"holosoma.managers.reward.terms.X:fn"`, of which the repository has
over a thousand).

**This is reachability in the module import graph, not reachability of the runnable
entry-point surface.** A module being unreachable from `ENTRIES` only means no other
module imports it -- it may still carry a `main()` that runs standalone under
`python -m`. Such a module is never imported by anything and yet executes fine, and
this tool alone would misreport it as unreachable. The wording below is therefore
narrowed from "unreachable" to "unsupported/undocumented runnable surface": it is a
starting point for human review, not proof that something can be deleted.

Results are grouped by provenance. Upstream modules are kept whether or not they are
reached. Only modules added by this project and not imported anywhere are review
candidates -- and reviewing one means separately checking whether it has its own
`if __name__ == "__main__"` / CLI entry point, rather than concluding from this
tool's output alone that it cannot be run.

The provenance split is only meaningful if "upstream" names something that really is
upstream. `is_upstream()` used to hardcode `git cat-file -e main:<path>`, which is
right here (`main` is the pinned upstream snapshot) and wrong in the published tree,
where `main` IS the release: every path then resolves, every module is classified
upstream, and the "added by this project" partition prints as empty -- a partition
that cannot be non-empty is not a review aid, it is decoration. The baseline ref now
comes from `tools/upstream_ref.py`, which prefers the `upstream-baseline` tag the
release build puts on the upstream snapshot commit and accepts a candidate only if
its recursive tree object IS the pinned baseline tree. If no baseline resolves, this tool exits non-zero
instead of printing a partition it cannot compute.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.upstream_ref import unresolved_baseline_message, upstream_baseline_ref

ROOT = "src/holosoma/holosoma"
ENTRIES = [
    "holosoma.fada.planner_idm.train",
    "holosoma.fada.planner_idm.eval_checkpoint",
    "holosoma.fada.planner_idm.finetune_idm_lora",
    "holosoma.train_agent",
    "holosoma.eval_agent",
]


def collect_modules() -> dict[str, str]:
    mods: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(ROOT):
        if "__pycache__" in dirpath or "/tests" in dirpath or dirpath.endswith("/tests"):
            continue
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dirpath, f)
                mods[p[len("src/holosoma/") : -3].replace("/", ".").replace(".__init__", "")] = p
    return mods


def deps(path: str) -> set[str]:
    src = open(path).read()
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("holosoma"):
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.startswith("holosoma")}
    out |= set(re.findall(r'["\'](holosoma\.[A-Za-z0-9_.]+)[:"\']', src))
    return out


class BaselineListingError(Exception):
    """`git` could not list the baseline tree, so provenance cannot be computed."""


def upstream_paths(baseline_ref: str) -> set[str]:
    """Every path in the baseline tree, from ONE git call whose status is checked.

    The previous implementation asked `git cat-file -e <ref>:<path>` once per module
    and read any non-zero status as "not upstream". `git` has more than two answers:
    128 covers a missing path, a missing ref, an ambiguous ref, a corrupt object and
    a broken object store alike -- so a repository that could not be read at all
    reported every module as "added by this project" and exited 0, which is a
    partition nobody can distinguish from a real one. Listing the tree once turns
    that into a single call with a single, checked failure mode: either the baseline
    is readable and membership is exact, or this raises and the tool exits non-zero.
    """
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", f"{baseline_ref}^{{tree}}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise BaselineListingError(
            f"git ls-tree exit {proc.returncode} for {baseline_ref!r}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip() or '(no stderr)'}"
        )
    return {entry for entry in proc.stdout.decode("utf-8", "replace").split("\0") if entry}


def is_upstream(path: str, baseline_paths: set[str]) -> bool:
    return path in baseline_paths


def main() -> int:
    baseline_ref = upstream_baseline_ref()
    if baseline_ref is None:
        print(f"ERROR: {unresolved_baseline_message()}", file=sys.stderr)
        return 1
    try:
        baseline_paths = upstream_paths(baseline_ref)
    except BaselineListingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "  The upstream/ours partition cannot be computed without reading the baseline "
            "tree, and printing it anyway would label every module 'added by this project'.",
            file=sys.stderr,
        )
        return 1
    if not baseline_paths:
        print(f"ERROR: the baseline tree {baseline_ref!r} lists no files at all.", file=sys.stderr)
        return 1
    mods = collect_modules()
    seen: set[str] = set()
    stack = list(ENTRIES)
    while stack:
        cur = stack.pop()
        cand = cur
        while cand and cand not in mods:
            cand = cand.rsplit(".", 1)[0] if "." in cand else ""
        if not cand or cand in seen:
            continue
        seen.add(cand)
        stack += list(deps(mods[cand]))

    not_import_reachable = sorted(set(mods) - seen)
    upstream = [m for m in not_import_reachable if is_upstream(mods[m], baseline_paths)]
    ours = [m for m in not_import_reachable if not is_upstream(mods[m], baseline_paths)]

    print(f"upstream baseline ref: {baseline_ref}")
    print(
        f"modules {len(mods)}  import-reachable {len(seen)}  "
        f"not imported (candidates for unsupported/undocumented runnable surface) "
        f"{len(not_import_reachable)}"
    )
    print(f"\nupstream, not imported ({len(upstream)}) -- kept unconditionally, do not delete:")
    for m in upstream:
        print(f"  KEEP  {m}")
    print(
        f"\nadded by this project, not imported ({len(ours)}) -- review candidates, "
        "not a deletion verdict: check each one for a standalone main()/CLI entry "
        "point before calling it unused:"
    )
    for m in ours:
        print(f"  REVIEW  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
