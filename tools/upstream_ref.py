"""Resolve the git ref that names the *upstream Holosoma baseline*.

Checks in this repository partition files into "inherited from upstream Holosoma"
and "added by this project" by asking git whether a path exists in the baseline.
The ref name ``main`` alone will not do: in the published tree ``main`` points at
the release commit itself, so ``main:<path>`` resolves to the release's own file
and every path classifies as upstream.

Resolution, in order:

1. ``FADA_UPSTREAM_REF``, if set. An override is final: if it does not resolve, or
   is not the baseline, resolution returns None rather than falling through to the
   candidates below.
2. ``upstream-baseline`` -- the annotated tag on the published tree's first commit,
   the upstream snapshot. A tag rather than a branch, so it names one fixed commit;
   ``git push --tags`` publishes it alongside ``main``.
3. ``main`` -- in a development checkout, the upstream snapshot this work is built
   on.

Every candidate then has to pass :func:`looks_like_upstream`.

Validation rule: a candidate ref is the upstream baseline iff its recursive tree
object equals :data:`UPSTREAM_BASELINE_TREE`. That SHA-1 covers every path, mode
and blob, so any addition, deletion, edit, rename or mode-bit change anywhere in
the tree changes it.

The pin is a *tree* hash rather than the baseline *commit* SHA because the
published release's commit 1 is a fresh commit (new author, date and message) with
a different commit SHA but the identical tree: ``build_release_tree.sh`` builds it
with ``git archive <pinned-sha> | tar -x`` followed by ``git add --force`` over
that commit's own ``ls-tree`` listing, which round-trips the tree object exactly
(asserted by the build and by ``tools/test_upstream_ref.py``). One value therefore
identifies the baseline in both contexts:

* a development checkout, where ``main`` is the pinned upstream snapshot named in
  ``tools/release/UPSTREAM_COMMIT``, and
* the published tree, where ``tools/release/`` and its pin file are withheld but
  ``upstream-baseline`` points at commit 1.

The constant lives in this module, which the release ships, so the published tree
can answer the question without the pin file. ``build_release_tree.sh``
cross-checks it against ``tools/release/UPSTREAM_COMMIT`` on every build; changing
the baseline means changing both in one edit.

:data:`UPSTREAM_ANCHOR_PATHS` is a second, independent assertion on the *pin*: it
catches a constant edited to an unrelated tree (an empty commit, another
repository), which would satisfy identity while making every file in the release
classify as newly added.

Callers must treat ``None`` as "cannot classify" and fail closed rather than
substituting a best-effort answer.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: Tag written onto the published tree's first commit (the upstream snapshot).
UPSTREAM_BASELINE_TAG = "upstream-baseline"

#: Env var for an explicit override (a fork with a differently-named baseline).
UPSTREAM_REF_ENV = "FADA_UPSTREAM_REF"

#: Candidate refs, in priority order, when no override is set.
DEFAULT_CANDIDATE_REFS: tuple[str, ...] = (UPSTREAM_BASELINE_TAG, "main")

#: The recursive git tree object of the upstream Holosoma baseline -- i.e. of the
#: commit pinned in ``tools/release/UPSTREAM_COMMIT``, and identically of commit 1
#: of the published release (the build reproduces the tree object exactly, and
#: asserts it).
#:
#: One SHA-1 over every path, mode and blob in the tree: no addition, deletion,
#: edit, rename or mode change anywhere leaves it unchanged.
#:
#: Changing this value re-defines what "upstream" means for every check in the
#: repository. ``tools/release/build_release_tree.sh`` refuses to build unless it
#: matches the pinned commit's tree, so it cannot be changed alone.
UPSTREAM_BASELINE_TREE = "14d62c87d8dd7c6a1457e6587fa96f85a43430e2"

#: Paths every upstream Holosoma snapshot has. Their absence means the *pin* names
#: something that is not a Holosoma tree at all (an empty commit, or someone else's
#: repository), which identity alone cannot distinguish from a real baseline.
UPSTREAM_ANCHOR_PATHS: tuple[str, ...] = (
    "LICENSE",
    "src/holosoma/holosoma/train_agent.py",
    "src/holosoma_inference/holosoma_inference/run_policy.py",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def ref_exists(ref: str, cwd: Path | None = None) -> bool:
    """True if `ref` resolves to a commit in the repository at `cwd`."""
    return _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd).returncode == 0


def tree_of(ref: str, cwd: Path | None = None) -> str | None:
    """The recursive tree object id of `ref`'s commit, or None if it doesn't resolve."""
    result = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{tree}}"], cwd)
    if result.returncode != 0:
        return None
    tree = result.stdout.strip()
    return tree or None


def _paths_in_tree(ref: str, paths: tuple[str, ...], cwd: Path | None = None) -> list[str]:
    """Which of `paths` have at least one blob under them in `ref`'s tree.

    `git ls-tree -r` expands directories, so an entry naming a directory reports a
    hit for any file anywhere beneath it.
    """
    present: list[str] = []
    for path in paths:
        result = _git(["ls-tree", "-r", "--name-only", ref, "--", path], cwd)
        if result.returncode == 0 and result.stdout.strip():
            present.append(path)
    return present


def check_upstream_ref(ref: str, cwd: Path | None = None) -> tuple[bool, list[str]]:
    """Validate `ref` as the upstream baseline and report what was actually checked.

    Returns ``(ok, lines)``, where `lines` state the concrete findings for a caller
    to print. `ok` is True only if `ref`'s tree object equals
    :data:`UPSTREAM_BASELINE_TREE` and every :data:`UPSTREAM_ANCHOR_PATHS` entry is
    present.
    """
    lines: list[str] = []
    if not ref_exists(ref, cwd):
        return False, [f"{ref}: does not resolve to a commit"]

    tree = tree_of(ref, cwd)
    identical = tree == UPSTREAM_BASELINE_TREE
    if identical:
        lines.append(
            f"{ref}: tree {tree} IS the pinned upstream baseline tree -- every path, mode and "
            "blob in it is the baseline's, byte for byte"
        )
    else:
        lines.append(
            f"{ref}: tree {tree} is not the upstream baseline tree "
            f"({UPSTREAM_BASELINE_TREE}) -- it differs from the baseline somewhere, so it is "
            "not the baseline"
        )

    anchors_present = _paths_in_tree(ref, UPSTREAM_ANCHOR_PATHS, cwd)
    anchors_missing = [p for p in UPSTREAM_ANCHOR_PATHS if p not in anchors_present]
    if anchors_missing:
        lines.append(
            f"{ref}: missing {len(anchors_missing)}/{len(UPSTREAM_ANCHOR_PATHS)} upstream anchor paths "
            "-- it is not a Holosoma tree"
        )
        lines.extend(f"    missing: {p}" for p in anchors_missing)
    else:
        lines.append(
            f"{ref}: all {len(UPSTREAM_ANCHOR_PATHS)} upstream Holosoma anchor paths present "
            f"({', '.join(UPSTREAM_ANCHOR_PATHS)})"
        )

    return (identical and not anchors_missing), lines


def looks_like_upstream(ref: str, cwd: Path | None = None) -> bool:
    """True if `ref` IS the upstream Holosoma baseline. See :func:`check_upstream_ref`."""
    return check_upstream_ref(ref, cwd)[0]


def upstream_baseline_ref(cwd: Path | None = None) -> str | None:
    """The ref naming the upstream Holosoma baseline, or None if there isn't one.

    None means "this repository has no upstream baseline to compare against".
    Callers must fail closed on it, not fall back to a guess.
    """
    override = os.environ.get(UPSTREAM_REF_ENV, "").strip()
    if override:
        # Final: an override that fails validation returns None rather than
        # falling through to DEFAULT_CANDIDATE_REFS.
        if ref_exists(override, cwd) and looks_like_upstream(override, cwd):
            return override
        return None
    for ref in DEFAULT_CANDIDATE_REFS:
        if ref_exists(ref, cwd) and looks_like_upstream(ref, cwd):
            return ref
    return None


def unresolved_baseline_message() -> str:
    """Explanation to print when :func:`upstream_baseline_ref` returns None."""
    return (
        "Cannot resolve the upstream Holosoma baseline ref. Tried "
        f"{UPSTREAM_REF_ENV} (env override), then {', '.join(DEFAULT_CANDIDATE_REFS)}; "
        "none of them resolves to a commit whose tree object IS the pinned baseline tree "
        f"{UPSTREAM_BASELINE_TREE} and carries every one of "
        f"{', '.join(UPSTREAM_ANCHOR_PATHS)}. "
        "In the published tree the baseline is the annotated tag "
        f"'{UPSTREAM_BASELINE_TAG}' on the first commit -- if it was not pushed "
        "(`git push --tags`) or was deleted, fetch it, or point "
        f"{UPSTREAM_REF_ENV} at the upstream snapshot commit. This check refuses to "
        "guess: comparing the release against itself would make it pass "
        "unconditionally, which is worse than not running it."
    )
