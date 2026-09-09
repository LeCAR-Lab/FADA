"""Uniform numeric-range validation for the FADA entry-point CLIs.

The three FADA entry points (`fada.planner_idm.train`, `.finetune_idm_lora`,
`.eval_checkpoint`) expose their hyperparameters as `default=None` override flags built
from a `(field, argparse type, choices, help)` spec table. `argparse`'s `type=int` /
`type=float` only checks that the token *parses*, so it accepts `nan`, negative
probabilities, dropout above 1, and zero-valued intervals. `nan` in particular passes
every ordinary ``<`` / ``>`` comparison.

Every numeric flag is classified into a *domain* (finite, probability, strictly positive,
non-negative, positive integer, ...) and the domain is enforced at the entry-point
boundary -- the resolver that turns the parsed `Namespace` into the object production
code reads (`config_from_args` / `resolve_finetune_defaults` / `resolve_eval_defaults`).
Validating there rather than in `argparse`'s `type=` callable means a hand-built
`Namespace` is checked too, and the raised `ValueError` names the flag, the field, the
offending value and the allowed range.

Values are never clamped: an out-of-range value is rejected. Where a value means
"disabled" (0 for an interval, -1 for an ablation level) the domain carries a
`disabled_note`, appended to the flag's `--help` text and repeated in the range error.

**Where a bound may be placed.** A domain may only exclude values the consuming code has
no defined behavior for. Where that code guards a field with `if x > 0:` / `if x >= 0:` /
`if x <= 0: return`, every value on the other side of the guard has a defined
"off"/"leave it alone" meaning and stays accepted -- the domain must not narrow it to a
single sentinel (`-1` out of "any negative", `0` out of "any non-positive").
`unbounded_int()` and `finite_float()` are for those fields: they carry a
`disabled_note` naming which side of the guard is "off" without imposing a bound. Fields
with no such guard (a zero learning rate, `--num-envs 0`, `nan` anywhere) are bounded
normally.

Nothing here changes what an in-range value does.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

__all__ = [
    "FINITE_FLOAT",
    "NON_NEGATIVE_FLOAT",
    "NON_NEGATIVE_INT",
    "POSITIVE_FLOAT",
    "POSITIVE_INT",
    "PROBABILITY",
    "UNIT_HALF_OPEN",
    "UNIT_LEFT_OPEN",
    "UNIT_OPEN",
    "NumericRange",
    "finite_float",
    "flag_spelling",
    "non_negative_float_disabled_by_zero",
    "non_negative_int_disabled_by_zero",
    "unbounded_int",
    "validate_numeric_values",
]


def flag_spelling(field: str) -> str:
    """`--`-prefixed CLI spelling of a snake_case field, matching the parser builders."""
    return "--" + field.replace("_", "-")


@dataclasses.dataclass(frozen=True)
class NumericRange:
    """One numeric domain: a kind (`int`/`float`), finite bounds, and inclusivity.

    `disabled_note` does not relax the bounds -- it records that some in-range value is
    the documented "off"/sentinel value (0 for an interval, -1 for an ablation level),
    so `help_suffix()` states it and the range error repeats it: the value is accepted
    and the help text says so.
    """

    kind: str
    lo: float | None = None
    hi: float | None = None
    lo_inclusive: bool = True
    hi_inclusive: bool = True
    disabled_note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("int", "float"):
            raise ValueError(f"NumericRange.kind must be 'int' or 'float', got {self.kind!r}")

    # -- description ------------------------------------------------------------------
    def describe(self) -> str:
        """Human-readable allowed range, e.g. ``a finite float in [0, 1)``."""
        noun = "an integer" if self.kind == "int" else "a finite float"
        if self.lo is None and self.hi is None:
            return noun
        if self.hi is None:
            op = ">=" if self.lo_inclusive else ">"
            return f"{noun} {op} {_fmt(self.lo, self.kind)}"
        if self.lo is None:
            op = "<=" if self.hi_inclusive else "<"
            return f"{noun} {op} {_fmt(self.hi, self.kind)}"
        left = "[" if self.lo_inclusive else "("
        right = "]" if self.hi_inclusive else ")"
        return f"{noun} in {left}{_fmt(self.lo, self.kind)}, {_fmt(self.hi, self.kind)}{right}"

    def help_suffix(self) -> str:
        """Sentence appended to the flag's ``--help`` text so the range is documented."""
        suffix = f" Accepts {self.describe()}."
        if self.disabled_note:
            suffix += " " + self.disabled_note
        return suffix

    # -- enforcement ------------------------------------------------------------------
    def check(self, value: object, *, field: str, flag: str | None = None) -> int | float:
        """Return `value` coerced to the domain's kind, or raise `ValueError`.

        The message names the flag, the field, the value and the allowed range, so a
        researcher reading only stderr can fix the invocation without opening the source.
        """
        flag = flag or flag_spelling(field)
        prefix = f"{flag} ({field}): invalid value {value!r}"
        if isinstance(value, bool):
            raise ValueError(f"{prefix} -- expected {self.describe()}, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} -- expected {self.describe()}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{prefix} -- must be finite (nan/inf are never valid); expected {self.describe()}")
        if self.kind == "int":
            if numeric != int(numeric):
                raise ValueError(f"{prefix} -- expected {self.describe()}")
            numeric = float(int(numeric))
        if self.lo is not None:
            if numeric < self.lo or (numeric == self.lo and not self.lo_inclusive):
                raise ValueError(f"{prefix} -- out of range; expected {self.describe()}{_disabled_hint(self)}")
        if self.hi is not None:
            if numeric > self.hi or (numeric == self.hi and not self.hi_inclusive):
                raise ValueError(f"{prefix} -- out of range; expected {self.describe()}{_disabled_hint(self)}")
        return int(numeric) if self.kind == "int" else float(value)


def _fmt(bound: float | None, kind: str) -> str:
    if bound is None:
        return "-"
    if kind == "int" or float(bound).is_integer():
        return str(int(bound))
    return str(bound)


def _disabled_hint(domain: NumericRange) -> str:
    if not domain.disabled_note:
        return ""
    return " (" + domain.disabled_note.rstrip(".") + ")"


# ---------------------------------------------------------------------------
# The domain vocabulary. Every numeric FADA flag is classified into one of these.
# ---------------------------------------------------------------------------

#: Any finite float. Used where the sign is genuinely free (e.g. `--grad-clip`, whose
#: documented contract is that any value <= 0 disables clipping).
FINITE_FLOAT = NumericRange("float")
#: Strictly positive finite float -- learning rates, timeouts, LoRA alpha.
POSITIVE_FLOAT = NumericRange("float", lo=0.0, lo_inclusive=False)
#: Non-negative finite float -- loss weights and noise scales, where 0 means "off".
NON_NEGATIVE_FLOAT = NumericRange("float", lo=0.0)
#: Probability / fraction on the closed unit interval.
PROBABILITY = NumericRange("float", lo=0.0, hi=1.0)
#: Fraction on [0, 1) -- dropout and mask/split ratios, where 1.0 is degenerate.
UNIT_HALF_OPEN = NumericRange("float", lo=0.0, hi=1.0, hi_inclusive=False)
#: Fraction on (0, 1) -- mixing ratios that must leave both sides non-empty.
UNIT_OPEN = NumericRange("float", lo=0.0, hi=1.0, lo_inclusive=False, hi_inclusive=False)
#: Fraction on (0, 1] -- sampling ratios where "all of it" is valid but "none" is not.
UNIT_LEFT_OPEN = NumericRange("float", lo=0.0, hi=1.0, lo_inclusive=False)
#: Count / size / interval that must be at least 1.
POSITIVE_INT = NumericRange("int", lo=1)
#: Count that may be 0 without 0 carrying a special meaning (e.g. a seed, a step budget
#: whose 0 simply means "none").
NON_NEGATIVE_INT = NumericRange("int", lo=0)


def unbounded_int(*, disabled_note: str) -> NumericRange:
    """Integer domain with no bounds at all, for a field whose original guard defines them.

    Used where the consuming code reads the field behind an `if x > 0:` / `if x >= 0:` /
    `if x <= 0: return` guard, so every value on the far side of the guard means
    "off"/"leave it alone". `disabled_note` is required, and names which side that is.
    Non-integral and non-finite values are still rejected.
    """
    return NumericRange("int", disabled_note=disabled_note)


def finite_float(
    *,
    hi: float | None = None,
    hi_inclusive: bool = True,
    disabled_note: str,
) -> NumericRange:
    """Float counterpart of `unbounded_int`: unbounded below, optionally bounded above.

    The upper bound is kept where one exists for a reason the original code also had (a
    ratio of 1.0 is degenerate, not "off"); the lower bound is the one a guard like
    `if ratio > 0.0:` already supplies.
    """
    return NumericRange("float", hi=hi, hi_inclusive=hi_inclusive, disabled_note=disabled_note)


def non_negative_float_disabled_by_zero(note: str) -> NumericRange:
    """`>= 0` float whose 0 means "off"."""
    return NumericRange("float", lo=0.0, disabled_note=note)


def non_negative_int_disabled_by_zero(note: str) -> NumericRange:
    """`>= 0` int whose 0 is a documented "off"/"auto" value."""
    return NumericRange("int", lo=0, disabled_note=note)


def validate_numeric_values(
    spec: Mapping[str, NumericRange],
    values: Mapping[str, object],
) -> None:
    """Enforce `spec` over every entry of `values` that is present and not `None`.

    `values` is the "what did the user actually pass" mapping the resolvers already
    build, so an omitted flag (still `None`) is never range-checked and the shipped
    default is never second-guessed. Fields are checked in sorted order so the first
    reported error is stable regardless of argparse ordering.
    """
    for field in sorted(values):
        domain = spec.get(field)
        if domain is None:
            continue
        value = values[field]
        if value is None:
            continue
        domain.check(value, field=field)
