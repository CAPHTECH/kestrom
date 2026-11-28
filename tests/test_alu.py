"""Unit tests for the Tier-1 inline `alu:` sidecar rule (NumPy-only, no MLX required)."""
import pytest

from kestrom import generate, rules


def test_compile_alu_default_and_custom():
    r_def = rules.compile_rules("alu:\n", None)
    assert r_def["alu"] == 4
    r_six = rules.compile_rules("alu: 6\n", None)
    assert r_six["alu"] == 6


def test_compile_alu_rejects_bad_syntax(tmp_path):
    for bad in ("alu: 0\n", "alu: -2\n", "alu: abc\n", "alu: 4 5\n", "alu x: 4\n", "alu: 4\nalu: 5\n"):
        with pytest.raises(ValueError, match="alu"):
            rules.compile_rules(bad, None)
    p = tmp_path / "bad_standalone.dsl"
    p.write_text('in a: match "x"\nh1 u: any(a)\nh2 v: any(u)\nout hit: 1*v\nalu: 4\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        rules.standalone(str(p))


def test_match_inline_alu_triggers_and_outside():
    # Multi-digit addition, subtraction, multiplication, commas, unicode times, trailing space
    assert generate.match_inline_alu("Step 1: 123456 * 12345 =", 4) == ("123456*12345", " 1524064320")
    assert generate.match_inline_alu("Step 1: 123,456 * 12,345 = ", 4) == ("123456*12345", "1524064320")
    assert generate.match_inline_alu("Total: 987654321 - 123456789 =", 4) == ("987654321-123456789", " 864197532")
    assert generate.match_inline_alu("Sum: 1000 + 2 =", 4) == ("1000+2", " 1002")
    assert generate.match_inline_alu("Prod: 312 × 415 =", 4) == ("312×415", " 129480")

    # Outside / below threshold must never fire
    assert generate.match_inline_alu("17 + 28 =", 4) == (None, None)
    assert generate.match_inline_alu("12 * 9 =", 4) == (None, None)
    assert generate.match_inline_alu("999 + 999 =", 4) == (None, None)
    assert generate.match_inline_alu("123456 * 12345 = 1524064320", 4) == (None, None)
    assert generate.match_inline_alu("The capital of Portugal is Lisbon.", 4) == (None, None)
    assert generate.match_inline_alu("1000 + 2 =", 5) == (None, None)
