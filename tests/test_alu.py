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

    # The lhs continues an expression or a number: never the last two operands alone
    for text in ("Total: 1200 + 3400 + 5600 =", "10000 / 1000 * 2000 =", "(1000 + 2000 =", "Balance: -5000 + 2000 =",
                 "Price 12.5 * 1000 =", "0x1000 + 2000 =", "$1,234.56 + 1000 =", "v1000 + 2 =", "1000 + 2 =\n", "1000 + 2 = \n"):
        assert generate.match_inline_alu(text, 4) == (None, None), text
    assert generate.match_inline_alu("1000 + 2000 = 3000 + 4000 =", 4) == ("3000+4000", " 7000")  # after an = sign: a new expression


def test_alu_processor_injects_and_gives_way_to_a_ban():
    mx = pytest.importorskip("mlx.core")

    class Tok:  # one token per character
        CH = "0123456789 +=*-x"

        def decode(self, t):
            return "".join(self.CH[i] for i in t)

        def encode(self, s, add_special_tokens=False):
            return [self.CH.index(c) for c in s]

    def steps(text, n, ban=()):  # the text taken at n steps after `text` (the model's top token is x), the events and the firings
        toks, spelled, fires, out = Tok().encode(text), [], [], []
        procs = generate.processors(list(toks), [Tok.CH.index(c) for c in ban], False, None, Tok(), None, spelled, alu=4, alu_fires=fires)
        for _ in range(n):
            z = mx.array([[1.0 if c == "x" else 0.0 for c in Tok.CH]])
            for f in procs:
                z = f(mx.array([9, 9] + toks + out), z)
            out.append(int(mx.argmax(z[0])))
        return Tok().decode(out), spelled, fires
    assert steps("1000 + 2 =", 6) == (" 1002x", [10, 11, 12, 13, 14], [{"position": 10, "expr": "1000+2", "injected": "1002", "length": 5}])
    assert steps("1000 + 2 = ", 5) == ("1002x", [11, 12, 13, 14], [{"position": 11, "expr": "1000+2", "injected": "1002", "length": 4}])
    assert steps("1000 + 2 =", 2, ban=" ") == ("xx", [], [])  # the first token is banned (a phrase restart): the firing is given up
    assert steps("1000 + 2 = 10", 2) == ("xx", [], [])  # a restart inside a value: no firing
    assert steps("7 + 1000 + 2 =", 2) == ("xx", [], [])
