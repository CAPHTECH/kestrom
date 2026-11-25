"""The command lines print their help, not a traceback, when asked for it or given too few arguments (no model)."""
import sys

import pytest

import kestrom.__main__ as k
import kestrom_verify.__main__ as v
from kestrom import ifc, inspect, kv, rows
from kestrom_verify import call


@pytest.mark.parametrize("argv, cli, doc", [
    ([], k, k.__doc__), (["--help"], k, k.__doc__), (["nope"], k, k.__doc__),
    (["kv"], k, kv.__doc__), (["kv", "--help"], k, kv.__doc__), (["kv", "nope"], k, kv.__doc__),
    (["kv", "compile"], k, kv.__doc__),
    (["ifc"], k, ifc.__doc__), (["ifc", "--help"], k, ifc.__doc__), (["ifc", "nope"], k, ifc.__doc__),
    (["ifc", "control"], k, ifc.__doc__), (["ifc", "firewall"], k, ifc.__doc__),
    (["inspect"], k, inspect.__doc__), (["inspect", "--help"], k, inspect.__doc__), (["inspect", "nope"], k, inspect.__doc__),
    (["inspect", "shape"], k, inspect.__doc__), (["inspect", "mirror"], k, inspect.__doc__), (["inspect", "disarm"], k, inspect.__doc__),
    (["rows", "ban", "-h"], k, rows.__doc__),
    ([], v, v.__doc__), (["range", "--help"], v, v.__doc__), (["range"], v, v.__doc__), (["generate", "mlx"], v, v.__doc__),
    (["logits", "--help"], v, v.__doc__), (["diff"], v, v.__doc__)])
def test_help(monkeypatch, argv, cli, doc):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == doc


def test_call_runs_arguments_that_fit():
    assert call(lambda a, b="y": a + b, ["x"], "help") == "xy"
    with pytest.raises(SystemExit):
        call(lambda a: a, ["x", "y"], "help")
