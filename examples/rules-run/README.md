# Rules as a network of their own, with no model

`kestrom rules run` runs a rules file with no model: nothing is edited, so there is no tier and no guarantee to check. What it gives
is the answers of the rules on the inputs you list, the units that fired for each, and how many answers agree with your labels. Use
it to try rules before fitting them to a model, or to write a small classifier or state machine by hand.

```sh
sh examples/rules-run/run.sh        # numpy only
```

**A router with two heads (`router.dsl`, `router.json`).** Eight questions about a building: which meter (air conditioning or
lighting), and whether the floor is low or high. The inputs are regexes on the text: found (`match`), counted (`count`) and a
number read (`number`). Classes named `HEAD/CLASS` are answered per head, each null on its own.

```
labelled 8 right 8 no answer 0 wrong []
```

- "3階の空調じゃなくて照明をお願い" (not the air conditioning of floor 3, the lighting) answers `{"meter": "light", "floor": "low"}`: the
  rule `a: all(ac, !not_ac)` keeps the negated word out.
- "3階と5階の空調をください" (floors 3 and 5) answers `{"meter": "ac", "floor": null}`: two floors counted, so no floor is answered
  rather than the first.
- "今日の天気は?" (today's weather) answers null for both heads: no unit fires.

**Euclid's algorithm (`gcd.dsl`, `gcd.json`).** Eight lines of rules. The registers `a` and `b` start at each entry's values; `"steps": 40`
runs the entry 40 times, and while `b >= 1` the set lines replace (a, b) by (b, a % b) and count the step. `targets` checks the
register `a` against the greatest common divisor.

```
labelled 4 right 4 no answer 0 wrong []
targets {'a': (4, 0.0)} (register: entries, mean absolute error)
```

| a, b | a after the last step | Steps counted |
|---|---:|---:|
| 1071, 462 | 21 | 3 |
| 30, 20 | 10 | 2 |
| 7, 13 | 1 | 4 |
| 832040, 514229 (two Fibonacci numbers) | 1 | 28 |

With `"steps": 12` the last pair is still `going` at the end and `a` is not the divisor yet: the steps are a budget you give, not a
loop that runs until done.

`kestrom rules build gcd.dsl out/gcd.npz` writes the network out (the weights, and each set line's register, expression and unit).

These are eight questions and four pairs written for this example, with the rules written to answer them: they show how the lines
read, not how well such rules do on questions nobody has seen. Where a rule can be written as plain code, plain code does the same job.
