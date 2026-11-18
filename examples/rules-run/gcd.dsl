# Euclid's algorithm: the registers a and b start at the entry's values, and each step replaces (a, b) by (b, a % b) while b > 0.
h1 pos: 1*b >= 1
h2 go: any(pos)
h2 stop: all(!pos)
set a: b if go
set b: a % b if go
set steps: steps + 1 if go
out going: 1*go
out done: 1*stop
