"""Stop a repetition loop in the thinking, greedy (MLX), and keep the output the original's until then.
A loop is a copy: the last WINDOW tokens of the thinking where every K-gram ending there already occurred earlier in the output (so a
loop of slightly different copies counts too, once its pieces have all been seen). At the first one, the decoding goes back to the
start of the copy and from there to the end of the thinking never completes a K-gram the output already holds (no-repeat n-gram, as
transformers' no_repeat_ngram_size, switched on only then). The answer after </think> is left free (it may restate the thinking); a
model with no </think> token is watched over its whole output. Before the event the decoding is plain greedy (mlx_lm generate_step,
as kestrom-verify generate), so the output is the original's up to it: check it with kestrom-verify compare events.
A loop whose copies differ in every K-gram (a count going up: n^{103}, n^{104}, ...) is not found.
In the rules language (python -m pydoc kestrom.rules): `loopguard: WINDOW` (512 unless given), with kestrom generate --rules."""

K = 16  # the length of the pieces a copy is made of, in tokens
THINK_END = "</think>"
WINDOW = 512


class Copies:
    """Over the output tokens: which K-grams occurred, and how many K-grams in a row up to the end occurred earlier (the run)."""

    def __init__(self, toks):
        self.seen, self.run, self.toks = set(), 0, []
        for x in toks:
            self.add(x)

    def add(self, x):
        self.toks.append(x)
        if len(self.toks) >= K:
            g = tuple(self.toks[-K:])
            self.run = self.run + 1 if g in self.seen else 0
            self.seen.add(g)

    def start(self, window):
        """The start of the copy when the last `window` tokens are one, else None."""
        return len(self.toks) - self.run - K + 1 if self.run >= window - K + 1 else None
