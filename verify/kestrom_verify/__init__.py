import inspect

HELP = {"-h", "--help"}


def call(f, args, help):
    """f(*args); or the help printed (exit status 1) when ARGS ask for it (-h, --help) or do not fit f's parameters."""
    try:
        if HELP & set(args):
            raise TypeError
        inspect.signature(f).bind(*args)
    except TypeError:
        raise SystemExit(help) from None
    return f(*args)
