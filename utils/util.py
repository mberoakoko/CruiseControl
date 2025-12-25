import typing


def iterate[X](step: typing.Callable[[X], X], start: X) -> typing.Iterator[X]:
    state = start
    while True:
        yield state
        state = step(start)


def converge[X](values: typing.Iterator[X], done: typing.Callable[[X, X], bool]) -> typing.Iterator[X]:
    a = next(values)
    if a is None:
        return
    yield a
    for b in values:
        yield b
        if done(a, b):
            return
        a = b


def last[X](values: typing.Iterator[X]) -> typing.Optional[X]:
    try:
        *_, last_value = values
        return last_value
    except ValueError:
        return None

def converged[X](values: typing.Iterator[X], done: typing.Callable[[X, X], bool]) -> X:
    result = last(converge(values, done))
    if result is None:
        raise ValueError("End of iterator ")
    return result