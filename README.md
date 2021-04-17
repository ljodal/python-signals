# Python Signals

This package implements a pub/sub style concept in Python.

It's inspired by Django's signals, but with some major changes:

 * Strict type checking, both at runtime and statically with mypy
 * Support for running receivers in the background with celery


## Usage

```python
from signals import Signal


# Signal definition, this defines the structure of the payload
class MySignal(Signal):
    a_number: int
    a_string: str


# Signal receiever
# Note that this only receives one of the signal parameters
@MySignal.receiver()
def handle_signal(*, a_number: int) -> None:
    pass


# Signal sender
MySignal.send(a_number=1, a_string='foo')
```
