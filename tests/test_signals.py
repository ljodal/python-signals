"""
Test that signals can be defined, sent and received.
"""

from signals import Signal


def test_receiver() -> None:
    class MySignal(Signal):
        number: int

    @MySignal.receiver()
    def receiver(*, number: int) -> None:  # pylint: disable=unused-variable
        pass

    MySignal.send(number=1)
