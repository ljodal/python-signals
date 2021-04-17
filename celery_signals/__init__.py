"""
Celery signals is an implementation of something very similar to Django's
signals, but unlike the Django signals the receivers are called in a celery
worker instead of inline when the signal is sent.

In addition these signals implement strict type checking at runtime and
optionally with mypy using the provided plugin.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
)

from celery import Task, shared_task

try:
    from django_redis import LockError, get_redis_connection
except ImportError:

    def get_redis_connection(name):
        """Dummy implementation"""
        raise NotImplementedError

    class LockError(RuntimeError):  # type: ignore
        """Dummy exception"""


logger = logging.getLogger(__name__)
SignalReceiver = TypeVar("SignalReceiver", bound=Callable)


class _Receiver:
    def __init__(
        self,
        *,
        signal: Type[Signal],
        func: Callable,
        parameters: FrozenSet[str],
        coalesce: bool,
        queue: Optional[str],
        lock_timeout: int,
    ):
        self.signal = signal
        self.func = func
        self.parameters = parameters
        self.coalesce = coalesce
        self.lock_timeout = lock_timeout

        autoretry_for = (LockError,) if lock_timeout > 0 else None

        self._task: Task = shared_task(
            queue=queue,
            name=f"{signal.__name__}.receiver.{func.__module__}.{func.__qualname__}",
            typing=False,
            autoretry_for=autoretry_for,
        )(self.task)

    def task(self, payload: Dict[str, Any], timestamp: float) -> None:
        """
        Body of our celery task, called by the celery worker
        """

        with self.lock(payload):
            start_time = time.monotonic()
            if self.should_call(timestamp, payload):
                self.func(**payload)
            end_time = time.monotonic()

            if self.lock_timeout >= 0 and end_time - start_time > self.lock_timeout:
                logger.error(
                    "Task %s took longer to execute than lock timeout %s",
                    self._task.name,
                    self.lock_timeout,
                )

    @contextlib.contextmanager
    def lock(self, payload: Dict[str, Any]) -> Iterator[None]:
        """
        Attempt to take a lock for this task, if requested.
        """

        if self.lock_timeout <= 0:
            yield
            return

        try:
            redis = get_redis_connection("redis")
        except NotImplementedError:
            logger.warning("Redis not configured, not taking lock while running task")
            yield
            return

        cache_key = f"{self.get_cache_key(payload)}.lock"
        with redis.lock(cache_key, timeout=self.lock_timeout, blocking_timeout=0):
            yield

    def should_call(self, timestamp: float, payload: Dict[str, Any]) -> bool:
        """
        Check if the task should be called based on the coalesce setting.
        """

        if not self.coalesce:
            return True

        try:
            redis = get_redis_connection("redis")
        except NotImplementedError:
            logger.warning("Redis not configured, not coalescing task")
            return True

        cache_key = f"{self.get_cache_key(payload)}.last_run_time"

        def _check_last_run(pipe) -> bool:
            """
            Check the last called time in an atomic manner. This method may
            be called multiple times by the redis client.
            """

            try:
                last_call_time = float(pipe.get(cache_key))
            except (ValueError, TypeError):
                return True

            if last_call_time > timestamp:
                return False

            seconds, ms = pipe.time()
            now = seconds + ms / 1_000_000

            pipe.multi()
            pipe.set(cache_key, now)

            return True

        return redis.transaction(_check_last_run, cache_key, value_from_callable=True)

    def delay(self, payload: Dict[str, Any], timestamp: float) -> None:
        """
        Schedule the celery task to be run.
        """

        # Extract the parameters this receiver is interested in
        params = {
            key: value for key, value in payload.items() if key in self.parameters
        }
        self._task.delay(payload=params, timestamp=timestamp)

    def get_cache_key(self, payload: Dict[str, Any]) -> str:
        """
        Get a cache key that's unique for this receiever with the given payload
        """

        params = ", ".join(f"{key}={value}" for key, value in payload.items())
        return f"{self.func.__module__}.{self.func.__qualname__}({params})"


class SignalRegistry(type):
    """
    Metaclass for signals. Keeps track of all defined signals
    """

    receivers: List[_Receiver]
    parameters: Mapping[str, Type]

    _signals: Dict[str, Type[Signal]] = {}

    def __init__(
        cls, name: str, bases: Tuple[Type, ...], namespace: Dict[str, Any]
    ) -> None:
        """Register signal classes when they are defined"""

        if name in SignalRegistry._signals:
            raise TypeError(f"A signal named {name} has already been defined")

        if bases:
            SignalRegistry._signals[name] = cls  # type: ignore

        super().__init__(name, bases, namespace)

    def __call__(cls, *args: Any, **kwargs: Any) -> None:
        """Block instansiation of Signal classes"""

        raise RuntimeError("Signals cannot be initialized")


class Signal(metaclass=SignalRegistry):
    """
    Base class used to define a signal.
    """

    @classmethod
    def send(cls: Type[Signal], **payload: Any) -> None:
        """
        Send the signal. The payload will be verified against the defined
        structure and then a celery task is scheduled for each receiver.

        Arguments:
            payload:
        """

        try:
            redis = get_redis_connection("redis")
            seconds, ms = redis.time()
            timestamp = seconds + ms / 1_000_000
        except NotImplementedError:
            timestamp = time.time()

        cls._check_paylaod(payload)

        for receiver in cls.receivers:
            try:
                receiver.delay(payload=payload, timestamp=timestamp)
            except Exception:
                logger.exception(
                    "An error occured when calling %s receiver %s",
                    cls.__name__,
                    receiver.func.__name__,
                )

    @classmethod
    def receiver(
        cls,
        *,
        queue: Optional[str] = None,
        coalesce: bool = False,
        lock_timeout: int = 0,
    ) -> Callable[[SignalReceiver], SignalReceiver]:
        """
        Decorator to register a receiver for the signal.

        Arguments:
            queue:
                Which Celery queue to call this handler from

            coalesce:
                Do not call receiver again if the receiver has already been
                called with the same parameters after the signal was sent.

            lock:
                If set to true a redis-lock will be held while running the task

            lock_timeout:
                Maximum number of seconds to hold the lock
        """

        def _inner(func: SignalReceiver) -> SignalReceiver:

            parameters = cls._get_receiver_parameters(func)

            cls.receivers.append(
                _Receiver(
                    signal=cls,
                    func=func,
                    parameters=parameters,
                    coalesce=coalesce,
                    queue=queue,
                    lock_timeout=lock_timeout,
                )
            )
            return func

        return _inner

    ####################
    # Internal helpers #
    ####################

    def __init_subclass__(cls) -> None:
        """
        Initialize a Signal subclass.
        """

        super().__init_subclass__()

        # Give each class has its own list of receivers
        cls.receivers = []

        # Build a list of parameters from the inherited annotations
        cls.parameters = get_type_hints(cls)

    @classmethod
    def _check_paylaod(cls, payload: Mapping[str, Any]) -> None:
        """
        Check that a given payload is valid
        """

        # Check if any extra params were given
        superfluous_params = [name for name in payload if name not in cls.parameters]
        if superfluous_params:
            raise TypeError(
                f"{cls.__name__} does not take these parameters: "
                f"{', '.join(superfluous_params)}"
            )

        # Check if any params are missing
        missing_params = [name for name in cls.parameters if name not in payload]
        if missing_params:
            raise TypeError(
                f"Missing parameters to {cls.__name__}.send: "
                f"{', '.join(missing_params)}"
            )

        # Check that all params have the right type
        invalid_types = [
            f"{name} (expected {expected.__name__}, got "
            f"{type(payload[name]).__name__})"
            for name, expected in cls.parameters.items()
            if not isinstance(payload[name], expected)
        ]
        if invalid_types:
            raise TypeError(
                f"Invalid types for these parameters to {cls.__name__}: "
                f"{','.join(invalid_types)}"
            )

    @classmethod
    def _get_receiver_parameters(cls, func: Callable) -> FrozenSet[str]:
        """
        Check that the signature of a receiver function is compatible with the
        signal parameters.
        """

        type_hints = get_type_hints(func)
        parameters = inspect.signature(func).parameters

        # Check that the function signature matches the signal payload
        superfluous_params = [
            name for name in parameters.keys() if name not in cls.parameters
        ]
        if superfluous_params:
            raise TypeError(
                f"{cls.__name__} does not provide these parameters: "
                f"{', '.join(superfluous_params)}"
            )

        # Check that the parameter types matches the signal payload types
        invalid_types = [
            f"{name} (expected {cls.parameters[name].__name__}, "
            f"got {parameter.annotation.__name__})"
            for name, parameter in parameters.items()
            if not (
                parameter.annotation is inspect.Parameter.empty
                or issubclass(type_hints[name], cls.parameters[name])
            )
        ]
        if invalid_types:
            raise TypeError(
                f"Incorrect parameter types for {cls.__name__} handler: "
                f"{','.join(invalid_types)}"
            )

        return frozenset(parameters.keys())
