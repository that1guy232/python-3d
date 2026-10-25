"""Explicit ownership for resources that must be released before context exit."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")


class ResourceScope:
    """Release registered resources once, in reverse acquisition order.

    A failed cleanup never prevents the remaining resources from being released.
    Games choose what they own; the scope supplies only lifetime mechanics.
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], None]] = []
        self._owned_ids: set[int] = set()
        self._disposed = False

    def own(self, resource: T, disposer: Callable[[T], None] | None = None) -> T:
        """Register a disposable object (or an explicit releaser) and return it."""
        if self._disposed:
            raise RuntimeError("cannot acquire resources in a disposed scope")
        if resource is None or id(resource) in self._owned_ids:
            return resource
        release = getattr(resource, "dispose", None) if disposer is None else None
        if disposer is None and not callable(release):
            raise TypeError("resource must provide dispose() or an explicit disposer")
        if disposer is None:
            self._callbacks.append(release)
        else:
            self._callbacks.append(lambda: disposer(resource))
        self._owned_ids.add(id(resource))
        return resource

    def defer(self, callback: Callable, *args, **kwargs) -> None:
        """Register a cleanup action, including actions without a resource object."""
        if self._disposed:
            raise RuntimeError("cannot acquire resources in a disposed scope")
        self._callbacks.append(lambda: callback(*args, **kwargs))

    def dispose(self) -> None:
        """Attempt every release, then raise the first failure, if any."""
        if self._disposed:
            return
        self._disposed = True
        callbacks, self._callbacks = self._callbacks, []
        self._owned_ids.clear()
        error = None
        for callback in reversed(callbacks):
            try:
                callback()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    @contextmanager
    def rollback_on_error(self) -> Iterator[ResourceScope]:
        """Retain resources on success; release them if construction fails."""
        try:
            yield self
        except BaseException:
            try:
                self.dispose()
            except BaseException:
                # Preserve the construction error after attempting every release.
                pass
            raise
