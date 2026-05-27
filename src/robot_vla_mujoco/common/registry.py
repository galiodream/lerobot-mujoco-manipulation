"""Generic registry for pluggable components."""

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None):
        def decorator(fn: Callable[..., Any]):
            k = key if key is not None else fn.__name__
            self._items[k] = fn
            return fn
        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            available = list(self._items.keys())
            raise KeyError(f"'{key}' not found in {self.name} registry. Available: {available}")
        return self._items[key]

    def keys(self):
        return list(self._items.keys())
