
from __future__ import annotations
from abc import ABC
from typing import Dict, Iterable, List

class FeatureModule(ABC):
    """
    Parent class for all feature modules.

    Each child defines a CLASS VARIABLE:
        METHOD_MAP: Dict[int, str] = {
            <method_id>: "<callable_name_on_child>",
            ...
        }

    Callers use: Child.extract(method_id, *args, **kwargs)
    This base class will dispatch to the target callable.
    """

    METHOD_MAP: Dict[int, str] = {}

    @classmethod
    def supported_methods(cls) -> List[int]:
        """List of supported method IDs for this module."""
        return sorted(cls.METHOD_MAP.keys())

    @classmethod
    def has_method(cls, method_id: int) -> bool:
        """Check whether this module supports a method id."""
        return method_id in cls.METHOD_MAP

    @classmethod
    def extract(cls, method_id: int, *args, **kwargs):
        """
        Uniform entrypoint: dispatch to the function registered for `method_id`.
        Target functions may be @staticmethod or @classmethod on the child.
        """
        if method_id not in cls.METHOD_MAP:
            raise ValueError(
                f"{cls.__name__} does not support method_id={method_id}. "
                f"Supported: {cls.supported_methods()}"
            )

        func_name = cls.METHOD_MAP[method_id]
        func = getattr(cls, func_name, None)

        if func is None:
            raise NotImplementedError(
                f"{cls.__name__}.METHOD_MAP points to '{func_name}', "
                "but that callable does not exist on the class."
            )

        return func(*args, **kwargs)


    @classmethod
    def describe_methods(cls) -> str:
        """Human-readable 'id -> callable' listing (handy for debugging/CLI)."""
        if not cls.METHOD_MAP:
            return f"{cls.__name__}: no methods registered."
        pairs = [f"{mid} -> {name}" for mid, name in sorted(cls.METHOD_MAP.items())]
        return f"{cls.__name__} methods: " + ", ".join(pairs)

    @classmethod
    def ensure_methods(cls, required: Iterable[int]) -> None:
        """Raise if any required ids are missing (useful in tests)."""
        missing = [mid for mid in required if mid not in cls.METHOD_MAP]
        if missing:
            raise AssertionError(
                f"{cls.__name__} is missing method ids: {missing}. "
                f"Currently has: {cls.supported_methods()}"
            )
