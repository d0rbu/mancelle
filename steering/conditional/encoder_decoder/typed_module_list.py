"""A runtime-validated, statically typed ``torch.nn.ModuleList``."""

from collections.abc import Iterable, Iterator
from typing import Self, cast, overload

import torch.nn as nn


class TypedModuleList[ModuleT: nn.Module](nn.ModuleList):
    """ModuleList that validates one module type at construction and mutation."""

    def __init__(
        self,
        modules: Iterable[nn.Module],
        *,
        module_type: type[ModuleT],
        item_label: str,
    ) -> None:
        super().__init__()
        self._module_type = module_type
        self._item_label = item_label
        self.extend(modules)

    @property
    def module_type(self) -> type[ModuleT]:
        """Return the runtime type accepted by this list."""
        return self._module_type

    def _require_module(self, module: nn.Module, index: int) -> ModuleT:
        if not isinstance(module, self._module_type):
            raise TypeError(
                f"{self._item_label} {index} must be {self._module_type.__name__}, "
                f"got {type(module).__name__}"
            )
        return module

    def __iter__(self) -> Iterator[ModuleT]:
        return cast(Iterator[ModuleT], super().__iter__())

    @overload
    def __getitem__(self, idx: int) -> ModuleT: ...

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[ModuleT]": ...

    def __getitem__(
        self, idx: int | slice
    ) -> "ModuleT | TypedModuleList[ModuleT]":
        if isinstance(idx, slice):
            return TypedModuleList(
                list(self)[idx],
                module_type=self._module_type,
                item_label=self._item_label,
            )
        return cast(ModuleT, super().__getitem__(idx))

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        super().__setitem__(idx, self._require_module(module, idx))

    def append(self, module: nn.Module) -> Self:
        super().append(self._require_module(module, len(self)))
        return self

    def extend(self, modules: Iterable[nn.Module]) -> Self:
        offset = len(self)
        typed_modules = tuple(
            self._require_module(module, offset + index)
            for index, module in enumerate(modules)
        )
        super().extend(typed_modules)
        return self

    def insert(self, index: int, module: nn.Module) -> None:
        super().insert(index, self._require_module(module, index))
