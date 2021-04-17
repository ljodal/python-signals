"""
Mypy plugin for improving the type checking of the signals.
"""

# Pylint doesn't understand mypy, so disable some checkers for this file
# pylint: disable=no-name-in-module

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    Argument,
    AssignmentStmt,
    NameExpr,
    PlaceholderNode,
    TempNode,
    TypeInfo,
    Var,
)
from mypy.plugin import ClassDefContext, Plugin, SemanticAnalyzerPluginInterface
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.server.trigger import make_wildcard_trigger
from mypy.typeops import map_type_from_supertype
from mypy.types import AnyType, JsonDict, NoneType, Type, TypeOfAny, TypeVarType

from .utils import add_method_to_class

#####################
# Plugin definition #
#####################


class SignalsPlugin(Plugin):
    """
    Plugin definition for mypy, implements hooks called by mypy
    """

    def get_base_class_hook(
        self, fullname: str
    ) -> Optional[Callable[[ClassDefContext], None]]:
        """
        Callback when a subclass is defined. We use this to extract field
        details for defined signals.
        """

        # Check if the class inherits from Signal
        symbol = self.lookup_fully_qualified(fullname)
        if (
            symbol
            and symbol.node
            and isinstance(symbol.node, TypeInfo)
            and (symbol.node.has_base("celery_signals.Signal"))
        ):
            return make_signal

        return None


def plugin(version: str):
    """
    Entry point for plugin, called by mypy. The version argument is the mypy
    version that's currently executing.
    """

    return SignalsPlugin


################
# Plugin logic #
################

METADATA_KEY = "signals"


class SignalAttribute:
    def __init__(
        self, name: str, type_annotation: Type, has_default: bool, info: TypeInfo
    ) -> None:
        self.name = name
        self.type_annotation = type_annotation
        self.has_default = has_default
        self.info = info

    def __str__(self) -> str:
        return f"{self.name}: {self.type_annotation}"

    def __repr__(self) -> str:
        return f"<SignalAttribute {str(self)}>"

    def to_argument(self) -> Argument:
        return Argument(
            variable=self.to_var(),
            type_annotation=self.type_annotation,
            initializer=None,
            kind=ARG_NAMED_OPT if self.has_default else ARG_NAMED,
        )

    def to_var(self) -> Var:
        return Var(self.name, self.type_annotation)

    def serialize(self) -> JsonDict:
        return {
            "name": self.name,
            "type": self.type_annotation.serialize(),
            "has_default": self.has_default,
        }

    @classmethod
    def deserialize(
        cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface
    ) -> SignalAttribute:
        data = data.copy()
        typ = deserialize_and_fixup_type(data.pop("type"), api)
        return cls(type_annotation=typ, info=info, **data)

    def expand_typevar_from_subtype(self, sub_type: TypeInfo) -> None:
        """Expands type vars in the context of a subtype when an attribute is inherited
        from a generic super type."""
        if not isinstance(self.type_annotation, TypeVarType):
            return

        self.type_annotation = map_type_from_supertype(
            self.type_annotation, sub_type, self.info
        )


class SignalTransformer:
    def __init__(self, ctx: ClassDefContext) -> None:
        self._ctx = ctx

    def transform(self) -> None:
        """
        Apply transformations to the Signal class to improve type checking.
        """

        attributes = self.collect_attributes()
        if not attributes:
            # print(f"No attributes on {self._ctx.cls.name}")
            return

        # print(f"Attributes off {self._ctx.cls.name}: {attributes}")

        add_method_to_class(
            api=self._ctx.api,
            cls=self._ctx.cls,
            name="send",
            args=[attr.to_argument() for attr in attributes],
            return_type=NoneType(),
            is_classmethod=True,
        )

        self._ctx.cls.info.metadata[METADATA_KEY] = {
            "attributes": [attr.serialize() for attr in attributes],
        }

    def collect_attributes(self) -> Optional[List[SignalAttribute]]:
        """
        Collect attributes defined on the signal and its parents.
        """

        ctx = self._ctx
        cls = self._ctx.cls

        attributes: Dict[str, SignalAttribute] = {}

        # First, collect attributes belonging to the current class.
        for stmt in cls.defs.body:
            # Any assignment that doesn't use the new type declaration
            # syntax can be ignored out of hand.
            if not (isinstance(stmt, AssignmentStmt) and stmt.new_syntax):
                continue

            # a: int, b: str = 1, 'foo' is not supported syntax so we
            # don't have to worry about it.
            lhs = stmt.lvalues[0]
            if not isinstance(lhs, NameExpr):
                continue

            sym = cls.info.names.get(lhs.name)
            if sym is None:
                # This name is likely blocked by a star import. We don't need to
                # defer because defer() is already called by mark_incomplete().
                continue

            node = sym.node
            if isinstance(node, PlaceholderNode):
                # This node is not ready yet.
                return None

            assert isinstance(node, Var)

            # x: ClassVar[int] is ignored by signals.
            if node.is_classvar:
                continue

            field_name = lhs.name

            # Try to extract the field type or emit an error and fall back to Any
            field_type = sym.type
            if field_type is None:
                ctx.api.fail(f"Missing type for signal field {field_name}", node)
                field_type = AnyType(TypeOfAny.from_error)

            field_has_default = not isinstance(stmt.rvalue, TempNode)

            attributes[field_name] = SignalAttribute(
                name=field_name,
                type_annotation=field_type,
                has_default=field_has_default,
                info=cls.info,
            )

        # Next, collect attributes belonging to any class in the MRO as long as
        # those attributes weren't already collected.  This makes it possible to
        # overwrite attributes in subclasses.
        for info in cls.info.mro[1:-1]:
            if METADATA_KEY not in info.metadata:
                continue

            # Each class depends on the set of attributes in its dataclass ancestors.
            ctx.api.add_plugin_dependency(make_wildcard_trigger(info.fullname))

            for data in info.metadata[METADATA_KEY]["attributes"]:
                name: str = data["name"]
                if name not in attributes:
                    attr = SignalAttribute.deserialize(
                        info=info, data=data, api=ctx.api
                    )
                    attr.expand_typevar_from_subtype(ctx.cls.info)
                    attributes[name] = attr

        return list(attributes.values())


def make_signal(ctx: ClassDefContext) -> None:
    """
    Type check and build a signal.
    """

    SignalTransformer(ctx).transform()
