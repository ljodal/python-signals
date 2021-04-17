"""
Various helpers for the mypy plugin
"""

# Pylint doesn't understand mypy, so disable some checkers for this file
# pylint: disable=no-name-in-module

from typing import List, Optional, Union

from mypy.checker import TypeChecker
from mypy.fixup import TypeFixer
from mypy.nodes import (
    ARG_POS,
    MDEF,
    Argument,
    Block,
    ClassDef,
    Decorator,
    FuncDef,
    NameExpr,
    PassStmt,
    SymbolTableNode,
    Var,
)
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.semanal import set_callable_name
from mypy.types import (
    CallableType,
    JsonDict,
    Type,
    TypeType,
    TypeVarDef,
    deserialize_type,
)
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name


def add_method_to_class(
    api: SemanticAnalyzerPluginInterface,
    cls: ClassDef,
    name: str,
    args: List[Argument],
    return_type: Type,
    self_type: Optional[Type] = None,
    tvar_def: Optional[TypeVarDef] = None,
    is_classmethod: bool = False,
) -> None:
    """
    Adds a new method to a class definition.

    NOTE:
    Copied from mypy/plugins/common.py and extended with support for adding
    classmethods based on https://github.com/python/mypy/pull/7796

    """

    info = cls.info

    # First remove any previously generated methods with the same name
    # to avoid clashes and problems in the semantic analyzer.
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)

    self_type = self_type or fill_typevars(info)

    # Add either self or cls as the first argument
    if is_classmethod:
        first = Argument(Var("cls"), TypeType.make_normalized(self_type), None, ARG_POS)
    else:
        first = Argument(Var("self"), self_type, None, ARG_POS)

    args = [first] + args
    arg_types, arg_names, arg_kinds = [], [], []
    for arg in args:
        assert arg.type_annotation, "All arguments must be fully typed."
        arg_types.append(arg.type_annotation)
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)

    function_type = api.named_type("__builtins__.function")
    signature = CallableType(
        arg_types, arg_kinds, arg_names, return_type, function_type
    )
    if tvar_def:
        signature.variables = [tvar_def]

    func = FuncDef(name, args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func._fullname = info.fullname + "." + name  # pylint: disable=protected-access
    func.line = info.line
    func.is_class = is_classmethod

    # NOTE: we would like the plugin generated node to dominate, but we still
    # need to keep any existing definitions so they get semantically analyzed.
    if name in info.names:
        # Get a nice unique name instead.
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]

    if is_classmethod:
        func.is_decorated = True
        v = Var(name, func.type)
        v.info = info
        v._fullname = func._fullname  # pylint: disable=protected-access
        v.is_classmethod = True
        dec = Decorator(func, [NameExpr("classmethod")], v)

        dec.line = info.line
        sym = SymbolTableNode(MDEF, dec)
    else:
        sym = SymbolTableNode(MDEF, func)
    sym.plugin_generated = True

    info.names[name] = sym
    info.defn.defs.body.append(func)


def deserialize_and_fixup_type(
    data: Union[str, JsonDict], api: Union[SemanticAnalyzerPluginInterface, TypeChecker]
) -> Type:
    typ = deserialize_type(data)
    typ.accept(TypeFixer(api.modules, allow_missing=False))
    return typ
