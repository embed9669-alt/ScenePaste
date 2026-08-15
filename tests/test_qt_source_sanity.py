from __future__ import annotations

import ast
import builtins
from pathlib import Path


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = child.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                names.add(arg.arg)
            if args.vararg:
                names.add(args.vararg.arg)
            if args.kwarg:
                names.add(args.kwarg.arg)
    return names


def _module_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != '*':
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
    return names


def test_factory_args_builder_has_no_undefined_bare_names():
    path = Path(__file__).resolve().parents[1] / 'compose_app_qt' / 'large_generate.py'
    text = path.read_text(encoding='utf-8')
    assert '_update_factory_hints' not in text
    tree = ast.parse(text)
    module_names = _module_defined_names(tree)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_args':
            target = node
            break
    assert target is not None

    local_names = _assigned_names(target)
    loaded = {
        node.id for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    # Methods can load attributes via self; only bare Name loads are checked.
    allowed = local_names | module_names | set(dir(builtins)) | {'self'}
    undefined = sorted(loaded - allowed)
    assert undefined == [], f'_args contains undefined names: {undefined}'
