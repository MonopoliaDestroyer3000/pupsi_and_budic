"""Merge model function patches without rewriting unrelated source."""
import ast
import textwrap


def _span(node):
    return min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]), node.end_lineno


def _replacement(old, new, source_lines, patch_lines):
    start,end = _span(old); a,b = _span(new)
    indent = source_lines[start-1][:len(source_lines[start-1])-len(source_lines[start-1].lstrip())]
    text = textwrap.indent(textwrap.dedent(''.join(patch_lines[a-1:b])).rstrip(), indent) + '\n'
    if old.decorator_list and not new.decorator_list:
        text = ''.join(source_lines[start-1:old.lineno-1]) + text
    return start,end,text


def merge(source: str, patch: str) -> str:
    if len(source) > 200000 or len(patch) > 40000:
        raise ValueError('Patch size limit')
    original, changes = ast.parse(source), ast.parse(patch)
    allowed = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    targets = {node.name: node for node in original.body if isinstance(node, allowed)}
    if len(targets) != sum(isinstance(node, allowed) for node in original.body):
        raise ValueError('Ambiguous duplicate definitions')
    edits, seen = [], set()
    source_lines, patch_lines = source.splitlines(keepends=True), patch.splitlines(keepends=True)
    for node in changes.body:
        if not isinstance(node, allowed) or node.name not in targets or node.name in seen:
            raise ValueError('Patch must contain unique existing function/class definitions')
        seen.add(node.name)
        old = targets[node.name]
        if type(old) is not type(node): raise ValueError('Definition kind changed')
        if isinstance(node, ast.ClassDef):
            if ast.dump(ast.Module(body=old.bases+old.decorator_list, type_ignores=[])) != ast.dump(ast.Module(body=node.bases+node.decorator_list, type_ignores=[])) or ast.dump(ast.Tuple(elts=old.keywords)) != ast.dump(ast.Tuple(elts=node.keywords)):
                raise ValueError('Class header must remain unchanged')
            methods = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef,ast.AsyncFunctionDef))}
            method_names = set()
            for member in node.body:
                if isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant) and isinstance(member.value.value, str): continue
                if not isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) or member.name not in methods or member.name in method_names:
                    raise ValueError('Class patches contain existing methods only')
                method_names.add(member.name)
                edits.append(_replacement(methods[member.name],member,source_lines,patch_lines))
        else:
            edits.append(_replacement(old,node,source_lines,patch_lines))
    if not edits: raise ValueError('Empty patch')
    for start,end,text in sorted(edits,reverse=True): source_lines[start-1:end] = [text]
    result = ''.join(source_lines)
    ast.parse(result)
    return result
