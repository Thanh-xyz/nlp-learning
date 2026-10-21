import os
import sys
import re
import json
import tokenize
import io
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def check_python_file(file_path):
    violations = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    reader = io.StringIO(content).readline
    tokens = []
    try:
        for tok in tokenize.generate_tokens(reader):
            tokens.append(tok)
    except Exception as e:
        violations.append((1, f"Tokenization error: {e}"))
        return violations

    for i, tok in enumerate(tokens):
        tok_type = tok.type
        tok_string = tok.string
        start_line = tok.start[0]

        if tok_type == tokenize.COMMENT:
            if start_line == 1 and tok_string.startswith('#!'):
                continue
            violations.append((start_line, f"Comment found: {tok_string.strip()[:60]}"))

        elif tok_type == tokenize.STRING:
            stripped = tok_string.strip()
            is_multiline_docstring = (
                (stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6) or
                (stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6)
            )
            if is_multiline_docstring:
                prev_non_nl = None
                for prev in reversed(tokens[:i]):
                    if prev.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                        prev_non_nl = prev
                        break
                if prev_non_nl is None or prev_non_nl.string in (':', ';') or prev_non_nl.type in (tokenize.NEWLINE,):
                    violations.append((start_line, f"Docstring found: {stripped[:60]}..."))

    return violations

def check_ipynb_file(file_path):
    violations = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            violations.append((1, f"JSON parse error: {e}"))
            return violations

    cells = nb.get('cells', [])
    for cell_idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            code_text = "".join(lines) if isinstance(lines, list) else str(lines)
            
            reader = io.StringIO(code_text).readline
            try:
                tokens = list(tokenize.generate_tokens(reader))
                for i, tok in enumerate(tokens):
                    start_line = tok.start[0]
                    if tok.type == tokenize.COMMENT:
                        if start_line == 1 and tok.string.startswith('#!'):
                            continue
                        violations.append((start_line, f"Cell {cell_idx+1} comment: {tok.string.strip()[:60]}"))
                    elif tok.type == tokenize.STRING:
                        stripped = tok.string.strip()
                        if (stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) >= 6) or \
                           (stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) >= 6):
                            violations.append((start_line, f"Cell {cell_idx+1} docstring: {stripped[:60]}..."))
            except Exception:
                for line_idx, line in enumerate(code_text.splitlines()):
                    l_strip = line.strip()
                    if l_strip.startswith('#') and not l_strip.startswith('#!'):
                        violations.append((line_idx + 1, f"Cell {cell_idx+1} comment: {l_strip[:60]}"))

    return violations

def check_generic_file(file_path, comment_patterns):
    violations = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    in_block = False
    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        l_strip = line.strip()

        if in_block:
            violations.append((line_num, f"Block comment content: {l_strip[:60]}"))
            if '*/' in l_strip:
                in_block = False
            continue

        if '/*' in l_strip:
            violations.append((line_num, f"Block comment start: {l_strip[:60]}"))
            if '*/' not in l_strip:
                in_block = True
            continue

        for p in comment_patterns:
            if l_strip.startswith(p):
                if line_num == 1 and l_strip.startswith('#!'):
                    continue
                violations.append((line_num, f"Comment line: {l_strip[:60]}"))
                break

    return violations

def audit_directory(target_path):
    target = Path(target_path)
    if not target.exists():
        print(f"Path does not exist: {target_path}")
        return 1

    files_to_check = []
    if target.is_file():
        files_to_check.append(target)
    else:
        ignored_dirs = {'.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode'}
        for p in target.rglob('*'):
            if p.is_file() and not any(part in ignored_dirs for part in p.parts):
                files_to_check.append(p)

    supported_exts = {
        '.py': ('python', None),
        '.ipynb': ('ipynb', None),
        '.js': ('generic', ['//']),
        '.ts': ('generic', ['//']),
        '.jsx': ('generic', ['//']),
        '.tsx': ('generic', ['//']),
        '.java': ('generic', ['//']),
        '.c': ('generic', ['//']),
        '.cpp': ('generic', ['//']),
        '.h': ('generic', ['//']),
        '.hpp': ('generic', ['//']),
        '.sh': ('generic', ['#']),
        '.sql': ('generic', ['--', '#']),
    }

    total_violations = 0
    checked_files = 0

    print("=" * 60)
    print("NO-COMMENT RULE AUDIT")
    print(f"Target: {target_path}")
    print("=" * 60)

    for fpath in files_to_check:
        ext = fpath.suffix.lower()
        if ext not in supported_exts:
            continue

        checked_files += 1
        file_type, patterns = supported_exts[ext]
        if file_type == 'python':
            violations = check_python_file(fpath)
        elif file_type == 'ipynb':
            violations = check_ipynb_file(fpath)
        else:
            violations = check_generic_file(fpath, patterns)

        if violations:
            total_violations += len(violations)
            print(f"\n[FAIL] {fpath} ({len(violations)} violations):")
            for line_no, msg in violations[:10]:
                print(f"  Line {line_no}: {msg}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more.")
        else:
            print(f"[PASS] {fpath.name}")

    print("\n" + "=" * 60)
    print(f"AUDIT SUMMARY: Checked {checked_files} files.")
    if total_violations == 0:
        print("RESULT: ALL CLEAN - 0 comments or docstrings detected.")
        print("=" * 60)
        return 0
    else:
        print(f"RESULT: FAILED - {total_violations} comments/docstrings detected.")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(audit_directory(target))
