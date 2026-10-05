#!/usr/bin/env python3
"""
CI / Pre-commit Linter for Markdown KaTeX & Mermaid Syntax
Prevents:
1. \\tag{...} which triggers "KaTeX parse error: \\tag works only in display equations"
2. file:/// local URI links
3. Half-open interval brackets in Mermaid labels like [0, n) or [n, n+Delta_n)
"""

import sys
import os
import re
import argparse

def lint_file(fpath, auto_fix=False):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []
    modified_content = content

    # Strip code blocks and inline code for \tag check
    no_code_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    no_code_content = re.sub(r'`.*?`', '', no_code_content)

    # 1. Check for \\tag{...}
    tag_matches = list(re.finditer(r'\\tag\{([^}]+)\}', no_code_content))
    if tag_matches:
        for m in tag_matches:
            tag_val = m.group(1)
            line_no = no_code_content[:m.start()].count("\n") + 1
            issues.append(f"Line {line_no}: Found '\\tag{{{tag_val}}}' (triggers 'KaTeX parse error: \\tag works only in display equations'). Use '\\qquad ({tag_val})' instead.")
        if auto_fix:
            modified_content = re.sub(r'(?<!`)\\tag\{([^}]+)\}(?!`)', r'\\qquad (\1)', modified_content)

    # 2. Check for \\left\\{ or \\right\\} which gets unescaped by CommonMark into \\left{ causing delimiter error
    delim_matches = list(re.finditer(r'\\(left|right)\s*\\([{}])', no_code_content))
    if delim_matches:
        for m in delim_matches:
            line_no = no_code_content[:m.start()].count("\n") + 1
            issues.append(f"Line {line_no}: Found '\\{m.group(1)}\\{m.group(2)}'. GFM unescapes this to '\\{m.group(1)}{m.group(2)}' causing 'Missing or unrecognized delimiter for \\left'. Use '\\{m.group(1)}\\{'lbrace' if m.group(2)=='{' else 'rbrace'}' instead.")
        if auto_fix:
            modified_content = re.sub(r'\\left\s*\\\{', r'\\left\\lbrace', modified_content)
            modified_content = re.sub(r'\\right\s*\\\}', r'\\right\\rbrace', modified_content)

    # 3. Check for indented $$ display blocks (must be at column 0 in GFM)
    for idx, line in enumerate(content.splitlines()):
        if re.match(r'^\s{1,}\$\$\s*$', line):
            issues.append(f"Line {idx+1}: Indented display math '$$'. In GFM, display math blocks must start at column 0 without indentation.")

    # 4. Check for local file:/// URIs
    if "file:///" in content:
        for idx, line in enumerate(content.splitlines()):
            if "file:///" in line:
                issues.append(f"Line {idx+1}: Contains local URI 'file:///' which breaks outside local machine.")

    # 5. Check for Mermaid half-open brackets inside label quotes
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    for b_idx, block in enumerate(mermaid_blocks):
        for l_idx, line in enumerate(block.splitlines()):
            opens = line.count('[')
            closes = line.count(']')
            if opens != closes:
                issues.append(f"Mermaid Block #{b_idx+1}, Line {l_idx+1}: Unbalanced brackets '[' vs ']' in: {line.strip()}")

    if auto_fix and modified_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print(f"✅ Auto-fixed issue(s) in {fpath}")

    return issues

def main():
    parser = argparse.ArgumentParser(description="Lint Markdown files for KaTeX and Mermaid compatibility.")
    parser.add_argument("--fix", action="store_true", help="Automatically replace \\tag{...} with \\qquad (...)")
    parser.add_argument("paths", nargs="*", default=["docs", "papers", "README.md"], help="Directories or files to check")
    args = parser.parse_args()

    target_files = []
    for p in args.paths:
        if os.path.isfile(p) and p.endswith(".md"):
            target_files.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".md"):
                        target_files.append(os.path.join(root, f))

    total_issues = 0
    for fpath in target_files:
        issues = lint_file(fpath, auto_fix=args.fix)
        if issues:
            total_issues += len(issues)
            print(f"❌ {fpath}:")
            for iss in issues:
                print(f"   • {iss}")

    if total_issues == 0:
        print(f"✨ All {len(target_files)} Markdown files passed KaTeX & Mermaid linter! Zero \\tag, zero file:/// links.")
        sys.exit(0)
    else:
        print(f"\n⚠️ Total issues found: {total_issues}")
        if not args.fix:
            print("Tip: Run with --fix to automatically convert \\tag{...} to \\qquad (...)")
        sys.exit(1 if not args.fix else 0)

if __name__ == "__main__":
    main()
