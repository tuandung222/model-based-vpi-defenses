#!/usr/bin/env bash
# Install the Markdown/KaTeX linter as a git pre-commit hook (hooks are not versioned by git).
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cat > "$root/.git/hooks/pre-commit" <<'HOOK'
#!/usr/bin/env bash
python3 scripts/check_markdown_katex.py
HOOK
chmod +x "$root/.git/hooks/pre-commit"
echo "Installed pre-commit hook at $root/.git/hooks/pre-commit"
