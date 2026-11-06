#!/usr/bin/env bash
# Run once after cloning: registers the commit message template and the format-check hook.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config commit.template .githooks/commit-template
git config core.hooksPath .githooks
echo "✓ commit.template = .githooks/commit-template"
echo "✓ core.hooksPath  = .githooks"
