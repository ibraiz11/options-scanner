#!/usr/bin/env bash
# Source local secrets, then run the server. Safer than putting Alpaca keys
# in ~/.zshrc because they're confined to this project and to this command.
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env.local ]]; then
  # `set -a` exports every variable set by the following file; `set +a` turns
  # that off so we don't accidentally export every later variable in this script.
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
else
  echo "WARNING: .env.local not found." >&2
  echo "Copy .env.example to .env.local and fill in your Alpaca paper keys." >&2
  echo "Continuing in dry-run mode (no real broker)." >&2
fi

# Refuse to run if .env.local exists but is world-readable — that's a foot-gun.
if [[ -f .env.local ]]; then
  perms=$(stat -f "%Lp" .env.local 2>/dev/null || stat -c "%a" .env.local)
  if [[ "$perms" != "600" && "$perms" != "400" ]]; then
    echo "WARNING: .env.local permissions are $perms (other users can read it)." >&2
    echo "Tightening to 600 (owner read/write only)." >&2
    chmod 600 .env.local
  fi
fi

# Build if missing or stale, then exec so signals (Ctrl-C, SIGTERM) reach the binary.
cargo build --release
exec ./target/release/options-scanner
