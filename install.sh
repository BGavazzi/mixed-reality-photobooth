#!/usr/bin/env bash
# One-command setup for the mixed-reality photo booth, macOS/Linux.
#
#     ./install.sh              # photo booth
#     ./install.sh --resolume   # + the OSC bridge (Spout itself is Windows-only)
#     ./install.sh --all        # + hosted backends + the test suite
#
# Creates .venv if absent, installs the right requirement files, then runs
# doctor.py so you're told what's still missing. Safe to re-run.
#
# Note: Spout is a Windows technology, so `--resolume` here gets you the OSC
# trigger path but no Spout output. The environment markers in
# requirements-resolume.txt skip those packages automatically rather than
# failing the install.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

want_resolume=0; want_backends=0; want_dev=0
for arg in "$@"; do
  case "$arg" in
    --resolume) want_resolume=1 ;;
    --backends) want_backends=1 ;;
    --dev)      want_dev=1 ;;
    --all)      want_resolume=1; want_backends=1; want_dev=1 ;;
    -h|--help)  sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n=== %s ===\n' "$1"; }

step "checking Python"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    PYTHON="$candidate"
    echo "  using $candidate ($("$candidate" -c 'import platform; print(platform.python_version())'))"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "No Python 3.10+ found on PATH. Install one and re-run." >&2
  exit 1
fi

if [ -x "$VENV_PYTHON" ]; then
  step "reusing existing .venv"
else
  step "creating .venv"
  "$PYTHON" -m venv "$VENV_DIR"
fi

step "upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet

files=("requirements.txt")
[ "$want_resolume" = 1 ] && files+=("requirements-resolume.txt")
[ "$want_backends" = 1 ] && files+=("requirements-backends.txt")
[ "$want_dev" = 1 ] && files+=("requirements-test.txt")

for file in "${files[@]}"; do
  step "installing $file"
  if ! "$VENV_PYTHON" -m pip install -r "$REPO_DIR/$file"; then
    echo >&2
    echo "pip failed on $file. Re-run this script to retry -- partial installs resume cleanly." >&2
    exit 1
  fi
done

step "checking the result"
doctor_args=("doctor.py")
[ "$want_resolume" = 1 ] && doctor_args+=("--resolume")
[ "$want_backends" = 1 ] && doctor_args+=("--backends")
set +e
"$VENV_PYTHON" "${doctor_args[@]}"
doctor_exit=$?
set -e

printf '\nActivate the environment with:\n  source .venv/bin/activate\n'
printf '\nThe photo booth also needs ComfyUI plus two model files -- doctor.py above\n'
printf 'says whether it can see them. See the README Models section for links.\n'

exit $doctor_exit
