#!/usr/bin/env bash
# Launch Lab 1 in Webots.
# On macOS the binary must be invoked by its full path. A bare "webots" on
# PATH makes argv[0] relative, and Webots then looks for its bundle under the
# current project (the qt_warning_filters.conf error).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBOTS_APP="${WEBOTS_APP:-/Applications/Webots.app}"
WEBOTS_BIN="$WEBOTS_APP/Contents/MacOS/webots"

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "Webots not found at $WEBOTS_BIN" >&2
  echo "Install Webots or set WEBOTS_APP to your Webots.app path." >&2
  exit 1
fi

export WEBOTS_HOME="$WEBOTS_APP"
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH

cd "$REPO_ROOT/simulation"
exec "$WEBOTS_BIN" worlds/Lab_1_navigation.wbt
