#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
VERSION_HEADER="$REPOSITORY_ROOT/QSVPipeline/rgy_version.h"

VERSION_LINES=$(sed -n 's/^#define VER_STR_FILEVERSION[ \t]*"\([^"]*\)".*$/\1/p' "$VERSION_HEADER" | tr -d '\r')
VERSION_COUNT=$(printf '%s\n' "$VERSION_LINES" | awk 'NF { count++ } END { print count + 0 }')
if [ "$VERSION_COUNT" -ne 1 ]; then
    echo "ERROR: VER_STR_FILEVERSION は1個だけ定義してください。" >&2
    exit 1
fi

VERSION=$VERSION_LINES
if ! printf '%s\n' "$VERSION" | grep -Eq '^[0-9]+(\.[0-9]+)+$'; then
    echo "ERROR: VER_STR_FILEVERSION の値が不正です: '$VERSION'" >&2
    exit 1
fi

printf '%s\n' "$VERSION"
