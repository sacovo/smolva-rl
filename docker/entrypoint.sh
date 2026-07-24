#!/bin/bash
set -e

# Activate the virtual environment
if [ -f "/work/.venv/bin/activate" ]; then
    source /work/.venv/bin/activate
fi

# Execute the CMD or run-time arguments
exec "$@"
