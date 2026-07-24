#!/bin/bash
# Shared arg-parsing for submit_critic.sh / submit_recap.sh / submit_snapflow.sh.
# Source this after `shift`-ing off the wrapper's own positional args; it
# parses --partition/--num-gpus out of "$@" and sets PARTITION, NUM_GPUS
# (from the caller's existing defaults), and EXTRA_ARGS from the rest.

parse_submit_args() {
    REMAINING_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --partition)
                PARTITION="$2"; shift 2 ;;
            --num-gpus)
                NUM_GPUS="$2"; shift 2 ;;
            *)
                REMAINING_ARGS+=("$1"); shift ;;
        esac
    done
    EXTRA_ARGS="${REMAINING_ARGS[@]}"
}
