#!/bin/bash
# Start the ping monitoring GUI dashboard
# Note: Requires root privileges for ICMP ping

cd "$(dirname "$0")"

if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges for ICMP ping."
    echo "Restarting with sudo..."
    sudo "$0" "$@"
    exit $?
fi

python3 gui_monitor.py
