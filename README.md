# Multi-Target Ping Monitor

Full-screen terminal dashboard for monitoring multiple network targets with real-time ping statistics and historical charts.

## Features

- **Real-time monitoring**
- **Full-screen terminal UI** with auto-refresh
- **Live charts** showing ping latency over time
- **Color-coded status indicators** (green=online, red=offline)
- **Packet loss tracking** and statistics
- **48-hour historical data** retention
- **5-second ping interval** for responsive monitoring

## Requirements

- Ubuntu 24 (or compatible Linux)
- Python 3.10+
- Root privileges (for ICMP ping)

## Installation

Dependencies are already installed:
- `rich` - Terminal UI framework
- `ping3` - Python ICMP ping library

## Usage

### Quick Start

```bash
sudo ./start.sh
```

Or run directly:

```bash
sudo python3 monitor.py
```

### Exit

Press `Ctrl+C` to exit the dashboard

## Configuration

Edit `config.json` to customize:

- `ping_interval`: Seconds between pings (default: 5)
- `chart_time_window_minutes`: Minutes of data visible on charts (default: 30)
- `history_retention_hours`: Hours to keep historical data (default: 48)
- `timeout_seconds`: Ping timeout (default: 2)
- `targets`: List of IP addresses and names to monitor

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Summary Table                             │
│  Shows all targets with current status and statistics       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Target 1 Chart (30 minutes of ping history)                │
│  Green bars = successful pings, Red bars = failed pings     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Target 2 Chart                                              │
└─────────────────────────────────────────────────────────────┘
...
```

## Statistics Displayed

- **Status**: Online/Offline indicator
- **Current**: Most recent ping RTT
- **Avg**: Average RTT
- **Min**: Minimum RTT
- **Max**: Maximum RTT
- **Loss %**: Packet loss percentage

## Troubleshooting

### "Operation not permitted" error
- Make sure you're running with `sudo` (root privileges required for ICMP)

### No data showing
- Wait 5-10 seconds for initial ping data to collect
- Check that target IPs are reachable

### Terminal too small
- Resize your terminal to at least 120x40 for best viewing
- Use full-screen terminal mode
