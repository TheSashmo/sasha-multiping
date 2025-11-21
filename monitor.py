#!/usr/bin/env python3
"""
Multi-target ping monitoring dashboard with full-screen terminal UI
"""

import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys

try:
    from ping3 import ping
except ImportError:
    print("Error: ping3 not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:
    print("Error: rich not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


@dataclass
class PingResult:
    """Single ping result"""
    timestamp: datetime
    rtt_ms: Optional[float]  # None if timeout/failed
    success: bool


@dataclass
class Target:
    """Monitoring target"""
    ip: str
    name: str


class PingMonitor:
    """Handles ping monitoring for multiple targets"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.targets = [Target(**t) for t in self.config['targets']]
        self.ping_interval = self.config['ping_interval']
        self.timeout = self.config['timeout_seconds']
        self.history_hours = self.config['history_retention_hours']
        self.chart_window_minutes = self.config['chart_time_window_minutes']

        # Store ping history: {ip: deque of PingResult}
        self.history: Dict[str, deque] = {
            target.ip: deque() for target in self.targets
        }

        self.running = False
        self.threads: List[threading.Thread] = []
        self.lock = threading.Lock()

    def _ping_worker(self, target: Target):
        """Worker thread that continuously pings a target"""
        while self.running:
            try:
                # Perform ping (returns time in seconds or None)
                result = ping(target.ip, timeout=self.timeout, unit='ms')

                ping_result = PingResult(
                    timestamp=datetime.now(),
                    rtt_ms=result if result is not None else None,
                    success=result is not None
                )

                with self.lock:
                    self.history[target.ip].append(ping_result)
                    self._cleanup_old_data(target.ip)

            except Exception as e:
                # On error, record as failed ping
                with self.lock:
                    self.history[target.ip].append(PingResult(
                        timestamp=datetime.now(),
                        rtt_ms=None,
                        success=False
                    ))

            time.sleep(self.ping_interval)

    def _cleanup_old_data(self, ip: str):
        """Remove data older than history_retention_hours"""
        cutoff = datetime.now() - timedelta(hours=self.history_hours)
        history = self.history[ip]

        while history and history[0].timestamp < cutoff:
            history.popleft()

    def start(self):
        """Start monitoring all targets"""
        self.running = True

        for target in self.targets:
            thread = threading.Thread(
                target=self._ping_worker,
                args=(target,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        """Stop all monitoring threads"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)

    def get_stats(self, ip: str) -> dict:
        """Calculate statistics for a target"""
        with self.lock:
            history = list(self.history[ip])

        if not history:
            return {
                'status': 'unknown',
                'current': None,
                'avg': None,
                'min': None,
                'max': None,
                'packet_loss': 0
            }

        # Get successful pings
        successful = [p.rtt_ms for p in history if p.success and p.rtt_ms is not None]
        total = len(history)
        failed = total - len(successful)

        current = history[-1].rtt_ms if history else None

        return {
            'status': 'online' if history[-1].success else 'offline',
            'current': current,
            'avg': sum(successful) / len(successful) if successful else None,
            'min': min(successful) if successful else None,
            'max': max(successful) if successful else None,
            'packet_loss': (failed / total * 100) if total > 0 else 0
        }

    def get_chart_data(self, ip: str) -> List[PingResult]:
        """Get ping data for chart (limited to chart_window_minutes)"""
        with self.lock:
            history = list(self.history[ip])

        cutoff = datetime.now() - timedelta(minutes=self.chart_window_minutes)
        return [p for p in history if p.timestamp >= cutoff]


class Dashboard:
    """Full-screen terminal dashboard"""

    def __init__(self, monitor: PingMonitor):
        self.monitor = monitor
        self.console = Console()

    def create_summary_table(self) -> Table:
        """Create the summary table showing all targets"""
        table = Table(
            title="Ping Monitor - Live Status",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Status", width=8)
        table.add_column("IP", width=18)
        table.add_column("Name", width=30)
        table.add_column("Current", width=10, justify="right")
        table.add_column("Avg", width=10, justify="right")
        table.add_column("Min", width=10, justify="right")
        table.add_column("Max", width=10, justify="right")
        table.add_column("Loss %", width=10, justify="right")

        for target in self.monitor.targets:
            stats = self.monitor.get_stats(target.ip)

            # Status indicator
            if stats['status'] == 'online':
                status = "[green]● ONLINE[/green]"
            elif stats['status'] == 'offline':
                status = "[red]● OFFLINE[/red]"
            else:
                status = "[yellow]● UNKNOWN[/yellow]"

            # Format RTT values
            current = f"{stats['current']:.1f}ms" if stats['current'] is not None else "-"
            avg = f"{stats['avg']:.1f}ms" if stats['avg'] is not None else "-"
            min_rtt = f"{stats['min']:.1f}ms" if stats['min'] is not None else "-"
            max_rtt = f"{stats['max']:.1f}ms" if stats['max'] is not None else "-"
            loss = f"{stats['packet_loss']:.1f}%"

            # Color code packet loss
            if stats['packet_loss'] > 10:
                loss = f"[red]{loss}[/red]"
            elif stats['packet_loss'] > 5:
                loss = f"[yellow]{loss}[/yellow]"
            else:
                loss = f"[green]{loss}[/green]"

            table.add_row(
                status,
                target.ip,
                target.name,
                current,
                avg,
                min_rtt,
                max_rtt,
                loss
            )

        return table

    def create_chart(self, target: Target, width: int = 100, height: int = 8) -> Panel:
        """Create a horizontal bar chart for a target"""
        chart_data = self.monitor.get_chart_data(target.ip)

        if not chart_data:
            return Panel("No data yet...", title=f"[cyan]{target.name}[/cyan] ({target.ip})")

        # Create simple ASCII chart
        chart_lines = []

        # Calculate max RTT for scaling (or use fixed scale)
        max_rtt = max([p.rtt_ms for p in chart_data if p.rtt_ms is not None], default=100)
        scale_max = max(max_rtt * 1.2, 50)  # At least 50ms scale

        # Create horizontal bars (simplified version)
        # We'll show last N data points that fit the width
        points_to_show = min(len(chart_data), width - 10)
        recent_data = chart_data[-points_to_show:] if points_to_show > 0 else chart_data

        # Build chart from bottom up (multiple rows for height)
        for row in range(height, 0, -1):
            threshold = (row / height) * scale_max
            line = ""

            for point in recent_data:
                if point.rtt_ms is None:
                    # Failed ping
                    line += "[red]│[/red]"
                elif point.rtt_ms >= threshold:
                    line += "[green]█[/green]"
                else:
                    line += " "

            chart_lines.append(line)

        # Add timeline at bottom
        timeline = "─" * len(recent_data)
        chart_lines.append(timeline)

        # Get stats for subtitle
        stats = self.monitor.get_stats(target.ip)
        subtitle = f"Avg: {stats['avg']:.1f}ms" if stats['avg'] else "No data"

        chart_text = Text.from_markup("\n".join(chart_lines))

        return Panel(
            chart_text,
            title=f"[cyan]{target.name}[/cyan] ({target.ip})",
            subtitle=subtitle,
            border_style="blue"
        )

    def generate_display(self) -> Layout:
        """Generate the full dashboard layout"""
        layout = Layout()

        # Split into header and body
        layout.split_column(
            Layout(name="header", size=len(self.monitor.targets) + 5),
            Layout(name="charts")
        )

        # Add summary table to header
        layout["header"].update(self.create_summary_table())

        # Add charts to body
        terminal_width = self.console.width
        chart_layouts = []

        for target in self.monitor.targets:
            chart_layouts.append(
                Layout(self.create_chart(target, width=terminal_width - 4, height=4))
            )

        # Split charts vertically
        if chart_layouts:
            layout["charts"].split_column(*chart_layouts)

        return layout

    def run(self):
        """Run the full-screen dashboard"""
        self.console.clear()

        with Live(
            self.generate_display(),
            console=self.console,
            screen=True,
            refresh_per_second=2
        ) as live:
            try:
                while True:
                    time.sleep(0.5)
                    live.update(self.generate_display())
            except KeyboardInterrupt:
                pass


def main():
    """Main entry point"""
    try:
        monitor = PingMonitor("config.json")
        dashboard = Dashboard(monitor)

        print("Starting ping monitor...")
        print(f"Monitoring {len(monitor.targets)} targets")
        print("Press Ctrl+C to exit\n")

        time.sleep(2)  # Give user time to read

        monitor.start()

        # Wait a moment for initial data
        time.sleep(monitor.ping_interval + 1)

        dashboard.run()

    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
    except FileNotFoundError:
        print("Error: config.json not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'monitor' in locals():
            monitor.stop()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
