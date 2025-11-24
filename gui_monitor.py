#!/usr/bin/env python3
"""
Multi-target ping monitoring GUI desktop application
Full-screen dashboard with thin line charts - PingPlotter style

Version: 1.05 (Flashing Alert for Failures)
"""

__version__ = "1.05"

import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk
import sys
import argparse
import socket
import netifaces
import subprocess
import re
from flask import Flask, render_template_string, jsonify, request

try:
    from ping3 import ping
except ImportError:
    print("Error: ping3 not installed. Run: pip3 install -r requirements.txt")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib not installed. Run: pip3 install -r requirements.txt")
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
    interface: Optional[str] = None  # Network interface to use for ping
    consecutive_loss_threshold: int = 1  # Number of consecutive losses before marking as down (1 = plot all)
    show_average: bool = True  # Show moving average line on chart


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
        self.line_thickness = self.config.get('line_thickness', 3.2)
        self.num_columns = self.config.get('num_columns', 2)
        self.grid_enabled = self.config.get('grid_enabled', True)

        # Store ping history: {ip: deque of PingResult}
        self.history: Dict[str, deque] = {
            target.ip: deque() for target in self.targets
        }

        # Track consecutive losses per target for threshold logic
        self.consecutive_losses: Dict[str, int] = {
            target.ip: 0 for target in self.targets
        }
        self.last_successful_rtt: Dict[str, Optional[float]] = {
            target.ip: None for target in self.targets
        }

        # Track alert state per target (for flashing alerts)
        self.in_alert_state: Dict[str, bool] = {
            target.ip: False for target in self.targets
        }

        self.running = False
        self.threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        self.targets_changed = False  # Flag to notify GUI of target changes

    def _ping_worker(self, target: Target):
        """Worker thread that runs continuous ping subprocess for a target"""
        # Start continuous ping process
        # Using -i for interval, -W for timeout
        cmd = ['ping', '-i', str(self.ping_interval), '-W', str(int(self.timeout)), target.ip]

        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )

            # Regex to parse ping output: "64 bytes from 8.8.8.8: ... time=12.3 ms"
            time_pattern = re.compile(r'time[=<](\d+\.?\d*)\s*ms')
            timeout_pattern = re.compile(r'no answer yet|timed out|100% packet loss|unreachable', re.IGNORECASE)

            print(f"[v1.03] Started continuous ping for {target.name} ({target.ip})")

            # Check if process started successfully
            if process.poll() is not None:
                stderr_output = process.stderr.read() if process.stderr else "No stderr"
                print(f"[v1.03] ERROR: Ping process died immediately for {target.ip}. Stderr: {stderr_output}")
                return

            # Read output line by line
            line_count = 0
            for line in iter(process.stdout.readline, ''):
                if not self.running:
                    print(f"[v1.03] DEBUG: Ping worker stopping (self.running=False) for {target.ip}")
                    break

                line_count += 1
                if line_count == 1:
                    print(f"[v1.03] DEBUG: First line received for {target.ip}: {line[:50]}")

                line = line.strip()
                if not line:
                    continue

                # Try to extract RTT from successful ping
                time_match = time_pattern.search(line)
                is_timeout = timeout_pattern.search(line)

                result = None
                if time_match:
                    try:
                        result = float(time_match.group(1))
                    except ValueError:
                        result = None

                # Process the result (same logic as before)
                with self.lock:
                    # Ensure history and tracking exist for this target
                    if target.ip not in self.history:
                        self.history[target.ip] = deque()
                    if target.ip not in self.consecutive_losses:
                        self.consecutive_losses[target.ip] = 0
                    if target.ip not in self.last_successful_rtt:
                        self.last_successful_rtt[target.ip] = None
                    if target.ip not in self.in_alert_state:
                        self.in_alert_state[target.ip] = False

                    if result is not None:
                        # Ping succeeded - clear alert state
                        self.consecutive_losses[target.ip] = 0
                        self.last_successful_rtt[target.ip] = result
                        self.in_alert_state[target.ip] = False

                        ping_result = PingResult(
                            timestamp=datetime.now(),
                            rtt_ms=result,
                            success=True
                        )
                        self.history[target.ip].append(ping_result)
                    elif is_timeout or (time_match is None and 'from' not in line and 'PING' not in line):
                        # Ping failed or timeout detected
                        self.consecutive_losses[target.ip] += 1
                        threshold = target.consecutive_loss_threshold

                        if self.consecutive_losses[target.ip] < threshold:
                            # Below threshold: mask the loss by duplicating last successful value
                            ping_result = PingResult(
                                timestamp=datetime.now(),
                                rtt_ms=self.last_successful_rtt[target.ip],
                                success=True  # Masked as success
                            )
                            self.history[target.ip].append(ping_result)
                        else:
                            # At or above threshold: record as actual failure and enter alert state
                            ping_result = PingResult(
                                timestamp=datetime.now(),
                                rtt_ms=None,
                                success=False
                            )
                            self.history[target.ip].append(ping_result)

                            # Enter alert state for flashing
                            self.in_alert_state[target.ip] = True

                            # Retroactively mark previous (threshold-1) masked losses as failures
                            if self.consecutive_losses[target.ip] == threshold and threshold > 1:
                                history = self.history[target.ip]
                                # Go backwards from second-to-last (just before the one we just added)
                                for i in range(len(history) - 2, max(-1, len(history) - threshold - 1), -1):
                                    if i >= 0 and history[i].success and history[i].rtt_ms == self.last_successful_rtt[target.ip]:
                                        # This looks like a masked loss, convert it to real failure
                                        history[i].rtt_ms = None
                                        history[i].success = False

                    self._cleanup_old_data(target.ip)

            # Loop exited
            print(f"[v1.03] DEBUG: Readline loop exited for {target.ip}. Lines read: {line_count}. Process returncode: {process.poll()}")

        except Exception as e:
            print(f"[v1.03] Ping worker error for {target.ip}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup: terminate the ping process
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            print(f"[v1.03] Stopped continuous ping for {target.name} ({target.ip})")

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
            # Ensure history exists for this IP
            if ip not in self.history:
                self.history[ip] = deque()
            history = list(self.history[ip])

        if not history:
            return {
                'status': 'unknown',
                'current': None,
                'avg': None,
                'min': None,
                'max': None,
                'packet_loss': 0,
                'count': 0
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
            'packet_loss': (failed / total * 100) if total > 0 else 0,
            'count': total
        }

    def get_chart_data(self, ip: str) -> List[PingResult]:
        """Get ping data for chart (limited to chart_window_minutes)"""
        with self.lock:
            # Ensure history exists for this IP
            if ip not in self.history:
                self.history[ip] = deque()
            history = list(self.history[ip])

        cutoff = datetime.now() - timedelta(minutes=self.chart_window_minutes)
        return [p for p in history if p.timestamp >= cutoff]


class PingMonitorGUI:
    """GUI Desktop Application - PingPlotter style with thin lines"""

    def __init__(self, monitor: PingMonitor, font_scale: float = 1.0):
        self.monitor = monitor
        self.font_scale = font_scale
        self.root = tk.Tk()
        self.root.title("Multi-Target Ping Monitor")

        # Set window to fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#2d2d2d')

        # Bind keys
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        self.root.bind('<F11>', lambda e: self.root.attributes('-fullscreen', True))
        self.root.bind('q', lambda e: self.root.quit())

        # Main container
        self.main_frame = tk.Frame(self.root, bg='#2d2d2d')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Add IP/URL info label in bottom right corner
        self.create_access_info_label()

        # Track current targets to detect changes
        self.current_target_ips = set()

        # Get line thickness and columns from monitor
        self.line_thickness = monitor.line_thickness
        self.num_columns = monitor.num_columns

        # Flash state for alert tiles (toggles every 1 second)
        self.flash_on = False
        self.start_flash_timer()

        # Create grid of charts with EQUAL heights
        self.create_equal_chart_grid()

        # Start update loop
        self.update_display()

    def start_flash_timer(self):
        """Start the 1-second flash toggle timer for alert states"""
        self.flash_on = not self.flash_on
        # Schedule next toggle in 1 second (1000ms)
        self.root.after(1000, self.start_flash_timer)

    def get_network_ips(self):
        """Get all network IP addresses (excluding localhost)"""
        ips = []
        try:
            import netifaces
            for iface in netifaces.interfaces():
                try:
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            ip = addr.get('addr', '')
                            # Skip localhost and empty IPs
                            if ip and not ip.startswith('127.'):
                                ips.append(ip)
                except Exception:
                    continue
        except Exception as e:
            print(f"Error getting network interfaces: {e}")

        return ips if ips else ['localhost']

    def create_access_info_label(self):
        """Create a label in the bottom right corner showing IP/URL access info"""
        # Get initial IPs
        self.current_ips = self.get_network_ips()

        # Create label text
        urls = [f'http://{ip}:5000' for ip in self.current_ips]
        url_text = ' | '.join(urls)
        info_text = f'Web Control Panel: {url_text}'

        # Create label with smaller font
        font_size = int(9 * self.font_scale)
        self.access_info_label = tk.Label(
            self.root,
            text=info_text,
            bg='#2d2d2d',
            fg='#888888',
            font=('Courier', font_size),
            anchor='e',
            justify='right'
        )

        # Place in bottom right corner
        self.access_info_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-5)

        # Track last update time for IP refresh
        self.last_ip_update = 0

    def update_access_info_label(self):
        """Update the IP/URL label if network IPs have changed"""
        import time
        current_time = time.time()

        # Only check for IP changes every 10 seconds
        if current_time - self.last_ip_update < 10:
            return

        self.last_ip_update = current_time

        # Get current IPs
        new_ips = self.get_network_ips()

        # Update label if IPs have changed
        if new_ips != self.current_ips:
            self.current_ips = new_ips
            urls = [f'http://{ip}:5000' for ip in self.current_ips]
            url_text = ' | '.join(urls)
            info_text = f'Web Control Panel: {url_text}'
            self.access_info_label.config(text=info_text)
            print(f"Network IP updated to: {', '.join(self.current_ips)}")

    def truncate_title(self, name, ip, max_length=25):
        """Truncate title to prevent overlap with stats"""
        full_title = f"{name} ({ip})"
        if len(full_title) <= max_length:
            return full_title

        # Calculate space for IP (including parentheses and space)
        ip_space = len(ip) + 3  # " ()" = 3 chars

        # Calculate available space for name
        available_for_name = max_length - ip_space - 3  # -3 for "..."

        if available_for_name > 0:
            truncated_name = name[:available_for_name] + "..."
            return f"{truncated_name} ({ip})"
        else:
            # If even truncated name doesn't fit, just truncate the whole thing
            return full_title[:max_length-3] + "..."

    def create_equal_chart_grid(self):
        """Create grid with configurable columns"""
        num_targets = len(self.monitor.targets)
        num_rows = (num_targets + self.num_columns - 1) // self.num_columns  # Round up

        # Configure grid to have equal weight for all rows
        for i in range(num_rows):
            self.main_frame.grid_rowconfigure(i, weight=1)
        for i in range(self.num_columns):
            self.main_frame.grid_columnconfigure(i, weight=1)

        # Create charts in grid
        self.charts = {}
        for i, target in enumerate(self.monitor.targets):
            row = i // self.num_columns
            col = i % self.num_columns
            self.create_line_chart(target, row, col)

        # Update target tracking
        self.current_target_ips = {target.ip for target in self.monitor.targets}
        print(f"Chart grid created/updated with {len(self.current_target_ips)} targets")

    def create_line_chart(self, target: Target, row: int, col: int):
        """Create a thin-line chart for a single target with stats"""
        # Chart frame
        chart_frame = tk.Frame(self.main_frame, bg='#3a3a3a', relief=tk.FLAT)
        chart_frame.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)

        # Create matplotlib figure with more space for labels
        fig = Figure(figsize=(10, 1.8), dpi=100, facecolor='#3a3a3a')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#404040')

        # Style axes - minimal with scaled fonts
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(0.5)
        ax.tick_params(colors='#aaaaaa', labelsize=int(7*self.font_scale), length=3, width=0.5)
        ax.set_ylabel('ms', color='#aaaaaa', fontsize=int(8*self.font_scale))

        # Title in plot area (left side)
        ax.set_title(f"{target.name} ({target.ip})",
                    color='#ffffff', fontsize=int(9*self.font_scale), loc='left', pad=3)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.charts[target.ip] = {
            'fig': fig,
            'ax': ax,
            'canvas': canvas,
            'target': target
        }

    def get_line_color(self, rtt_ms: float) -> str:
        """Get color based on RTT thresholds"""
        if rtt_ms is None or rtt_ms == 0:
            return '#ff0000'  # Red for failed
        elif rtt_ms >= 200:
            return '#ff0000'  # Red for critical
        elif rtt_ms >= 100:
            return '#ffaa00'  # Orange for warning
        else:
            return '#00ff00'  # Green for good

    def update_charts(self):
        """Update all charts with WHITE LINE and COLOR ZONES"""
        now = datetime.now()
        time_window_start = now - timedelta(minutes=self.monitor.chart_window_minutes)

        for target in self.monitor.targets:
            chart_data = self.monitor.get_chart_data(target.ip)

            chart = self.charts[target.ip]
            ax = chart['ax']

            # Clear previous plot
            ax.clear()

            # Check if this target is in alert state (for flashing)
            in_alert = self.monitor.in_alert_state.get(target.ip, False)
            flash_active = in_alert and self.flash_on

            # Re-apply title after clear (left side) - flash red when in alert
            target_obj = chart['target']
            title_color = '#ff0000' if flash_active else '#ffffff'
            ax.set_title(f"{target_obj.name} ({target_obj.ip})",
                        color=title_color, fontsize=int(9*self.font_scale), loc='left', pad=3)

            # Determine Y limit and max RTT from ALL data in window
            # DYNAMIC Y-AXIS based on actual data with 25% headroom
            has_packet_loss = False
            if chart_data:
                # Check ALL pings in the window
                rtts_valid = [p.rtt_ms for p in chart_data if p.rtt_ms is not None]
                failed_pings = [p for p in chart_data if not p.success or p.rtt_ms is None]
                has_packet_loss = len(failed_pings) > 0

                if rtts_valid:
                    max_rtt = max(rtts_valid)
                    # DYNAMIC Y-AXIS: 1.25x the max RTT (25% headroom)
                    y_max = max_rtt * 1.25

                    # If we have packet loss, ensure y_max is at least 250 to show all zones
                    if has_packet_loss:
                        y_max = max(y_max, 250)
                else:
                    # No successful pings - ALL PACKET LOSS
                    max_rtt = 300  # Treat as critical
                    y_max = 250  # Show all zones
                    has_packet_loss = True
            else:
                max_rtt = 0
                y_max = 50  # Minimal default

            # DYNAMIC LAYERED BACKGROUND COLOR ZONES (PingPlotter style)
            # Packet loss is CRITICAL - ALWAYS show all three zones if there's ANY packet loss
            # BRIGHTER COLORS for better visibility

            # Green zone (0-100ms) - BRIGHT GREEN
            # Show if y_max allows it
            if y_max >= 100:
                ax.axhspan(0, 100, facecolor='#00aa00', alpha=0.25, zorder=1)
            else:
                # For very low latency, scale green zone to y_max
                ax.axhspan(0, y_max, facecolor='#00aa00', alpha=0.25, zorder=1)

            # Yellow zone (100-200ms) - BRIGHT YELLOW
            # Show if: data reached >100ms OR has packet loss
            if (max_rtt > 100 or has_packet_loss) and y_max >= 100:
                yellow_top = min(200, y_max)
                ax.axhspan(100, yellow_top, facecolor='#ffcc00', alpha=0.3, zorder=1)

            # Red zone (200ms+) - BRIGHT RED
            # Show if: data reached >200ms OR has packet loss
            if (max_rtt > 200 or has_packet_loss) and y_max >= 200:
                ax.axhspan(200, y_max, facecolor='#ff3333', alpha=0.3, zorder=1)

            if not chart_data:
                # Empty chart styling - flash background red when in alert
                ax.set_xlim(time_window_start, now)
                ax.set_ylim(0, y_max)
                bg_color = '#ff0000' if flash_active else '#404040'
                ax.set_facecolor(bg_color)
                for spine in ax.spines.values():
                    spine.set_color('#ff0000' if flash_active else '#666666')
                    spine.set_linewidth(2 if flash_active else 0.5)
                ax.tick_params(colors='#aaaaaa', labelsize=int(6*self.font_scale), length=2, width=0.5)
                ax.set_ylabel('ms', color='#aaaaaa', fontsize=int(7*self.font_scale))
                ax.grid(True, alpha=0.15, color='#666666', linewidth=0.3)


                chart['canvas'].draw()
                continue

            # Prepare data for WHITE LINE and PACKET LOSS
            success_times = []
            success_rtts = []
            failed_times = []

            for p in chart_data:
                if p.success and p.rtt_ms is not None:
                    success_times.append(p.timestamp)
                    success_rtts.append(p.rtt_ms)
                else:
                    # Track failed pings separately
                    failed_times.append(p.timestamp)

            # Plot PACKET LOSS as VERTICAL RED LINES (drawn first, behind white line)
            if failed_times:
                for t in failed_times:
                    ax.vlines(t, 0, y_max, colors='#ff0000', linewidth=self.monitor.line_thickness * 1.1, alpha=0.7, zorder=2)

            # Plot WHITE CONTINUOUS LINE for successful pings (on top)
            if success_times and success_rtts:
                ax.plot(success_times, success_rtts, color='#ffffff', linewidth=self.monitor.line_thickness, alpha=0.9, zorder=3)

                # Plot MOVING AVERAGE LINE (cumulative average in visible window)
                target_obj = chart['target']
                if target_obj.show_average and len(success_rtts) > 1:
                    # Calculate cumulative average for all points in window
                    cumulative_avg = sum(success_rtts) / len(success_rtts)
                    # Draw horizontal average line across the visible time window
                    ax.axhline(y=cumulative_avg, color='#6699ff', linestyle=':',
                              linewidth=self.monitor.line_thickness * 0.7, alpha=0.8, zorder=2.5)

            # FIXED TIME WINDOW
            ax.set_xlim(time_window_start, now)
            ax.set_ylim(0, y_max)

            # Add subtle threshold lines
            ax.axhline(y=100, color='#888888', linestyle='--', linewidth=0.5, alpha=0.4, zorder=2)
            ax.axhline(y=200, color='#888888', linestyle='--', linewidth=0.5, alpha=0.4, zorder=2)

            # Style - flash background and border red when in alert
            bg_color = '#ff0000' if flash_active else '#404040'
            ax.set_facecolor(bg_color)
            for spine in ax.spines.values():
                spine.set_color('#ff0000' if flash_active else '#666666')
                spine.set_linewidth(2 if flash_active else 0.5)
            ax.tick_params(colors='#aaaaaa', labelsize=int(6*self.font_scale), length=2, width=0.5)
            ax.set_ylabel('ms', color='#aaaaaa', fontsize=int(7*self.font_scale))

            # Format x-axis with smooth time labels
            # Use AutoDateLocator for smooth scrolling without jumping
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M%p'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))

            # Optional grid lines (global setting) - AFTER axis formatting
            if self.monitor.grid_enabled:
                ax.grid(True, which='both', axis='both', alpha=0.3, color='#aaaaaa',
                       linewidth=0.5, linestyle=':', zorder=1)

            # Add stats text at top right (OUTSIDE graph, as right title)
            stats = self.monitor.get_stats(target.ip)
            cur = f"{stats['current']:.1f}" if stats['current'] is not None else "-"
            avg = f"{stats['avg']:.1f}" if stats['avg'] is not None else "-"
            min_val = f"{stats['min']:.1f}" if stats['min'] is not None else "-"
            max_val = f"{stats['max']:.1f}" if stats['max'] is not None else "-"
            pl = f"{stats['packet_loss']:.1f}"

            stats_text = f"Cur:{cur} Avg:{avg} Min:{min_val} Max:{max_val} PL:{pl}%"
            # Position stats on second line below title
            ax.text(1.0, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=int(7*self.font_scale), color='#ffffff', ha='right', va='top')


            # Redraw
            chart['canvas'].draw()

    def update_display(self):
        """Update all display elements"""
        try:
            # Check if targets have changed (using flag from monitor)
            if self.monitor.targets_changed:
                print(f"[GUI] Detected target change! Rebuilding charts...")
                # Reset the flag
                self.monitor.targets_changed = False
                # Clear existing widgets
                for widget in self.main_frame.winfo_children():
                    widget.destroy()
                # Recreate chart grid with new targets
                self.create_equal_chart_grid()
                print(f"[GUI] Charts rebuilt with {len(self.monitor.targets)} targets")

            self.update_charts()

            # Update IP address label (checks every 10 seconds for changes)
            self.update_access_info_label()
        except Exception as e:
            print(f"Update error: {e}")
            import traceback
            traceback.print_exc()

        # Schedule next update based on ping interval (minimum 500ms for smooth updates)
        update_interval_ms = max(500, int(self.monitor.ping_interval * 1000))
        self.root.after(update_interval_ms, self.update_display)

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


class WebController:
    """Web interface for controlling the ping monitor"""

    def __init__(self, monitor: PingMonitor, config_path: str = "config.json", port: int = 5000):
        self.monitor = monitor
        self.config_path = config_path
        self.port = port
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False

        # Suppress Flask logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.setup_routes()
        self.thread = None

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Serve the main control panel"""
            return render_template_string(WEB_UI_TEMPLATE)

        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get current configuration"""
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return jsonify(config)

        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update configuration"""
            try:
                new_config = request.json

                # Validate configuration
                if 'targets' not in new_config or not isinstance(new_config['targets'], list):
                    return jsonify({'error': 'Invalid targets'}), 400

                # Save to file
                with open(self.config_path, 'w') as f:
                    json.dump(new_config, f, indent=2)

                # Reload configuration in monitor
                self.reload_monitor_config(new_config)

                return jsonify({'status': 'success', 'message': 'Configuration updated'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/targets', methods=['POST'])
        def update_targets():
            """Update target list"""
            try:
                targets = request.json.get('targets', [])

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Update targets
                config['targets'] = targets

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                # Reload
                self.reload_monitor_config(config)

                return jsonify({'status': 'success'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/ping-interval', methods=['POST'])
        def update_ping_interval():
            """Update ping interval"""
            try:
                interval = request.json.get('interval')
                if not interval or interval < 1:
                    return jsonify({'error': 'Invalid interval'}), 400

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                config['ping_interval'] = interval

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                # Reload
                self.reload_monitor_config(config)

                return jsonify({'status': 'success'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/window-duration', methods=['POST'])
        def update_window_duration():
            """Update chart window duration"""
            try:
                duration = request.json.get('duration')
                # Allow all time window values: 1min, 5min, 10min, 30min, 60min, 3h, 6h, 12h, 24h
                if duration not in [1, 5, 10, 30, 60, 180, 360, 720, 1440]:
                    return jsonify({'error': 'Invalid duration. Must be 1, 5, 10, 30, 60, 180, 360, 720, or 1440 minutes'}), 400

                # Update monitor directly (takes effect immediately)
                self.monitor.chart_window_minutes = duration

                return jsonify({'status': 'success'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/interfaces', methods=['GET'])
        def get_interfaces():
            """Get available network interfaces"""
            try:
                interfaces = []
                for iface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            ip = addr.get('addr')
                            if ip and ip != '127.0.0.1':
                                interfaces.append({
                                    'name': iface,
                                    'ip': ip
                                })
                return jsonify(interfaces)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/access-info', methods=['GET'])
        def get_access_info():
            """Get URLs to access the control panel"""
            try:
                urls = []
                port = 5000

                # Get all non-loopback IPv4 addresses
                for iface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            ip = addr.get('addr')
                            if ip and ip != '127.0.0.1':
                                urls.append(f'http://{ip}:{port}')

                # Add localhost as fallback
                if not urls:
                    urls.append(f'http://localhost:{port}')

                return jsonify({'urls': urls})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get current monitor status"""
            return jsonify({
                'running': self.monitor.running,
                'num_targets': len(self.monitor.targets),
                'ping_interval': self.monitor.ping_interval,
                'chart_window_minutes': self.monitor.chart_window_minutes,
                'line_thickness': self.monitor.line_thickness,
                'num_columns': self.monitor.num_columns,
                'targets': [{'name': t.name, 'ip': t.ip, 'interface': t.interface, 'consecutive_loss_threshold': t.consecutive_loss_threshold} for t in self.monitor.targets]
            })

        @self.app.route('/api/line-thickness', methods=['POST'])
        def update_line_thickness():
            """Update line thickness"""
            try:
                thickness = request.json.get('thickness')
                if thickness is None or thickness < 1 or thickness > 5:
                    return jsonify({'error': 'Invalid thickness. Must be between 1 and 5'}), 400

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                config['line_thickness'] = thickness

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                # Update monitor
                self.monitor.line_thickness = thickness

                return jsonify({'status': 'success'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/font-scale', methods=['POST'])
        def update_font_scale():
            """Update font scale (requires restart to take effect)"""
            try:
                scale = request.json.get('scale')
                if scale is None or scale < 0.5 or scale > 4.0:
                    return jsonify({'error': 'Invalid scale. Must be between 0.5 and 4.0'}), 400

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                config['font_scale'] = scale

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                return jsonify({'status': 'success', 'message': 'Font scale saved. Restart required.'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/num-columns', methods=['POST'])
        def update_num_columns():
            """Update number of chart columns (requires restart to take effect)"""
            try:
                num_cols = request.json.get('num_columns')
                if num_cols is None or num_cols < 1 or num_cols > 10:
                    return jsonify({'error': 'Invalid number of columns. Must be between 1 and 10'}), 400

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                config['num_columns'] = num_cols

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                return jsonify({'status': 'success', 'message': 'Chart columns saved. Restart required.'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/grid-enabled', methods=['POST'])
        def update_grid_enabled():
            """Update grid lines visibility (takes effect immediately)"""
            try:
                grid_enabled = request.json.get('grid_enabled')
                if grid_enabled is None:
                    return jsonify({'error': 'Missing grid_enabled parameter'}), 400

                # Load current config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                config['grid_enabled'] = grid_enabled

                # Save
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                # Update monitor in realtime
                self.monitor.grid_enabled = grid_enabled

                return jsonify({'status': 'success', 'message': 'Grid lines ' + ('enabled' if grid_enabled else 'disabled')})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/restart-target', methods=['POST'])
        def restart_target():
            """Restart a single target without affecting others"""
            try:
                target_ip = request.json.get('ip')
                if not target_ip:
                    return jsonify({'error': 'Missing target IP'}), 400

                # Find the target
                target = None
                for t in self.monitor.targets:
                    if t.ip == target_ip:
                        target = t
                        break

                if not target:
                    return jsonify({'error': 'Target not found'}), 404

                with self.monitor.lock:
                    # Reset consecutive loss counter
                    if target_ip in self.monitor.consecutive_losses:
                        self.monitor.consecutive_losses[target_ip] = 0

                    # Clear last successful RTT
                    if target_ip in self.monitor.last_successful_rtt:
                        self.monitor.last_successful_rtt[target_ip] = None

                    # Clear history for this target
                    if target_ip in self.monitor.history:
                        self.monitor.history[target_ip].clear()

                return jsonify({'status': 'success', 'message': f'Target {target.name} restarted'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/force-reload', methods=['POST'])
        def force_reload():
            """Force reload configuration and restart logging"""
            try:
                # Reload config from file
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

                # Reload monitor configuration
                self.reload_monitor_config(config)

                return jsonify({'status': 'success', 'message': 'Configuration reloaded and logging restarted'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # Dashboard and chart routes removed

        @self.app.route('/api/targets')
        def get_targets():
            """Get list of all targets for reference"""
            targets_list = []
            for t in self.monitor.targets:
                targets_list.append({
                    'name': t.name,
                    'ip': t.ip
                })
            return jsonify(targets_list)

        @self.app.route('/api/dummy')
        def dummy():
            """Placeholder route"""
            return jsonify({'status': 'ok'})

        # Keeping for compatibility, returns error
        @self.app.route('/api/chart-data/<target_ip>')
        def get_chart_data(target_ip):
            """Removed - dashboard deprecated"""
            return jsonify({'error': 'Dashboard removed'}), 404

    def reload_monitor_config(self, new_config):
        """Reload monitor configuration without restarting"""
        with self.monitor.lock:
            # Update targets
            new_targets = [Target(**t) for t in new_config['targets']]

            # Sort alphabetically
            new_targets.sort(key=lambda t: t.name.lower())

            # Preserve history for existing targets, create new deques for new ones
            new_history = {}
            for target in new_targets:
                if target.ip in self.monitor.history:
                    new_history[target.ip] = self.monitor.history[target.ip]
                else:
                    new_history[target.ip] = deque()

            self.monitor.targets = new_targets
            self.monitor.history = new_history
            self.monitor.ping_interval = new_config.get('ping_interval', self.monitor.ping_interval)

            # Signal GUI that targets have changed
            self.monitor.targets_changed = True
            print(f"[API] Targets updated! New count: {len(new_targets)}, Flag set: {self.monitor.targets_changed}")

            # Restart ping threads
            was_running = self.monitor.running

            # Stop old threads if they exist
            if self.monitor.running and self.monitor.threads:
                self.monitor.running = False
                for thread in self.monitor.threads:
                    if thread.is_alive():
                        thread.join(timeout=2.0)

            # Start new threads (always start if monitor was ever running)
            if was_running:
                self.monitor.running = True
                self.monitor.threads = []
                for target in self.monitor.targets:
                    thread = threading.Thread(target=self.monitor._ping_worker, args=(target,), daemon=True)
                    thread.start()
                    self.monitor.threads.append(thread)
                print(f"[API] Restarted {len(self.monitor.threads)} ping threads")

    def start(self):
        """Start the web server in a background thread"""
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        print(f"Web interface started at http://localhost:{self.port}")


# HTML Template for Web UI
WEB_UI_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Ping Monitor Control Panel</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #4CAF50; margin-bottom: 30px; font-size: 32px; }
        h2 { color: #2196F3; margin: 0 0 10px; border-bottom: 2px solid #333; padding-bottom: 8px; font-size: 16px; }
        .section {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .status {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .status-item {
            background: #333;
            padding: 6px 12px;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .status-item label {
            color: #aaa;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .status-item .value {
            font-size: 13px;
            font-weight: bold;
            color: #4CAF50;
        }
        .form-group { margin-bottom: 12px; }
        label {
            display: block;
            margin-bottom: 4px;
            color: #bbb;
            font-weight: 600;
            font-size: 12px;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 8px;
            background: #333;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 13px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #4CAF50;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.3s;
            margin-right: 8px;
        }
        button:hover { background: #45a049; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
        button:active { transform: translateY(0px); }
        .btn-secondary { background: #2196F3; }
        .btn-secondary:hover { background: #1976D2; }
        .btn-danger { background: #f44336; }
        .btn-danger:hover { background: #d32f2f; }
        .btn-small { padding: 8px 16px; font-size: 12px; margin: 0; }
        .message {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            display: none;
            font-weight: 500;
        }
        .message.success { background: #4CAF50; color: white; display: block; }
        .message.error { background: #f44336; color: white; display: block; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th {
            background: #333;
            padding: 6px 8px;
            text-align: left;
            color: #4CAF50;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }
        td {
            padding: 3px 8px;
            border-bottom: 1px solid #404040;
            font-size: 12px;
            line-height: 1.1;
        }
        tr:hover { background: #333; }
        .add-target-form {
            display: grid;
            grid-template-columns: 2fr 2fr auto;
            gap: 15px;
            align-items: end;
            margin-top: 20px;
        }
        .interface-list {
            display: grid;
            gap: 12px;
            margin-top: 15px;
        }
        .interface-item {
            background: #333;
            padding: 15px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .interface-name {
            font-weight: 600;
            color: #4CAF50;
            font-size: 16px;
        }
        .interface-ip {
            color: #888;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            margin-top: 4px;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #444;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
        }
        input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            background: #4CAF50;
            cursor: pointer;
            border-radius: 50%;
            border: none;
        }
        @media (max-width: 768px) {
            .settings-grid, .add-target-form {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 Ping Monitor Control Panel</h1>

        <div class="section">
            <div class="status">
                <div class="status-item">
                    <label>Status</label>
                    <div class="value" id="status-running">-</div>
                </div>
                <div class="status-item">
                    <label>Monitored Targets</label>
                    <div class="value" id="status-targets">-</div>
                </div>
                <div class="status-item">
                    <label>Ping Interval</label>
                    <div class="value" id="status-interval">-</div>
                </div>
                <div class="status-item">
                    <label>Time Window</label>
                    <div class="value" id="status-window">-</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Manage Targets</h2>
            <div id="targets-message" class="message"></div>

            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>IP Address</th>
                        <th>Interface</th>
                        <th>Loss Threshold</th>
                        <th style="width: 90px; text-align: center;">Show Avg</th>
                        <th style="width: 120px; text-align: center;">Actions</th>
                    </tr>
                </thead>
                <tbody id="targets-table">
                    <tr><td colspan="6" style="text-align: center; color: #888;">Loading...</td></tr>
                </tbody>
            </table>

            <div class="add-target-form" style="grid-template-columns: 2fr 2fr 1.5fr 1fr auto;">
                <div class="form-group" style="margin: 0;">
                    <label>Target Name</label>
                    <input type="text" id="new-target-name" placeholder="e.g., Google DNS">
                </div>
                <div class="form-group" style="margin: 0;">
                    <label>IP Address</label>
                    <input type="text" id="new-target-ip" placeholder="e.g., 8.8.8.8">
                </div>
                <div class="form-group" style="margin: 0;">
                    <label>Interface (optional)</label>
                    <select id="new-target-interface">
                        <option value="">Default</option>
                    </select>
                </div>
                <div class="form-group" style="margin: 0;">
                    <label>Loss Threshold</label>
                    <input type="number" id="new-target-threshold" min="1" max="10" value="1" placeholder="1" title="Number of consecutive ping losses before marking as down (1 = plot all losses)">
                </div>
                <div>
                    <label style="opacity: 0;">Add</label>
                    <button onclick="addTarget()">➕ Add Target</button>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Settings</h2>
            <div id="settings-message" class="message"></div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Ping Interval (sec)</label>
                    <input type="number" id="ping-interval" min="1" max="60" value="5" style="width: 60px; padding: 6px;">
                    <button onclick="updatePingInterval()" style="padding: 6px 10px;">💾</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Time Window</label>
                    <select id="window-duration" style="padding: 6px; width: 110px;">
                        <option value="1">1 min</option>
                        <option value="5">5 min</option>
                        <option value="10">10 min</option>
                        <option value="30" selected>30 min</option>
                        <option value="60">60 min</option>
                        <option value="180">3 hours</option>
                        <option value="360">6 hours</option>
                        <option value="720">12 hours</option>
                        <option value="1440">24 hours</option>
                    </select>
                    <button onclick="updateWindowDuration()" style="padding: 6px 10px;">💾</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Line Thickness</label>
                    <input type="number" id="line-thickness" min="1" max="5" value="2" step="1" style="width: 60px; padding: 6px;">
                    <button onclick="updateLineThickness()" style="padding: 6px 10px;">💾</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Font Scale</label>
                    <input type="number" id="font-scale" min="0.5" max="4.0" value="2.0" step="0.1" style="width: 60px; padding: 6px;">
                    <button onclick="updateFontScale()" style="padding: 6px 10px;">💾 & Restart</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Chart Columns</label>
                    <input type="number" id="num-columns" min="1" max="10" value="2" step="1" style="width: 60px; padding: 6px;">
                    <button onclick="updateNumColumns()" style="padding: 6px 10px;">💾 & Restart</button>
                </div>

                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="min-width: 140px;">Show Grid Lines</label>
                    <input type="checkbox" id="grid-enabled" checked style="width: 20px; height: 20px; cursor: pointer;">
                    <button onclick="updateGridEnabled()" style="padding: 6px 10px;">💾</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTargets = [];
        let availableInterfaces = [];
        let editingIndex = null; // Track which target is being edited

        function showMessage(elementId, message, type) {
            const el = document.getElementById(elementId);
            el.textContent = message;
            el.className = 'message ' + type;
            setTimeout(() => {
                el.className = 'message';
            }, 5000);
        }

        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                document.getElementById('status-running').textContent = data.running ? '✅ Running' : '❌ Stopped';
                document.getElementById('status-targets').textContent = data.num_targets;
                document.getElementById('status-interval').textContent = data.ping_interval + 's';

                // Format window duration
                const minutes = data.chart_window_minutes;
                let windowText;
                if (minutes >= 60) {
                    const hours = minutes / 60;
                    windowText = hours + (hours === 1 ? ' hour' : ' hours');
                } else {
                    windowText = minutes + ' min';
                }
                document.getElementById('status-window').textContent = windowText;

                document.getElementById('ping-interval').value = data.ping_interval;
                document.getElementById('window-duration').value = data.chart_window_minutes;

                // Update line thickness
                if (data.line_thickness !== undefined) {
                    document.getElementById('line-thickness').value = data.line_thickness;
                }

                // Update num_columns
                if (data.num_columns !== undefined) {
                    document.getElementById('num-columns').value = data.num_columns;
                }

                // Load font scale from config
                const configResponse = await fetch('/api/config');
                const config = await configResponse.json();
                if (config.font_scale !== undefined) {
                    document.getElementById('font-scale').value = config.font_scale;
                }

                // Update grid_enabled
                if (config.grid_enabled !== undefined) {
                    document.getElementById('grid-enabled').checked = config.grid_enabled;
                }

                currentTargets = data.targets || [];
                renderTargetsTable();
            } catch (error) {
                console.error('Error loading status:', error);
                showMessage('targets-message', 'Error loading status', 'error');
            }
        }

        async function loadInterfaces() {
            try {
                const response = await fetch('/api/interfaces');
                availableInterfaces = await response.json();

                // Update the interface dropdown in add-target form
                const select = document.getElementById('new-target-interface');
                select.innerHTML = '<option value="">Default</option>' +
                    availableInterfaces.map(iface =>
                        `<option value="${iface.name}">${iface.name} (${iface.ip})</option>`
                    ).join('');
            } catch (error) {
                console.error('Error loading interfaces:', error);
            }
        }

        function renderTargetsTable() {
            const tbody = document.getElementById('targets-table');
            if (currentTargets.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No targets configured</td></tr>';
                return;
            }

            tbody.innerHTML = currentTargets.map((target, index) => {
                if (editingIndex === index) {
                    // Inline edit mode
                    return `
                        <tr style="background: #2a2a2a;">
                            <td><input type="text" id="edit-name-${index}" value="${target.name}" style="width: 100%; padding: 4px;"></td>
                            <td><input type="text" id="edit-ip-${index}" value="${target.ip}" style="width: 100%; padding: 4px; font-family: monospace;"></td>
                            <td>
                                <select id="edit-interface-${index}" style="width: 100%; padding: 4px;">
                                    <option value="">Default</option>
                                    ${availableInterfaces.map(iface =>
                                        `<option value="${iface.name}" ${target.interface === iface.name ? 'selected' : ''}>${iface.name} (${iface.ip})</option>`
                                    ).join('')}
                                </select>
                            </td>
                            <td><input type="number" id="edit-threshold-${index}" value="${target.consecutive_loss_threshold || 1}" min="1" max="10" style="width: 60px; padding: 4px;"></td>
                            <td style="text-align: center; white-space: nowrap;">
                                <button onclick="saveEdit(${index})" class="btn-primary btn-small" style="margin-right: 3px;">💾</button>
                                <button onclick="cancelEdit()" class="btn-secondary btn-small">✖️</button>
                            </td>
                        </tr>
                    `;
                } else {
                    // Normal view mode
                    return `
                        <tr id="target-row-${index}">
                            <td>${target.name}</td>
                            <td style="font-family: monospace;">${target.ip}</td>
                            <td>${target.interface || 'Default'}</td>
                            <td>
                                <input type="number" min="1" max="10" value="${target.consecutive_loss_threshold || 1}"
                                       onchange="updateThreshold(${index}, this.value)"
                                       style="width: 60px; padding: 4px; text-align: center;">
                            </td>
                            <td style="text-align: center;">
                                <input type="checkbox" ${target.show_average !== false ? 'checked' : ''}
                                       onchange="updateShowAverage(${index}, this.checked)"
                                       style="width: 20px; height: 20px; cursor: pointer;">
                            </td>
                            <td style="text-align: center; white-space: nowrap;">
                                <button onclick="editTarget(${index})" class="btn-primary btn-small" style="margin-right: 3px;" title="Edit">✏️</button>
                                <button onclick="restartTarget(${index})" class="btn-secondary btn-small" style="margin-right: 3px;" title="Restart">🔄</button>
                                <button onclick="deleteTarget(${index})" class="btn-danger btn-small" title="Delete">🗑️</button>
                            </td>
                        </tr>
                    `;
                }
            }).join('');
        }

        async function addTarget() {
            const name = document.getElementById('new-target-name').value.trim();
            const ip = document.getElementById('new-target-ip').value.trim();
            const iface = document.getElementById('new-target-interface').value;
            const threshold = parseInt(document.getElementById('new-target-threshold').value) || 1;

            if (!name || !ip) {
                showMessage('targets-message', 'Please enter both name and IP address', 'error');
                return;
            }

            const targetData = {name, ip, consecutive_loss_threshold: threshold};
            if (iface) {
                targetData.interface = iface;
            }

            // Add new target
            currentTargets.push(targetData);

            // Update via API
            try {
                const response = await fetch('/api/targets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({targets: currentTargets})
                });

                if (response.ok) {
                    showMessage('targets-message', `Target "${name}" added successfully!`, 'success');
                    // Clear form
                    document.getElementById('new-target-name').value = '';
                    document.getElementById('new-target-ip').value = '';
                    document.getElementById('new-target-interface').value = '';
                    document.getElementById('new-target-threshold').value = '1';
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                    currentTargets.pop();
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
                currentTargets.pop();
            }
        }

        function editTarget(index) {
            editingIndex = index;
            renderTargetsTable();
        }

        async function saveEdit(index) {
            const name = document.getElementById(`edit-name-${index}`).value.trim();
            const ip = document.getElementById(`edit-ip-${index}`).value.trim();
            const iface = document.getElementById(`edit-interface-${index}`).value;
            const threshold = parseInt(document.getElementById(`edit-threshold-${index}`).value) || 1;

            if (!name || !ip) {
                showMessage('targets-message', 'Please enter both name and IP address', 'error');
                return;
            }

            const targetData = {name, ip, consecutive_loss_threshold: threshold};
            if (iface) {
                targetData.interface = iface;
            }

            // Update target
            currentTargets[index] = targetData;

            // Update via API
            try {
                const response = await fetch('/api/targets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({targets: currentTargets})
                });

                if (response.ok) {
                    showMessage('targets-message', `Target "${name}" updated successfully!`, 'success');
                    cancelEdit();
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
            }
        }

        function cancelEdit() {
            editingIndex = null;
            renderTargetsTable();
        }

        async function restartTarget(index) {
            const target = currentTargets[index];

            if (!confirm(`Restart target "${target.name}"? This will clear its ping history.`)) {
                return;
            }

            try {
                const response = await fetch('/api/restart-target', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ip: target.ip})
                });

                if (response.ok) {
                    const result = await response.json();
                    showMessage('targets-message', result.message || 'Target restarted successfully!', 'success');
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateThreshold(index, newValue) {
            const threshold = parseInt(newValue) || 1;
            if (threshold < 1 || threshold > 10) {
                showMessage('targets-message', 'Loss threshold must be between 1 and 10', 'error');
                renderTargetsTable(); // Reset to original value
                return;
            }

            // Update in local array
            currentTargets[index].consecutive_loss_threshold = threshold;

            // Save to backend
            try {
                const response = await fetch('/api/targets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({targets: currentTargets})
                });

                if (response.ok) {
                    showMessage('targets-message', `Loss threshold updated to ${threshold}`, 'success');
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateShowAverage(index, checked) {
            // Update in local array
            currentTargets[index].show_average = checked;

            // Save to backend
            try {
                const response = await fetch('/api/targets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({targets: currentTargets})
                });

                if (response.ok) {
                    showMessage('targets-message', `Average line ${checked ? 'enabled' : 'disabled'}`, 'success');
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
            }
        }

        async function deleteTarget(index) {
            const target = currentTargets[index];
            if (!confirm(`Delete target "${target.name}" (${target.ip})?`)) {
                return;
            }

            currentTargets.splice(index, 1);

            try {
                const response = await fetch('/api/targets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({targets: currentTargets})
                });

                if (response.ok) {
                    showMessage('targets-message', `Target "${target.name}" deleted successfully!`, 'success');
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('targets-message', 'Error: ' + error.error, 'error');
                    loadStatus(); // Reload to restore
                }
            } catch (error) {
                showMessage('targets-message', 'Error: ' + error.message, 'error');
                loadStatus(); // Reload to restore
            }
        }

        async function updatePingInterval() {
            try {
                const interval = parseInt(document.getElementById('ping-interval').value);

                if (interval < 1 || interval > 60) {
                    showMessage('settings-message', 'Interval must be between 1 and 60 seconds', 'error');
                    return;
                }

                const response = await fetch('/api/ping-interval', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({interval})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Ping interval updated to ' + interval + ' seconds!', 'success');
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateWindowDuration() {
            try {
                const duration = parseInt(document.getElementById('window-duration').value);

                const response = await fetch('/api/window-duration', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({duration})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Window duration updated to ' + duration + ' minutes!', 'success');
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateLineThickness() {
            try {
                const thickness = parseInt(document.getElementById('line-thickness').value);

                const response = await fetch('/api/line-thickness', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({thickness})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Line thickness updated to ' + thickness + '! Reloading...', 'success');

                    // Force reload to apply changes
                    setTimeout(async () => {
                        await fetch('/api/force-reload', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'}
                        });
                        loadStatus();
                    }, 500);
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateFontScale() {
            try {
                const scale = parseFloat(document.getElementById('font-scale').value);

                const response = await fetch('/api/font-scale', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scale})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Font scale updated to ' + scale.toFixed(1) + '! Reloading...', 'success');

                    // Force reload to apply changes
                    setTimeout(async () => {
                        await fetch('/api/force-reload', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'}
                        });
                        loadStatus();
                    }, 500);
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateNumColumns() {
            try {
                const num_columns = parseInt(document.getElementById('num-columns').value);

                const response = await fetch('/api/num-columns', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({num_columns})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Chart columns updated to ' + num_columns + '! Reloading...', 'success');

                    // Force reload to apply changes
                    setTimeout(async () => {
                        await fetch('/api/force-reload', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'}
                        });
                        loadStatus();
                    }, 500);
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function updateGridEnabled() {
            try {
                const grid_enabled = document.getElementById('grid-enabled').checked;

                const response = await fetch('/api/grid-enabled', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({grid_enabled})
                });

                if (response.ok) {
                    showMessage('settings-message', 'Grid lines ' + (grid_enabled ? 'enabled' : 'disabled') + '!', 'success');
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        async function forceReload() {
            try {
                if (!confirm('Force reload will restart all ping logging. Continue?')) {
                    return;
                }

                const response = await fetch('/api/force-reload', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });

                if (response.ok) {
                    const result = await response.json();
                    showMessage('settings-message', result.message || 'Configuration reloaded successfully!', 'success');
                    loadStatus();
                } else {
                    const error = await response.json();
                    showMessage('settings-message', 'Error: ' + error.error, 'error');
                }
            } catch (error) {
                showMessage('settings-message', 'Error: ' + error.message, 'error');
            }
        }

        // Setup event listeners
        document.addEventListener('DOMContentLoaded', () => {
            // Allow Enter key to add target
            document.getElementById('new-target-ip').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    addTarget();
                }
            });
        });

        // Load initial data
        window.addEventListener('load', () => {
            loadStatus();
            loadInterfaces();

            // Auto-refresh status every 10 seconds
            setInterval(loadStatus, 10000);
        });
    </script>
</body>
</html>

'''


# Dashboard Template for Live Visualization
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Ping Monitor Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #2d2d2d;
            color: #e0e0e0;
            overflow: hidden;
        }
        #dashboard {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px;
            padding: 4px;
            height: 100vh;
        }
        .chart-container {
            background: #3a3a3a;
            border-radius: 4px;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
            padding: 0 5px;
        }
        .chart-title {
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
        }
        .chart-stats {
            color: #ffffff;
            font-size: 11px;
            text-align: right;
        }
        .chart-wrapper {
            flex: 1;
            position: relative;
            min-height: 0;
        }
        canvas {
            width: 100% !important;
            height: 100% !important;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
        }
        .nav-bar {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }
        .nav-bar a {
            background: #4CAF50;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .nav-bar a:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="nav-bar">
        <a href="/">⚙️ Control Panel</a>
    </div>
    <div id="dashboard"></div>

    <script>
        const charts = {};
        let targets = [];

        // Initialize dashboard
        async function initDashboard() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                targets = status.targets;

                // Adjust grid based on number of columns from config
                const configResponse = await fetch('/api/config');
                const config = await configResponse.json();
                const numColumns = config.num_columns || 2;
                document.getElementById('dashboard').style.gridTemplateColumns = `repeat(${numColumns}, 1fr)`;

                // Create charts for each target
                const dashboard = document.getElementById('dashboard');
                dashboard.innerHTML = '';

                console.log(`Creating dashboard for ${targets.length} targets...`);

                for (let i = 0; i < targets.length; i++) {
                    const target = targets[i];
                    const safeId = `target-${i}`;

                    console.log(`Creating chart ${i} for ${target.name} (${target.ip}) with ID ${safeId}`);

                    const container = document.createElement('div');
                    container.className = 'chart-container';
                    container.innerHTML = `
                        <div class="chart-header">
                            <div class="chart-title">${target.name} (${target.ip})</div>
                            <div class="chart-stats" id="stats-${safeId}">Loading...</div>
                        </div>
                        <div class="chart-wrapper">
                            <canvas id="chart-${safeId}"></canvas>
                        </div>
                    `;
                    dashboard.appendChild(container);

                    // Create chart with error handling
                    try {
                        createChart(target, safeId);
                        console.log(`Chart ${i} created successfully`);
                    } catch (error) {
                        console.error(`Error creating chart ${i}:`, error);
                    }
                }

                console.log(`Total charts created: ${Object.keys(charts).length}`);

                // Force chart resize after layout settles
                setTimeout(() => {
                    console.log('Resizing all charts...');
                    Object.values(charts).forEach(chartObj => {
                        if (chartObj && chartObj.chart) {
                            chartObj.chart.resize();
                        }
                    });
                }, 100);

                // Start auto-refresh
                setInterval(updateAllCharts, 1000);
            } catch (error) {
                console.error('Error initializing dashboard:', error);
            }
        }

        function createChart(target, safeId) {
            const canvasId = `chart-${safeId}`;
            const ctx = document.getElementById(canvasId).getContext('2d');

            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'RTT',
                            data: [],
                            borderColor: '#ffffff',
                            backgroundColor: 'transparent',
                            borderWidth: 3,
                            pointRadius: 0,
                            tension: 0.1,
                            spanGaps: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: true }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute',
                                displayFormats: { minute: 'HH:mm' }
                            },
                            grid: { color: '#666666', lineWidth: 0.3 },
                            ticks: { color: '#aaaaaa', font: { size: 10 } }
                        },
                        y: {
                            beginAtZero: true,
                            grid: { color: '#666666', lineWidth: 0.3 },
                            ticks: { color: '#aaaaaa', font: { size: 10 } }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                },
                plugins: [{
                    id: 'colorZones',
                    beforeDraw: (chart) => {
                        const ctx = chart.ctx;
                        const chartArea = chart.chartArea;
                        const yScale = chart.scales.y;

                        // Get max RTT to determine which zones to show
                        const data = chart.data.datasets[0].data;
                        const maxRTT = data.reduce((max, point) => {
                            return point.y !== null && point.y > max ? point.y : max;
                        }, 0);

                        const yMax = yScale.max;

                        ctx.save();

                        // Green zone (0-100ms)
                        if (yMax >= 100) {
                            ctx.fillStyle = 'rgba(0, 170, 0, 0.25)';
                            ctx.fillRect(
                                chartArea.left,
                                yScale.getPixelForValue(100),
                                chartArea.right - chartArea.left,
                                chartArea.bottom - yScale.getPixelForValue(100)
                            );
                        } else {
                            ctx.fillStyle = 'rgba(0, 170, 0, 0.25)';
                            ctx.fillRect(
                                chartArea.left,
                                chartArea.top,
                                chartArea.right - chartArea.left,
                                chartArea.bottom - chartArea.top
                            );
                        }

                        // Yellow zone (100-200ms)
                        if ((maxRTT > 100 || hasPacketLoss(data)) && yMax >= 100) {
                            const yellowTop = Math.min(200, yMax);
                            ctx.fillStyle = 'rgba(255, 204, 0, 0.3)';
                            ctx.fillRect(
                                chartArea.left,
                                yScale.getPixelForValue(yellowTop),
                                chartArea.right - chartArea.left,
                                yScale.getPixelForValue(100) - yScale.getPixelForValue(yellowTop)
                            );
                        }

                        // Red zone (200ms+)
                        if ((maxRTT > 200 || hasPacketLoss(data)) && yMax >= 200) {
                            ctx.fillStyle = 'rgba(255, 51, 51, 0.3)';
                            ctx.fillRect(
                                chartArea.left,
                                chartArea.top,
                                chartArea.right - chartArea.left,
                                yScale.getPixelForValue(200) - chartArea.top
                            );
                        }

                        ctx.restore();
                    }
                }]
            });

            charts[target.ip] = { chart: chart, safeId: safeId };
        }

        function hasPacketLoss(data) {
            return data.some(point => point.y === null);
        }

        async function updateAllCharts() {
            for (let i = 0; i < targets.length; i++) {
                const target = targets[i];
                const safeId = `target-${i}`;
                await updateChart(target, safeId);
            }
        }

        async function updateChart(target, safeId) {
            try {
                const response = await fetch(`/api/chart-data/${encodeURIComponent(target.ip)}`);
                const data = await response.json();

                if (data.error) {
                    console.error(`Error fetching data for ${target.ip}:`, data.error);
                    return;
                }

                const chartObj = charts[target.ip];
                if (!chartObj) return;

                const chart = chartObj.chart;

                // Update chart data
                const chartData = data.data.map(point => ({
                    x: new Date(point.timestamp),
                    y: point.success ? point.rtt_ms : null
                }));

                chart.data.datasets[0].data = chartData;

                // Calculate dynamic Y-axis max
                const validRTTs = chartData.filter(p => p.y !== null).map(p => p.y);
                const hasLoss = chartData.some(p => p.y === null);

                let yMax = 50;
                if (validRTTs.length > 0) {
                    const maxRTT = Math.max(...validRTTs);
                    yMax = maxRTT * 1.25;
                    if (hasLoss) {
                        yMax = Math.max(yMax, 250);
                    }
                } else if (hasLoss) {
                    yMax = 250;
                }

                chart.options.scales.y.max = yMax;

                chart.update('none');

                // Update stats
                const stats = data.stats;
                const statsEl = document.getElementById(`stats-${safeId}`);
                if (statsEl) {
                    const cur = stats.current !== null ? stats.current.toFixed(1) : '-';
                    const avg = stats.avg !== null ? stats.avg.toFixed(1) : '-';
                    const min = stats.min !== null ? stats.min.toFixed(1) : '-';
                    const max = stats.max !== null ? stats.max.toFixed(1) : '-';
                    const pl = stats.packet_loss.toFixed(1);
                    statsEl.textContent = `Cur:${cur} Avg:${avg} Min:${min} Max:${max} PL:${pl}%`;
                }
            } catch (error) {
                console.error(`Error updating chart for ${target.ip}:`, error);
            }
        }

        // Initialize on page load
        window.addEventListener('load', initDashboard);

        // Handle window resize
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                console.log('Window resized, updating charts...');
                Object.values(charts).forEach(chartObj => {
                    if (chartObj && chartObj.chart) {
                        chartObj.chart.resize();
                    }
                });
            }, 250);
        });

        // Reinitialize if targets change
        setInterval(async () => {
            const response = await fetch('/api/status');
            const status = await response.json();
            if (JSON.stringify(status.targets) !== JSON.stringify(targets)) {
                console.log('Targets changed, reinitializing dashboard...');
                initDashboard();
            }
        }, 5000);
    </script>
</body>
</html>
'''


# Screen Capture Template
SCREEN_CAPTURE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Ping Monitor - Screen Capture</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }
        #screen-container {
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
        }
        #screen-stream {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
        }
        .nav-bar {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            display: flex;
            gap: 10px;
        }
        .nav-bar a {
            background: #4CAF50;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .nav-bar a:hover {
            background: #45a049;
        }
        .status {
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: #4CAF50;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="nav-bar">
        <a href="/dashboard">📊 Chart Dashboard</a>
        <a href="/">⚙️ Control Panel</a>
    </div>
    <div id="screen-container">
        <img id="screen-stream" src="/screen-capture.jpg" alt="Display Capture">
    </div>
    <div class="status">
        🔴 LIVE • Screen Capture (3s refresh)
    </div>
    <script>
        // Refresh the screenshot every 3 seconds
        setInterval(() => {
            const img = document.getElementById('screen-stream');
            // Add timestamp to force refresh
            img.src = '/screen-capture.jpg?' + new Date().getTime();
        }, 3000);
    </script>
</body>
</html>
'''


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Multi-Target Ping Monitor')
    parser.add_argument('--font-scale', type=float, default=2.0,
                       help='Font size scale factor (default: 2.0, larger values for distant viewing)')
    parser.add_argument('--window', type=int, default=30, choices=[1, 5, 10, 30, 60],
                       help='Time window in minutes (default: 30, choices: 1, 5, 10, 30, 60)')
    args = parser.parse_args()

    try:
        print("Loading configuration...")
        monitor = PingMonitor("config.json")

        # Override chart window with command-line argument
        monitor.chart_window_minutes = args.window

        # Sort targets alphabetically by name
        monitor.targets.sort(key=lambda t: t.name.lower())

        print(f"Starting ping monitor for {len(monitor.targets)} targets...")
        print(f"Font scale: {args.font_scale}")
        print(f"Time window: {args.window} minutes")
        monitor.start()

        # Start web interface
        web = WebController(monitor, config_path="config.json", port=5000)
        web.start()

        # Wait for initial data
        print("Collecting initial ping data...")
        time.sleep(monitor.ping_interval + 2)

        print("Launching GUI...")
        gui = PingMonitorGUI(monitor, font_scale=args.font_scale)
        gui.run()

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
