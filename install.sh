#!/bin/bash
#
# Sasha Multi-Ping Monitor - Installation Script
# Version 1.04
#
# This script installs and configures the ping monitoring application
# with systemd service for auto-start on boot.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_USER="${SUDO_USER:-$USER}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sasha Multi-Ping Monitor - Installer${NC}"
echo -e "${BLUE}Version 1.04${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if script is run with appropriate permissions
if [ "$EUID" -eq 0 ] && [ -z "$SUDO_USER" ]; then
    print_error "Please do not run as root directly. Use: sudo ./install.sh"
    exit 1
fi

print_info "Installing as user: $APP_USER"
print_info "Installation directory: $SCRIPT_DIR"
echo ""

# Step 1: Install system dependencies
print_info "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y python3 python3-pip python3-tk curl net-tools > /dev/null 2>&1
print_status "System dependencies installed"
echo ""

# Step 2: Install Python dependencies
print_info "Step 2: Installing Python dependencies..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip3 install -r "$SCRIPT_DIR/requirements.txt" --quiet
    print_status "Python dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing core dependencies..."
    pip3 install matplotlib ping3 flask netifaces --quiet
    print_status "Core Python dependencies installed"
fi
echo ""

# Step 3: Verify ping capabilities
print_info "Step 3: Verifying ping command capabilities..."
if ! getcap /bin/ping | grep -q "cap_net_raw"; then
    print_warning "Setting ping capabilities..."
    setcap cap_net_raw=ep /bin/ping
fi
print_status "Ping capabilities verified"
echo ""

# Step 4: Set up systemd service
print_info "Step 4: Setting up systemd service..."

# Create systemd user directory
sudo -u "$APP_USER" mkdir -p /home/$APP_USER/.config/systemd/user

# Create service file
cat > /home/$APP_USER/.config/systemd/user/sasha-multiping.service << EOF
[Unit]
Description=Sasha Multi-Ping Monitor
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/$APP_USER/.Xauthority"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/gui_monitor.py --window 30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

chown $APP_USER:$APP_USER /home/$APP_USER/.config/systemd/user/sasha-multiping.service
print_status "Systemd service file created"

# Enable lingering for the user (allows service to start at boot)
loginctl enable-linger $APP_USER
print_status "User lingering enabled"

# Reload systemd and enable service
sudo -u "$APP_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $APP_USER) systemctl --user daemon-reload
sudo -u "$APP_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $APP_USER) systemctl --user enable sasha-multiping.service
print_status "Service enabled for auto-start"
echo ""

# Step 5: Set file permissions
print_info "Step 5: Setting file permissions..."
chmod +x "$SCRIPT_DIR/gui_monitor.py"
chown -R $APP_USER:$APP_USER "$SCRIPT_DIR"
print_status "File permissions set"
echo ""

# Step 6: Create default config if it doesn't exist
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    print_info "Step 6: Creating default configuration..."
    cat > "$SCRIPT_DIR/config.json" << 'EOF'
{
  "ping_interval": 5,
  "chart_time_window_minutes": 30,
  "history_retention_hours": 48,
  "timeout_seconds": 2,
  "line_thickness": 1,
  "num_columns": 4,
  "grid_enabled": true,
  "targets": [
    {
      "consecutive_loss_threshold": 3,
      "interface": null,
      "ip": "8.8.8.8",
      "name": "Google DNS",
      "show_average": true
    },
    {
      "consecutive_loss_threshold": 3,
      "interface": null,
      "ip": "1.1.1.1",
      "name": "Cloudflare DNS",
      "show_average": true
    }
  ]
}
EOF
    chown $APP_USER:$APP_USER "$SCRIPT_DIR/config.json"
    print_status "Default configuration created with 2 sample targets"
else
    print_info "Step 6: Using existing configuration"
    print_status "Configuration file already exists"
fi
echo ""

# Step 7: Start the service
print_info "Step 7: Starting the service..."
sudo -u "$APP_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $APP_USER) systemctl --user start sasha-multiping.service
sleep 3

# Verify service is running
if sudo -u "$APP_USER" XDG_RUNTIME_DIR=/run/user/$(id -u $APP_USER) systemctl --user is-active --quiet sasha-multiping.service; then
    print_status "Service started successfully"
else
    print_error "Service failed to start. Check status with: systemctl --user status sasha-multiping.service"
    exit 1
fi
echo ""

# Step 8: Display service status
print_info "Verifying installation..."
PING_COUNT=$(ps -ef | grep "[p]ing -i" | wc -l)
GUI_RUNNING=$(ps -ef | grep "[g]ui_monitor.py" | wc -l)

if [ "$GUI_RUNNING" -ge 1 ] && [ "$PING_COUNT" -ge 1 ]; then
    print_status "GUI process running: Yes"
    print_status "Ping processes running: $PING_COUNT"
else
    print_warning "Service may not be fully initialized yet. Wait a few seconds and check status."
fi
echo ""

# Get network IP for web interface
NETWORK_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Application Status:${NC}"
echo -e "  • Service: ${GREEN}Enabled & Running${NC}"
echo -e "  • Auto-start: ${GREEN}Enabled on boot${NC}"
echo -e "  • Working Directory: $SCRIPT_DIR"
echo ""
echo -e "${BLUE}Access Information:${NC}"
echo -e "  • Web Portal: ${GREEN}http://$NETWORK_IP:5000${NC}"
echo -e "  • GUI Display: Should appear on your screen"
echo ""
echo -e "${BLUE}Service Management:${NC}"
echo -e "  • Start:   ${YELLOW}systemctl --user start sasha-multiping.service${NC}"
echo -e "  • Stop:    ${YELLOW}systemctl --user stop sasha-multiping.service${NC}"
echo -e "  • Restart: ${YELLOW}systemctl --user restart sasha-multiping.service${NC}"
echo -e "  • Status:  ${YELLOW}systemctl --user status sasha-multiping.service${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  • Edit targets via web portal: ${GREEN}http://$NETWORK_IP:5000${NC}"
echo -e "  • Config file: $SCRIPT_DIR/config.json"
echo ""
echo -e "${BLUE}Features:${NC}"
echo -e "  • Grid lines: Toggle in Settings → Show Grid Lines"
echo -e "  • Moving average: Toggle per target in Manage Targets"
echo -e "  • Time windows: 1min - 24 hours"
echo -e "  • Continuous ping monitoring with auto-restart"
echo ""
echo -e "${GREEN}Installation completed successfully!${NC}"
echo ""
