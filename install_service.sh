#!/bin/bash

# Installation script for Sasha Multi-Ping Monitor systemd user service

echo "Installing Sasha Multi-Ping Monitor as a systemd user service..."

# Create systemd user directory if it doesn't exist
mkdir -p ~/.config/systemd/user

# Copy service file
cp sasha-multiping.service ~/.config/systemd/user/

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable the service to start on login
systemctl --user enable sasha-multiping.service

echo ""
echo "Service installed successfully!"
echo ""
echo "Available commands:"
echo "  Start service:   systemctl --user start sasha-multiping"
echo "  Stop service:    systemctl --user stop sasha-multiping"
echo "  Restart service: systemctl --user restart sasha-multiping"
echo "  Check status:    systemctl --user status sasha-multiping"
echo "  View logs:       journalctl --user -u sasha-multiping -f"
echo "  Disable service: systemctl --user disable sasha-multiping"
echo ""
echo "The service will start automatically on next login."
echo "To start it now, run: systemctl --user start sasha-multiping"
