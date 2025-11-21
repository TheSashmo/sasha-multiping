#!/bin/bash
# Restore desktop notifications

echo "Restoring desktop notifications..."

# Re-enable notification banners
sudo -u user DISPLAY=:0 gsettings set org.gnome.desktop.notifications show-banners true

# Re-enable update notifications
sudo -u user DISPLAY=:0 gsettings set org.gnome.desktop.notifications.application:/org/gnome/desktop/notifications/application/update-manager/ enable true

# Re-enable apt daily updates
sudo systemctl unmask apt-daily.service apt-daily-upgrade.service

echo "Notifications restored!"
