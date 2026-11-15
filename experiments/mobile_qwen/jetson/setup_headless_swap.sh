#!/usr/bin/env bash
# User authorized SD-card swap after confirming there is no NVMe.
set -euo pipefail
swap_path=/swapfile-ali-jetson
if [[ ! -e "$swap_path" ]]; then
  fallocate -l 8G "$swap_path"
  chmod 600 "$swap_path"
  mkswap "$swap_path"
fi
if ! swapon --show=NAME --noheadings | grep -Fxq "$swap_path"; then
  swapon --priority -2 "$swap_path"
fi
if ! grep -Fq "$swap_path " /etc/fstab; then
  cp -a /etc/fstab /etc/fstab.before-ali-jetson-20261114
  printf '%s\n' "$swap_path none swap sw,pri=-2 0 0" >> /etc/fstab
fi
systemctl set-default multi-user.target
swapon --show
free -h
/usr/sbin/nvpmodel -q
/usr/bin/jetson_clocks --show
