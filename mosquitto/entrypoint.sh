#!/bin/sh
# Build the Mosquitto password file from MQTT_PASSWORD_<USER> env vars, then start.
# Users must match the `user` entries in mosquitto/acl (tests/test_acl.py checks this).
set -eu

USERS="simulator bootstrap edge-adapter historian graph-sync anomaly yield dashboard i3x healthcheck explorer"
RUN_DIR=/mosquitto/run
PASSWD="$RUN_DIR/passwd"

mkdir -p "$RUN_DIR"
rm -f "$PASSWD"
touch "$PASSWD"
chmod 0600 "$PASSWD"

for user in $USERS; do
  var="MQTT_PASSWORD_$(echo "$user" | tr 'a-z-' 'A-Z_')"
  eval "pw=\${$var:-}"
  if [ -z "$pw" ]; then
    echo "entrypoint: $var is not set" >&2
    exit 1
  fi
  mosquitto_passwd -b "$PASSWD" "$user" "$pw"
done

chown -R mosquitto:mosquitto "$RUN_DIR" /mosquitto/data
chmod 0700 "$RUN_DIR"

exec mosquitto -c /mosquitto/config/mosquitto.conf
