#!/usr/bin/env bash

[[ -e /hive/custom ]] && . /hive/custom/keryx-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/keryx-miner/h-manifest.conf

# The zero-dup binary talks to the GPU via Vulkan (mining AND in-process inference).
# Most current HiveOS images ship libvulkan1; install it if this one doesn't.
if ! ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then
    echo "keryx-miner: libvulkan1 not found — installing"
    apt-get update -qq && apt-get install -y -qq libvulkan1 || \
        echo "WARNING: could not install libvulkan1; the miner will not start without it"
fi

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
