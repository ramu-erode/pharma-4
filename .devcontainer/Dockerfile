FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships an apt source for dl.yarnpkg.com whose signing key is no
# longer present. Every later `apt-get update` then fails with
#   NO_PUBKEY 62D54FD4003F6525 / "The repository ... is not signed"
# which aborts dev container feature installs (docker-in-docker hits it first).
#
# Features are installed on top of this image, so the source has to be removed
# here, before any feature runs. Nothing in this project uses Yarn.
#
# Upstream: https://github.com/devcontainers/features/issues/1543
#           https://github.com/devcontainers/images/issues/1797
USER root
RUN set -eux; \
    sed -i '/dl\.yarnpkg\.com/d' /etc/apt/sources.list 2>/dev/null || true; \
    grep -rl 'dl\.yarnpkg\.com' /etc/apt/sources.list.d 2>/dev/null | xargs -r rm -f || true; \
    apt-get update; \
    rm -rf /var/lib/apt/lists/*

# devcontainer.json mounts a named volume at /home/vscode/.claude so the Claude
# Code login survives rebuilds. Docker seeds a new named volume from whatever is
# at the mount point in the image, ownership included -- and creates it owned by
# root if nothing is there. Claude Code runs as `vscode`, so a root-owned volume
# makes it fail with "Failed to retrieve auth status after login". Create the
# directory with the right owner so the volume inherits it.
RUN mkdir -p /home/vscode/.claude \
 && chown -R vscode:vscode /home/vscode/.claude
