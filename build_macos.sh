#!/usr/bin/env bash
# =============================================================================
# build_macos.sh — configure + build easyai with the Metal/AMX backend.
#
# macOS-only. CMake auto-enables Metal on Apple Silicon and AMX on
# recent Intel; no explicit `-DGGML_METAL=ON` needed. This script just
# stays out of llama.cpp's way and lets it pick the right kernels.
#
# llama.cpp source — IMPORTANT
# ----------------------------
# This script does NOT clone llama.cpp. It links against whatever
# checkout already exists at the sibling directory `../llama.cpp`.
# The expected setup is that you ran
#
#   ./scripts/install_easyai_macos.sh
#
# at least once: the installer drops PrismML's fork next to easyai
# (the only fork with the Bonsai-8B-Q1_0 1.125-bit kernel), then
# builds + installs. After that you can use this script for fast
# rebuilds during development without re-cloning or re-installing.
#
# If the sibling is missing, this script bails with the exact clone
# command — pick the fork that matches the GGUF you intend to load:
#   * PrismML fork  → Bonsai-8B-Q1_0 (Q1_0 kernel only lives there)
#   * Upstream      → every other quant family
#
# This script ONLY configures + builds. No install step. Output lands
# under ./build-macos/. For the full first-time setup, use
# scripts/install_easyai_macos.sh.
#
# Build deps (one-time):
#   xcode-select --install
#   brew install cmake git curl
#
# Usage:
#   ./build_macos.sh                 # Release, all logical cores
#   ./build_macos.sh --debug         # Debug build
#   ./build_macos.sh --jobs 6        # explicit -j 6
#   ./build_macos.sh --rebuild       # wipe build-macos/ first
#   ./build_macos.sh --build-dir build-foo
# =============================================================================

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-$script_dir/build-macos}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)   REBUILD=1; shift ;;
        --debug)     BUILD_TYPE=Debug; shift ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# ===/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_macos.sh: not macOS — use ./build_vulkan.sh or ./build_cuda.sh" >&2
    exit 1
fi

# Sibling llama.cpp — must already be in place from
# install_easyai_macos.sh (PrismML fork) or a manual clone.
llama_dir="$(dirname "$script_dir")/llama.cpp"
if [[ ! -d "$llama_dir/.git" ]]; then
    cat >&2 <<EOF
build_macos.sh: missing sibling checkout at
   $llama_dir

easyai builds against ../llama.cpp. Two ways to populate it:

   # Recommended (also installs deps + Bonsai-tuned defaults):
   ./scripts/install_easyai_macos.sh

   # Manual — pick the fork that matches your GGUF:
   git clone https://github.com/PrismML-Eng/llama.cpp.git "$llama_dir"   # for Bonsai-8B-Q1_0
   #   OR
   git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir"      # upstream, every other quant
EOF
    exit 1
fi

# Show what fork is actually wired in, so the operator can confirm
# they're not accidentally building against an upstream checkout when
# they meant the PrismML fork (Bonsai will load but fail at decode).
llama_origin="$(git -C "$llama_dir" config --get remote.origin.url 2>/dev/null || echo unknown)"
llama_head="$(git -C "$llama_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "==> llama.cpp sibling: $llama_origin @ $llama_head"
case "$llama_origin" in
    *PrismML-Eng/llama.cpp*) echo "    (PrismML fork — Bonsai-8B-Q1_0 supported)" ;;
    *ggml-org/llama.cpp*)    echo "    (upstream — Bonsai-8B-Q1_0 will fail at decode; switch fork or pick another GGUF)" ;;
    *)                       echo "    (custom fork — proceeding, you know what you're doing)" ;;
esac

if [[ "$REBUILD" == "1" && -d "$BUILD_DIR" ]]; then
    echo "==> --rebuild: removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Foreign-ownership guard — same protection as install_easyai_macos.sh:
# a stray sudo run leaves root-owned files in the build dir, which
# breaks the next unprivileged cmake --build with a confusing
# permission error. Detect early.
if [[ -d "$BUILD_DIR" ]]; then
    foreign_owner="$(find "$BUILD_DIR" -maxdepth 2 ! -user "$USER" -print -quit 2>/dev/null || true)"
    if [[ -n "$foreign_owner" ]]; then
        cat >&2 <<EOF
==> $BUILD_DIR contains files not owned by $USER:
       $foreign_owner

This typically happens after a sudo invocation. Pick one and rerun:

   # keep the build cache, fix ownership:
   sudo chown -R $USER $BUILD_DIR

   # nuke and start fresh:
   sudo rm -rf $BUILD_DIR
EOF
        exit 1
    fi
fi

# brew openssl@3 is keg-only — find_package(OpenSSL) needs an explicit
# hint or it half-detects the macOS system libs and fails to create the
# imported OpenSSL::SSL target. Resolve once here and pass to CMake.
# Empty string is harmless to CMake (it'll fall back to its own search).
openssl_root=""
if command -v brew >/dev/null 2>&1; then
    openssl_root="$(brew --prefix openssl@3 2>/dev/null || true)"
fi
if [[ -z "$openssl_root" ]]; then
    echo "==> brew openssl@3 not found — HTTPS in easyai-cli will be disabled."
    echo "    install with: brew install openssl@3"
fi

echo "==> Configuring (Metal/AMX, $BUILD_TYPE, $BUILD_DIR)"
cmake -S "$script_dir" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DOPENSSL_ROOT_DIR="$openssl_root" \
    -DEASYAI_BUILD_EXAMPLES=ON \
    -DEASYAI_WITH_CURL=ON \
    -DEASYAI_BUILD_WEBUI=ON \
    -DEASYAI_INSTALL=ON

echo "==> Building (jobs=$JOBS)"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo "==> Done. Binaries in $BUILD_DIR:"
ls "$BUILD_DIR" 2>/dev/null | grep -E '^easyai' | sed 's/^/    /'
echo
echo "Run one without installing, e.g.:"
echo "    $BUILD_DIR/easyai-server -m \$HOME/easyai/models/Bonsai-8B-Q1_0.gguf \\"
echo "        --ngl 99 -c 8192 --temperature 0.5 --top-p 0.85 --top-k 20 \\"
echo "        --host 0.0.0.0 --port 8080"
