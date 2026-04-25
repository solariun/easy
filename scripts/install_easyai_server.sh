#!/usr/bin/env bash
# ============================================================================
# easyai installer — replaces install_llama_server.sh
#
# What this script does:
#   1. Installs build deps (CMake/Ninja/git/pkg-config + libcurl) and the
#      backend SDK that matches your hardware (Vulkan / CUDA / ROCm-HIP).
#   2. Clones llama.cpp + easyai (or uses existing sibling dirs).
#   3. Builds easyai (libeasyai + easyai-cli + easyai-server) with the
#      selected GPU backend.
#   4. Installs the binaries to $prefix/bin.
#   5. Creates a system user, /var/lib/easyai/{models,workspace}, an
#      /etc/easyai/system.txt + api_key file.
#   6. Drops a hardened systemd unit that runs easyai-server with mlock,
#      flash-attn, q8_0 KV cache, Bearer auth, and Prometheus /metrics.
#   7. (Linux only, optional) AMD-iGPU GTT kernel cmdline tweak; mDNS via
#      avahi; memlock+nofile limits; system swap off.
#
# What this script does NOT need to do (vs the old llama-server installer):
#   - No transparent proxy: easyai's HTTP layer already does the OpenAI-
#     compatible /v1/chat/completions itself.
#   - No SearXNG: web_search is a built-in tool that scrapes DuckDuckGo
#     directly via libcurl.
#   - No MCP bridge: tools live inside easyai-server and are auto-registered.
#   - No webui rebrand: the webui is a self-contained file embedded in the
#     binary (--webui-title would have nothing to patch).
#
# Linux / Debian target. The build itself works anywhere; this *installer*
# uses apt-get + systemd. macOS users: see the project README for the manual
# build matrix.
#
# Usage:
#   ./install_easyai_server.sh                       # full setup
#   ./install_easyai_server.sh --model /path/to.gguf # required for first run
#   ./install_easyai_server.sh --backend vulkan      # force backend
#   ./install_easyai_server.sh --service-port 8080
#   ./install_easyai_server.sh --service-host 0.0.0.0
#   ./install_easyai_server.sh --ctx-size 32768
#   ./install_easyai_server.sh --ngl 99            # GPU layers (-1=auto, 0=CPU)
#   ./install_easyai_server.sh --no-mlock --use-mmap
#   ./install_easyai_server.sh --webui-title "AI Box"
#   ./install_easyai_server.sh --webui-icon /path/to/logo.svg   # ico|png|svg|gif|jpg|webp
#   ./install_easyai_server.sh --upgrade             # git pull + rebuild
#   ./install_easyai_server.sh --enable-now          # systemctl start now
#   ./install_easyai_server.sh --no-service          # build/install only
#   ./install_easyai_server.sh -h                    # show this help
# ============================================================================

set -euo pipefail

# ---------- defaults --------------------------------------------------------
src_root="${src_root:-$HOME/opt}"
easyai_dir="$src_root/easyai"
llama_dir="$src_root/llama.cpp"
easyai_repo="${easyai_repo:-https://github.com/solariun/easy.git}"
llama_repo="${llama_repo:-https://github.com/ggml-org/llama.cpp.git}"
easyai_ref=""                                 # empty = main; pass --ref <sha|tag>
llama_ref=""                                  # empty = main; pass --llama-ref

install_prefix="/usr"
backend="auto"                                # auto|vulkan|cuda|hip|cpu
gtt_gb=28                                     # AMD iGPU GTT (only used by RDNA2/iGPU)
jobs="$(nproc 2>/dev/null || echo 4)"

do_install=1                                  # apt-get install deps
do_build=1
do_groups=1                                   # add user to render+video
do_limits=1                                   # /etc/security/limits.d for mlock
do_swap="off"                                 # off|tune|"" (keep)
do_kernel=1                                   # AMD iGPU GTT cmdline
do_service=1
do_force_service=0
do_enable_now=0
do_avahi=1
do_presets=1                                  # symlink easyai-cli → /usr/bin/ai
do_model=1
do_upgrade=0
copy_model=0

# easyai-server runtime config (compiled into the unit file)
service_user="easyai"
service_group="easyai"
service_home="/var/lib/easyai"
service_model_dir="$service_home/models"
service_model_link="ai.gguf"
service_workspace="$service_home/workspace"
service_host="0.0.0.0"
service_port=80                                # matches install_llama_server.sh default
service_alias="easyai"
service_name="easyai-server.service"

config_dir="/etc/easyai"
system_file="$config_dir/system.txt"
api_key_file="$config_dir/api_key"

ctx_size=128000
ngl=99                                        # --ngl <n>  (-1=auto, 0=CPU only, 99=all layers)
webui_title="easyai"                          # --webui-title <text>
webui_icon=""                                 # --webui-icon <path/to/.ico|.png|.svg|.gif|.jpg|.webp>
webui_icon_dest="$config_dir/favicon"         # final installed path under /etc/easyai
n_threads_default="$jobs"
n_threads_batch_default="$jobs"
preset="balanced"
thinking="on"
enable_metrics=1
enable_flash_attn=1
cache_type_k="q8_0"
cache_type_v="q8_0"
mlock=1
no_mmap=1
api_key=""                                    # leave empty to skip auth (open server)

model_src=""                                  # required when --no-model NOT passed

# ---------- arg parsing -----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-root)         src_root="$2"; easyai_dir="$src_root/easyai"; llama_dir="$src_root/llama.cpp"; shift 2 ;;
        --easyai-dir)       easyai_dir="$2"; shift 2 ;;
        --llama-dir)        llama_dir="$2"; shift 2 ;;
        --easyai-repo)      easyai_repo="$2"; shift 2 ;;
        --llama-repo)       llama_repo="$2"; shift 2 ;;
        --ref)              easyai_ref="$2"; shift 2 ;;
        --llama-ref)        llama_ref="$2"; shift 2 ;;
        --prefix)           install_prefix="$2"; shift 2 ;;
        --backend)          backend="$2"; shift 2 ;;
        --gtt)              gtt_gb="$2"; shift 2 ;;
        -j|--jobs)          jobs="$2"; shift 2 ;;
        --no-install)       do_install=0; shift ;;
        --no-build)         do_build=0; shift ;;
        --no-groups)        do_groups=0; shift ;;
        --no-limits)        do_limits=0; shift ;;
        --no-kernel)        do_kernel=0; shift ;;
        --no-service)       do_service=0; shift ;;
        --no-model)         do_model=0; shift ;;
        --no-avahi)         do_avahi=0; shift ;;
        --no-presets)       do_presets=0; shift ;;
        --no-swap)          do_swap="off"; shift ;;
        --swap-tune)        do_swap="tune"; shift ;;
        --keep-swap)        do_swap=""; shift ;;
        --upgrade)          do_upgrade=1; shift ;;
        --enable-now)       do_enable_now=1; shift ;;
        --no-enable)        do_enable_now=0; shift ;;
        --force-service)    do_force_service=1; shift ;;
        --service-host)     service_host="$2"; shift 2 ;;
        --service-port)     service_port="$2"; shift 2 ;;
        --alias)            service_alias="$2"; shift 2 ;;
        --ctx-size)         ctx_size="$2"; shift 2 ;;
        --ngl|--n-gpu-layers) ngl="$2"; shift 2 ;;
        --threads)          n_threads_default="$2"; shift 2 ;;
        --threads-batch)    n_threads_batch_default="$2"; shift 2 ;;
        --preset)           preset="$2"; shift 2 ;;
        --thinking)         thinking="$2"; shift 2 ;;
        --no-metrics)       enable_metrics=0; shift ;;
        --no-flash-attn)    enable_flash_attn=0; shift ;;
        --cache-type-k)     cache_type_k="$2"; shift 2 ;;
        --cache-type-v)     cache_type_v="$2"; shift 2 ;;
        --no-mlock)         mlock=0; shift ;;
        --use-mmap)         no_mmap=0; shift ;;
        --api-key)          api_key="$2"; shift 2 ;;
        --model)            model_src="$2"; shift 2 ;;
        --copy-model)       copy_model=1; shift ;;

        # ---- install_llama_server.sh drop-in compat ----------------------
        # These were the proxy / SearXNG / MCP / spec-decoding / webui-rebrand
        # knobs of the old installer.  They're accepted here so existing
        # provisioning scripts keep working unchanged; most are no-ops because
        # the corresponding feature is now built into easyai-server.
        --source-dir)       easyai_dir="$2"; shift 2 ;;   # alias for --easyai-dir
        --with-mcp|--no-mcp)
            warn "$1: ignored — easyai bundles web_search/web_fetch as built-in tools"
            shift ;;
        --webui-title)      webui_title="$2"; shift 2 ;;
        --webui-icon)       webui_icon="$2";  shift 2 ;;
        --thinking-budget)
            warn "--thinking-budget: not yet supported in easyai (use --thinking on/off + --max-tokens at runtime)"
            shift 2 ;;
        --draft-model|--draft-max|--draft-min)
            warn "$1: speculative decoding not yet wired up in easyai — flag ignored"
            shift 2 ;;
        --no-draft)
            warn "--no-draft: ignored (speculative decoding not enabled by default in easyai)"
            shift ;;
        --list-tags)
            # Mirror the original installer's behaviour: list recent tags of the
            # easyai repo instead of llama.cpp's.
            if [[ -d "$easyai_dir/.git" ]]; then
                git -C "$easyai_dir" fetch --tags --prune --force >&2 || true
                git -C "$easyai_dir" tag -l 2>/dev/null | tail -20 | sed 's/^/  /'
            else
                echo "  (no local clone yet — pass --easyai-dir or run a normal install first)"
            fi
            exit 0 ;;
        # -----------------------------------------------------------------

        -h|--help)          sed -n '2,46p' "$0"; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "run with --help for usage" >&2
            exit 2 ;;
    esac
done

# ---------- helpers ---------------------------------------------------------
log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }
ask()  { local r; read -rp "    $* [y/N] " r; [[ "$r" =~ ^[Yy]$ ]]; }

# ---------- pre-flight ------------------------------------------------------
[[ $EUID -eq 0 ]] && die "do not run as root — script calls sudo as needed"

if [[ "$(uname -s)" != "Linux" ]]; then
    die "this installer targets Linux. On macOS, build manually — see README.md (Build for your hardware)."
fi

command -v apt-get >/dev/null \
    || die "this script targets Debian/Ubuntu (apt). Adapt package names for other distros."

# ---------- backend auto-detect --------------------------------------------
detect_backend() {
    # honour explicit --backend
    if [[ "$backend" != "auto" ]]; then echo "$backend"; return; fi
    # NVIDIA?
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo cuda; return
    fi
    # AMD ROCm?
    if command -v rocminfo >/dev/null 2>&1; then
        echo hip; return
    fi
    # any GPU visible to Vulkan?
    if command -v vulkaninfo >/dev/null 2>&1 \
            && vulkaninfo --summary 2>/dev/null | grep -qiE 'deviceName'; then
        echo vulkan; return
    fi
    # AMD GPU PCI present but Vulkan not yet installed → still pick vulkan, we'll install the SDK
    if command -v lspci >/dev/null 2>&1 \
            && lspci 2>/dev/null | grep -qiE 'vga|display' \
            && lspci 2>/dev/null | grep -qiE 'amd|radeon'; then
        echo vulkan; return
    fi
    echo cpu
}
backend_resolved="$(detect_backend)"

case "$backend_resolved" in
    auto|vulkan|cuda|hip|cpu) ;;
    *) die "unknown backend: $backend_resolved" ;;
esac

# ---------- print effective config -----------------------------------------
gtt_pages=$(( gtt_gb * 262144 ))     # GTT page count, 4 KiB per page

log "effective config:"
printf '    src_root         = %s\n' "$src_root"
printf '    easyai_dir       = %s\n' "$easyai_dir"
printf '    llama_dir        = %s\n' "$llama_dir"
printf '    install_prefix   = %s\n' "$install_prefix"
printf '    backend          = %s\n' "$backend_resolved"
printf '    service_host     = %s\n' "$service_host"
printf '    service_port     = %s\n' "$service_port"
printf '    service_alias    = %s\n' "$service_alias"
printf '    ctx_size         = %s\n' "$ctx_size"
printf '    ngl              = %s   (-1=auto, 0=CPU only, 99=all GPU layers)\n' "$ngl"
printf '    threads / batch  = %s / %s\n' "$n_threads_default" "$n_threads_batch_default"
printf '    preset           = %s  thinking=%s\n' "$preset" "$thinking"
printf '    KV cache         = K=%s  V=%s  flash_attn=%s\n' "$cache_type_k" "$cache_type_v" "$enable_flash_attn"
printf '    memory           = mlock=%s  no_mmap=%s\n' "$mlock" "$no_mmap"
printf '    metrics          = %s\n' "$enable_metrics"
printf '    webui_title      = %s\n' "$webui_title"
printf '    webui_icon       = %s\n' "${webui_icon:-<default — no icon>}"
printf '    api_key          = %s\n' "$([[ -n "$api_key" ]] && echo "<set>" || echo "<none — server is open>")"
printf '    model_src        = %s\n' "${model_src:-<none — pass --model PATH>}"
printf '    flags            = install:%s build:%s groups:%s limits:%s kernel:%s\n' \
    "$do_install" "$do_build" "$do_groups" "$do_limits" "$do_kernel"
printf '                       service:%s model:%s avahi:%s presets:%s\n' \
    "$do_service" "$do_model" "$do_avahi" "$do_presets"
printf '                       enable_now:%s force_service:%s upgrade:%s\n' \
    "$do_enable_now" "$do_force_service" "$do_upgrade"
echo

if [[ $do_model -eq 1 && -z "$model_src" ]]; then
    warn "no --model PATH given. Pass it later or rerun with --no-model to skip."
fi

# ---------- detected hardware ----------------------------------------------
log "detected hardware:"
printf '    CPU : %s\n' "$(lscpu | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')"
printf '    RAM : %s\n' "$(free -h | awk '/^Mem:/ {print $2}')"
if command -v lspci >/dev/null; then
    printf '    GPU : %s\n' \
        "$(lspci | grep -iE 'vga|display' | head -1 | cut -d: -f3- | sed 's/^ //')"
fi

# ---------- install dependencies -------------------------------------------
if [[ $do_install -eq 1 ]]; then
    log "installing common build deps + libcurl + system tools"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ccache pkg-config curl ca-certificates \
        libcurl4-openssl-dev libomp-dev libcap2-bin jq

    case "$backend_resolved" in
        vulkan)
            log "installing Vulkan SDK + Mesa drivers"
            sudo apt-get install -y --no-install-recommends \
                mesa-vulkan-drivers vulkan-tools libvulkan-dev \
                glslc glslang-tools spirv-tools libshaderc-dev
            ;;
        cuda)
            log "CUDA backend selected — assuming nvidia-cuda-toolkit is installed."
            warn "If 'nvcc --version' fails, install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
            ;;
        hip)
            log "ROCm/HIP backend selected — assuming rocm-dev is installed."
            warn "Install the ROCm SDK manually if rocminfo / hipcc are missing."
            ;;
        cpu)
            log "CPU-only backend — no GPU SDK to install."
            ;;
    esac

    if [[ $do_avahi -eq 1 ]]; then
        sudo apt-get install -y --no-install-recommends avahi-daemon avahi-utils
    fi
fi

# Sanity check (Vulkan only).
if [[ "$backend_resolved" == "vulkan" ]] && command -v vulkaninfo >/dev/null; then
    log "Vulkan device check"
    if ! vulkaninfo --summary 2>/dev/null \
            | grep -qiE 'deviceName'; then
        warn "vulkaninfo returned no device — Mesa/driver state may be wrong"
    else
        vulkaninfo --summary 2>/dev/null \
            | grep -iE 'deviceName|driverName' | sed 's/^/    /'
    fi
fi

# ---------- clone / fetch sources ------------------------------------------
fetch_repo() {
    local dir="$1" repo="$2" ref="$3"
    if [[ -d "$dir/.git" ]]; then
        if [[ $do_upgrade -eq 1 ]]; then
            log "git fetch in $dir"
            git -C "$dir" fetch --tags --prune --force
        fi
    else
        log "cloning $repo → $dir"
        mkdir -p "$(dirname "$dir")"
        git clone --filter=blob:none "$repo" "$dir"
    fi
    if [[ -n "$ref" ]]; then
        log "checking out ref '$ref' in $dir"
        git -C "$dir" checkout "$ref"
    fi
}

# llama.cpp must sit next to easyai (CMakeLists looks at ../llama.cpp).
fetch_repo "$llama_dir"  "$llama_repo"  "$llama_ref"
fetch_repo "$easyai_dir" "$easyai_repo" "$easyai_ref"

# Symlink llama.cpp as a sibling of easyai if they aren't already.
expected_llama="$(dirname "$easyai_dir")/llama.cpp"
if [[ "$llama_dir" != "$expected_llama" ]]; then
    if [[ ! -e "$expected_llama" ]]; then
        log "symlinking $llama_dir → $expected_llama"
        ln -s "$llama_dir" "$expected_llama"
    elif [[ ! -L "$expected_llama" ]]; then
        warn "$expected_llama exists and is not a symlink; leaving it alone"
    fi
fi

# ---------- build easyai ----------------------------------------------------
if [[ $do_build -eq 1 ]]; then
    log "configuring easyai build (backend=$backend_resolved)"
    cmake_flags=( -DCMAKE_BUILD_TYPE=Release -DEASYAI_BUILD_EXAMPLES=ON )
    case "$backend_resolved" in
        vulkan)  cmake_flags+=( -DGGML_VULKAN=ON ) ;;
        cuda)    cmake_flags+=( -DGGML_CUDA=ON ) ;;
        hip)     cmake_flags+=( -DGGML_HIP=ON ) ;;
        cpu)     ;;  # no GPU flag
    esac

    pushd "$easyai_dir" >/dev/null
    cmake -S . -B build "${cmake_flags[@]}"
    log "building (jobs=$jobs)"
    cmake --build build -j "$jobs"
    popd >/dev/null
fi

# ---------- post-build sanity: which GPU backend was actually compiled? ----
# We honour what the build produced over what the user asked for, so a CPU-
# only binary doesn't spam GPU-related errors at runtime via --ngl.
detected_backends=""
while IFS= read -r so; do
    name=$(basename "$so" | sed -E 's/^libggml-([a-z0-9]+)\.so.*/\1/')
    case "$name" in
        base|cpu) ;;
        *) detected_backends="$detected_backends $name" ;;
    esac
done < <(find "$easyai_dir/build" -maxdepth 8 -name 'libggml-*.so*' 2>/dev/null | sort -u)
detected_backends=$(echo "$detected_backends" | xargs -n1 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//; s/,/, /g')

if [[ -z "$detected_backends" ]]; then
    if [[ "$ngl" -ne 0 ]]; then
        warn "build is CPU-only (no libggml-{vulkan,cuda,hip,metal}.so found)"
        warn "forcing --ngl 0 in the systemd unit"
        ngl=0
    fi
else
    log "GPU backends compiled into the build: $detected_backends"
    if [[ "$backend_resolved" != "cpu" ]]; then
        # Sanity: the chosen backend actually produced a library?
        case "$backend_resolved" in
            vulkan) [[ "$detected_backends" == *vulkan* ]] || warn "asked for vulkan but no libggml-vulkan.so was produced" ;;
            cuda)   [[ "$detected_backends" == *cuda*   ]] || warn "asked for cuda but no libggml-cuda.so was produced" ;;
            hip)    [[ "$detected_backends" == *hip*    ]] || warn "asked for hip but no libggml-hip.so was produced" ;;
        esac
    fi
fi

# ---------- install binaries ------------------------------------------------
if [[ $do_build -eq 1 ]]; then
    log "installing binaries to $install_prefix/bin"
    sudo install -Dm755 "$easyai_dir/build/easyai-server" "$install_prefix/bin/easyai-server"
    sudo install -Dm755 "$easyai_dir/build/easyai-cli"    "$install_prefix/bin/easyai-cli"
    sudo install -Dm755 "$easyai_dir/build/easyai-agent"  "$install_prefix/bin/easyai-agent" || true
    sudo install -Dm755 "$easyai_dir/build/easyai-chat"   "$install_prefix/bin/easyai-chat"  || true
    # libeasyai.so + dynamically-loaded llama / ggml libs.
    #
    # llama.cpp's targets land under several subdirs of build/_deps/llama.cpp/
    # (src/, common/, ggml/src/, ggml/src/ggml-*/) and use versioned SONAMEs
    # like libllama.so.0.  We need ALL of them — both the bare *.so symlinks
    # and the actual *.so.N files they point at — so the runtime loader can
    # resolve everything libeasyai.so depends on.
    sudo install -Dm644 "$easyai_dir/build/libeasyai.so" "$install_prefix/lib/libeasyai.so" || true
    while IFS= read -r so; do
        [[ -f "$so" || -L "$so" ]] || continue
        # `cp -P` preserves the symlink chain so libllama.so → libllama.so.0
        # → libllama.so.0.10.0 all land alongside each other.
        sudo cp -Pf "$so" "$install_prefix/lib/$(basename "$so")"
    done < <(find "$easyai_dir/build" \
                  \( -name 'libllama*.so*' -o -name 'libggml*.so*' \
                  -o -name 'libllama-common*.so*' -o -name 'libllava*.so*' \
                  -o -name 'libcpp-httplib*.so*' \) -print 2>/dev/null)
    sudo ldconfig || true

    if [[ $do_presets -eq 1 ]]; then
        log "installing easyai-cli as 'ai' shortcut → $install_prefix/bin/ai"
        sudo ln -sf "$install_prefix/bin/easyai-cli" "$install_prefix/bin/ai"
    fi
fi

# ---------- system user + dirs ---------------------------------------------
if [[ $do_service -eq 1 ]]; then
    if ! id -u "$service_user" >/dev/null 2>&1; then
        log "creating system user '$service_user'"
        sudo useradd --system --home-dir "$service_home" --shell /usr/sbin/nologin \
            --comment "easyai server" "$service_user"
    fi

    log "creating $service_home + subdirs"
    sudo install -d -o "$service_user" -g "$service_group" -m 750 \
        "$service_home" "$service_model_dir" "$service_workspace"

    log "creating $config_dir"
    sudo install -d -o root -g "$service_group" -m 750 "$config_dir"

    if [[ ! -f "$system_file" ]]; then
        log "writing default $system_file (edit later with: sudo nano $system_file)"
        sudo bash -c "cat > '$system_file'" <<'SYS'
You are easyai, a concise and helpful assistant. When a tool is appropriate, use it.
Prefer correctness over verbosity. Cite sources when you use web_fetch / web_search.
SYS
        sudo chmod 640 "$system_file"
        sudo chown root:"$service_group" "$system_file"
    fi

    if [[ -n "$api_key" ]]; then
        log "writing $api_key_file (mode 600)"
        sudo bash -c "printf '%s' '$api_key' > '$api_key_file'"
        sudo chmod 600 "$api_key_file"
        sudo chown "$service_user":"$service_group" "$api_key_file"
    elif [[ -f "$api_key_file" ]]; then
        warn "leaving existing $api_key_file in place (use --api-key '' to clear)"
    fi

    # ---- favicon: copy operator-supplied icon to /etc/easyai/favicon
    #              and let the unit point easyai-server at it -----------
    if [[ -n "$webui_icon" ]]; then
        [[ -f "$webui_icon" ]] || die "favicon not found: $webui_icon"
        # preserve the original extension so easyai-server can pick the
        # right Content-Type at runtime.
        ext="${webui_icon##*.}"
        webui_icon_dest="$config_dir/favicon.$ext"
        log "installing favicon → $webui_icon_dest"
        sudo install -Dm644 -o root -g "$service_group" \
            "$webui_icon" "$webui_icon_dest"
    else
        webui_icon_dest=""
    fi
fi

# ---------- groups (render/video for GPU access) ---------------------------
if [[ $do_groups -eq 1 && $do_service -eq 1 ]]; then
    case "$backend_resolved" in
        vulkan|hip|cuda)
            for grp in render video; do
                if getent group "$grp" >/dev/null; then
                    log "adding $service_user to group '$grp'"
                    sudo usermod -aG "$grp" "$service_user" || true
                fi
            done
            ;;
    esac
fi

# ---------- memlock / nofile limits ----------------------------------------
if [[ $do_limits -eq 1 && $do_service -eq 1 ]]; then
    limits_file="/etc/security/limits.d/easyai.conf"
    log "writing $limits_file"
    sudo tee "$limits_file" >/dev/null <<EOF
$service_user soft memlock unlimited
$service_user hard memlock unlimited
$service_user soft nofile  1048576
$service_user hard nofile  1048576
EOF
fi

# ---------- swap tuning -----------------------------------------------------
case "$do_swap" in
    off)
        log "disabling swap (pair with --no-mlock to opt out)"
        sudo swapoff -a || true
        if [[ -f /etc/fstab ]]; then
            sudo sed -ri.bak '/^[^#].*\sswap\s/s/^/#/' /etc/fstab || true
        fi
        ;;
    tune)
        log "keeping swap, setting swappiness=1, vfs_cache_pressure=50"
        echo 1  | sudo tee /proc/sys/vm/swappiness        >/dev/null || true
        echo 50 | sudo tee /proc/sys/vm/vfs_cache_pressure >/dev/null || true
        sudo tee /etc/sysctl.d/60-easyai-swap.conf >/dev/null <<'EOF'
# easyai: keep swap as a safety net but make the kernel almost never use it,
# and don't aggressively reclaim VFS caches (large GGUFs benefit from it).
vm.swappiness = 1
vm.vfs_cache_pressure = 50
EOF
        sudo sysctl --system 2>&1 | grep -E 'swappiness|cache_pressure' | sed 's/^/    /'
        ;;
    "")
        log "leaving swap untouched"
        ;;
esac

# ---------- AMD iGPU GTT kernel cmdline ------------------------------------
# Only meaningful for RDNA2 iGPUs that need a large GTT to fit a model.
if [[ $do_kernel -eq 1 && "$backend_resolved" == "vulkan" ]]; then
    if grep -qiE 'amd|radeon' <(lspci 2>/dev/null) \
            && [[ -f /etc/default/grub ]]; then
        log "patching /etc/default/grub for ttm.pages_limit=$gtt_pages (GTT $gtt_gb GiB)"
        if ! grep -q 'ttm.pages_limit' /etc/default/grub; then
            sudo sed -ri \
                "s|^(GRUB_CMDLINE_LINUX_DEFAULT=\")(.*)\"|\\1\\2 ttm.pages_limit=$gtt_pages\"|" \
                /etc/default/grub
            sudo update-grub || sudo grub-mkconfig -o /boot/grub/grub.cfg || true
            warn "kernel cmdline updated — reboot needed for the new GTT to take effect"
        else
            log "ttm.pages_limit already present; skipping"
        fi
    fi
fi

# ---------- model placement -------------------------------------------------
if [[ $do_service -eq 1 && $do_model -eq 1 && -n "$model_src" ]]; then
    [[ -f "$model_src" ]] || die "model not found: $model_src"
    dest="$service_model_dir/$(basename "$model_src")"
    if [[ "$copy_model" -eq 1 ]]; then
        log "copying model → $dest"
        sudo install -Dm640 -o "$service_user" -g "$service_group" "$model_src" "$dest"
    else
        log "moving model → $dest"
        sudo mv "$model_src" "$dest"
        sudo chown "$service_user":"$service_group" "$dest"
        sudo chmod 640 "$dest"
    fi
    log "symlinking $service_model_dir/$service_model_link → $(basename "$model_src")"
    sudo ln -sfn "$(basename "$model_src")" "$service_model_dir/$service_model_link"
fi

# ---------- systemd unit ----------------------------------------------------
if [[ $do_service -eq 1 ]]; then
    if [[ $do_force_service -eq 1 ]]; then
        log "stopping + removing existing $service_name (--force-service)"
        sudo systemctl stop    "$service_name" 2>/dev/null || true
        sudo systemctl disable "$service_name" 2>/dev/null || true
        sudo rm -f "/etc/systemd/system/$service_name"
    fi

    # Build the easyai-server arg list at template time so the unit is fully
    # explicit and the operator can `systemctl cat easyai-server` to audit.
    args=( -m "$service_model_dir/$service_model_link" )
    args+=( --host "$service_host" --port "$service_port" )
    args+=( --alias "$service_alias" )
    args+=( -c "$ctx_size" )
    args+=( --ngl "$ngl" )
    args+=( -t "$n_threads_default" -tb "$n_threads_batch_default" )
    args+=( --preset "$preset" )
    args+=( --sandbox "$service_workspace" )
    args+=( --system-file "$system_file" )
    [[ -n "$webui_title"     ]] && args+=( --webui-title "$webui_title" )
    [[ -n "$webui_icon_dest" ]] && args+=( --webui-icon  "$webui_icon_dest" )
    [[ "$enable_flash_attn" -eq 1 ]] && args+=( -fa )
    [[ -n "$cache_type_k" ]]         && args+=( -ctk "$cache_type_k" )
    [[ -n "$cache_type_v" ]]         && args+=( -ctv "$cache_type_v" )
    [[ "$mlock"   -eq 1 ]]           && args+=( --mlock )
    [[ "$no_mmap" -eq 1 ]]           && args+=( --no-mmap )
    [[ "$enable_metrics" -eq 1 ]]    && args+=( --metrics )
    [[ "$thinking" == "off" ]]       && args+=( --reasoning off )

    # api-key sourced from a file at runtime (so it's never visible in `ps`).
    exec_pre=""
    if [[ -f "$api_key_file" ]]; then
        # systemd `EnvironmentFile=` won't substitute into ExecStart, so we
        # wrap with a tiny shell that reads the file and exports the key.
        # We use --api-key "$EASYAI_API_KEY" inside the unit.
        args+=( --api-key '${EASYAI_API_KEY}' )
        exec_pre="EnvironmentFile=$api_key_file_envfile_marker"
    fi

    # Convert the args array to a single quoted line for ExecStart.
    arg_string=""
    for a in "${args[@]}"; do
        # quote every arg
        arg_string+=" $(printf '%q' "$a")"
    done

    log "writing /etc/systemd/system/$service_name"
    sudo tee "/etc/systemd/system/$service_name" >/dev/null <<UNIT
[Unit]
Description=easyai OpenAI-compatible LLM server
Documentation=https://github.com/solariun/easy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
Group=$service_group
WorkingDirectory=$service_home

# Render / video groups give the service access to /dev/dri/* (GPU). The
# usermod step above adds them to the user, but baking them into the unit
# means they apply even if the user gets recreated.
SupplementaryGroups=render video

# Make the service much less likely to be picked by the OOM killer when
# RAM gets tight (model + KV + activations dominate the box's footprint).
OOMScoreAdjust=-700

# Run the inference loop at SCHED_FIFO priority — the original
# install_llama_server.sh used this for steady token throughput on the
# 680M iGPU.  Requires "RestrictRealtime=no" (default) and an unbounded
# rt budget on modern systemd, which is the kernel default.
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

# Mesa RADV graphics-pipeline-library: ~10-15% faster inference on RDNA2
# iGPUs (Radeon 680M, 780M, …). Harmless on other backends.
Environment=RADV_PERFTEST=gpl
Environment=HOME=$service_home
Environment=XDG_CACHE_HOME=$service_home/cache
$( [[ -f "$api_key_file" ]] && echo "Environment=\"EASYAI_API_KEY_FILE=$api_key_file\"" )
$( [[ -f "$api_key_file" ]] && echo "ExecStartPre=/bin/sh -c 'test -r \"$api_key_file\"'" )
ExecStart=/bin/sh -c '$( [[ -f "$api_key_file" ]] && echo "EASYAI_API_KEY=\$(cat $api_key_file) " )exec $install_prefix/bin/easyai-server$arg_string'
Restart=on-failure
RestartSec=10
KillSignal=SIGINT
# Loading a 30B+ GGUF over a slow disk can take a minute; don't time out.
TimeoutStartSec=0
TimeoutStopSec=20

# hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=$service_home $config_dir
ProtectHome=true
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
LockPersonality=yes
RestrictSUIDSGID=yes
LimitMEMLOCK=infinity
LimitNOFILE=1048576

# Allow binding to low ports (only relevant if --service-port < 1024)
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
UNIT

    sudo systemctl daemon-reload

    if [[ $do_enable_now -eq 1 ]]; then
        log "enabling + starting $service_name"
        sudo systemctl enable --now "$service_name"
        sleep 1
        sudo systemctl --no-pager --full status "$service_name" || true
    else
        log "unit installed but not started. Start it with:"
        printf '      sudo systemctl enable --now %s\n' "$service_name"
        printf '      sudo journalctl -u %s -f\n' "$service_name"
    fi
fi

# ---------- avahi / mDNS ----------------------------------------------------
if [[ $do_avahi -eq 1 && $do_service -eq 1 ]]; then
    if command -v avahi-daemon >/dev/null; then
        avahi_service="/etc/avahi/services/easyai.service"
        log "writing $avahi_service ($(hostname).local advertisement)"
        sudo tee "$avahi_service" >/dev/null <<AVA
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">easyai @ %h</name>
  <service>
    <type>_http._tcp</type>
    <port>$service_port</port>
    <txt-record>path=/v1</txt-record>
    <txt-record>alias=$service_alias</txt-record>
  </service>
</service-group>
AVA
        sudo systemctl restart avahi-daemon || true
    fi
fi

# ---------- summary ---------------------------------------------------------
echo
log "DONE."
echo
printf '  binary    : %s\n' "$install_prefix/bin/easyai-server"
printf '  shortcut  : %s\n' "$install_prefix/bin/ai (-> easyai-cli)"
printf '  service   : %s\n' "$service_name"
printf '  webui     : http://%s:%s/\n' \
    "$([[ "$service_host" == "0.0.0.0" ]] && hostname || echo "$service_host")" "$service_port"
printf '  api base  : http://%s:%s/v1\n' \
    "$([[ "$service_host" == "0.0.0.0" ]] && hostname || echo "$service_host")" "$service_port"
printf '  health    : http://localhost:%s/health\n' "$service_port"
[[ $enable_metrics -eq 1 ]] && \
    printf '  metrics   : http://localhost:%s/metrics\n' "$service_port"
printf '  system.txt: %s   (sudo nano %s)\n' "$system_file" "$system_file"
printf '  webui     : title="%s"\n' "$webui_title"
[[ -n "$webui_icon_dest" ]] && \
    printf '              icon="%s" (served at /favicon and /favicon.ico)\n' "$webui_icon_dest"
[[ -f "$api_key_file" ]] && \
    printf '  api key   : %s   (Bearer auth required on /v1)\n' "$api_key_file"
echo
printf '  test (open server):\n'
printf '    curl http://localhost:%s/v1/chat/completions \\\n' "$service_port"
printf '         -H "Content-Type: application/json" \\\n'
if [[ -f "$api_key_file" ]]; then
    printf '         -H "Authorization: Bearer $(cat %s)" \\\n' "$api_key_file"
fi
printf '         -d '\''{"messages":[{"role":"user","content":"hi"}]}'\''\n'
echo
printf '  point any OpenAI client at:  http://%s:%s/v1\n' \
    "$([[ "$service_host" == "0.0.0.0" ]] && hostname || echo "$service_host")" "$service_port"
echo
