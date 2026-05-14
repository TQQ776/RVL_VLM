#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="${ANYGRASP_SDK_DIR:-/home/tqq/TQQ_ws/third_party/anygrasp_sdk}"
LICENSE_SRC=""
CHECKPOINT_SRC=""

usage() {
  cat <<'USAGE'
Install AnyGrasp license and checkpoint files into the local SDK tree.

Usage:
  install_anygrasp_assets.sh --license /path/to/license.zip --checkpoint /path/to/checkpoint_detection.tar
  install_anygrasp_assets.sh --license /path/to/license
  install_anygrasp_assets.sh --checkpoint /path/to/checkpoint_detection.tar

Environment:
  ANYGRASP_SDK_DIR  Override SDK directory. Default:
                   /home/tqq/TQQ_ws/third_party/anygrasp_sdk
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --license)
      LICENSE_SRC="${2:-}"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_SRC="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$LICENSE_SRC" ] && [ -z "$CHECKPOINT_SRC" ]; then
  usage >&2
  exit 2
fi

DETECTION_DIR="$SDK_DIR/grasp_detection"
LICENSE_DST="$DETECTION_DIR/license"
CHECKPOINT_DST="$DETECTION_DIR/log/checkpoint_detection.tar"

if [ ! -d "$DETECTION_DIR" ]; then
  echo "AnyGrasp SDK grasp_detection directory not found: $DETECTION_DIR" >&2
  exit 1
fi

install_license_dir() {
  local src_dir="$1"
  if [ -f "$src_dir/licenseCfg.json" ]; then
    rm -rf "$LICENSE_DST"
    mkdir -p "$LICENSE_DST"
    cp -a "$src_dir"/. "$LICENSE_DST"/
  elif [ -d "$src_dir/license" ] && [ -f "$src_dir/license/licenseCfg.json" ]; then
    rm -rf "$LICENSE_DST"
    cp -a "$src_dir/license" "$LICENSE_DST"
  else
    echo "Could not find licenseCfg.json in $src_dir or $src_dir/license" >&2
    exit 1
  fi
}

if [ -n "$LICENSE_SRC" ]; then
  if [ ! -e "$LICENSE_SRC" ]; then
    echo "License path does not exist: $LICENSE_SRC" >&2
    exit 1
  fi

  if [ -d "$LICENSE_SRC" ]; then
    install_license_dir "$LICENSE_SRC"
  else
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    unzip -q "$LICENSE_SRC" -d "$tmp_dir"
    install_license_dir "$tmp_dir"
  fi

  echo "Installed AnyGrasp license to: $LICENSE_DST"
fi

if [ -n "$CHECKPOINT_SRC" ]; then
  if [ ! -f "$CHECKPOINT_SRC" ]; then
    echo "Checkpoint file does not exist: $CHECKPOINT_SRC" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$CHECKPOINT_DST")"
  cp -a "$CHECKPOINT_SRC" "$CHECKPOINT_DST"
  echo "Installed AnyGrasp checkpoint to: $CHECKPOINT_DST"
fi

GSNET_SRC="$DETECTION_DIR/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so"
LIB_CXX_SRC="$SDK_DIR/license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so"
if [ -f "$GSNET_SRC" ]; then
  cp -a "$GSNET_SRC" "$DETECTION_DIR/gsnet.so"
fi
if [ -f "$LIB_CXX_SRC" ]; then
  cp -a "$LIB_CXX_SRC" "$DETECTION_DIR/lib_cxx.so"
fi

mkdir -p "$SDK_DIR/lib" "$SDK_DIR/bin"
OPENSSL_DIR="/home/tqq/miniconda3/pkgs/openssl-1.1.1w-h7f8727e_0/lib"
if [ -f "$OPENSSL_DIR/libcrypto.so.1.1" ]; then
  ln -sf "$OPENSSL_DIR/libcrypto.so.1.1" "$SDK_DIR/lib/libcrypto.so.1.1"
fi
if [ -f "$OPENSSL_DIR/libssl.so.1.1" ]; then
  ln -sf "$OPENSSL_DIR/libssl.so.1.1" "$SDK_DIR/lib/libssl.so.1.1"
fi
if ! command -v ifconfig >/dev/null 2>&1; then
  cat > "$SDK_DIR/bin/ifconfig" <<'EOF'
#!/bin/sh
exec /usr/sbin/ip addr
EOF
  chmod +x "$SDK_DIR/bin/ifconfig"
fi

echo
echo "Next test:"
echo "  LD_LIBRARY_PATH=$SDK_DIR/lib:\$LD_LIBRARY_PATH /usr/bin/python3 - <<'PY'"
echo "  import sys"
echo "  root='$SDK_DIR'"
echo "  det=root+'/grasp_detection'"
echo "  for p in [root, det, det+'/gsnet_versions', root+'/pointnet2']:"
echo "      sys.path.insert(0, p)"
echo "  from gsnet import AnyGrasp"
echo "  print('AnyGrasp import ok')"
echo "  PY"
