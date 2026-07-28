#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: glm52-lmcache-wrapper.sh requires a model server command" >&2
  exit 2
fi

mode="${LMCACHE_MODE:-off}"
mode="${mode,,}"
case "${mode}" in
  off|0)
    exec "$@"
    ;;
  ram|memory|1)
    mode=ram
    ;;
  disk|ram-disk|memory-disk)
    mode=disk
    ;;
  *)
    echo "ERROR: LMCACHE_MODE must be off, ram, or disk; got ${mode}" >&2
    exit 2
    ;;
esac

command -v lmcache >/dev/null || {
  echo "ERROR: LMCACHE_MODE=${mode}, but the lmcache CLI is not installed" >&2
  exit 2
}

service_port="${PORT:-8000}"
port_offset=0
if [[ "${service_port}" =~ ^[0-9]+$ ]] && (( service_port >= 8000 )); then
  port_offset=$((service_port - 8000))
fi

lmcache_host="${LMCACHE_HOST:-127.0.0.1}"
lmcache_port="${LMCACHE_PORT:-$((5555 + port_offset))}"
lmcache_http_port="${LMCACHE_HTTP_PORT:-$((8089 + port_offset))}"
lmcache_prometheus_port="${LMCACHE_PROMETHEUS_PORT:-$((9090 + port_offset))}"
lmcache_chunk_size="${LMCACHE_CHUNK_SIZE:-512}"
lmcache_l1_gb="${LMCACHE_L1_GB:-24}"
lmcache_l1_init_gb="${LMCACHE_L1_INIT_GB:-${lmcache_l1_gb}}"
lmcache_gpu_workers="${LMCACHE_MAX_GPU_WORKERS:-1}"
lmcache_cpu_workers="${LMCACHE_MAX_CPU_WORKERS:-4}"
lmcache_log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"

server_args=(
  server
  --host "${lmcache_host}"
  --port "${lmcache_port}"
  --chunk-size "${lmcache_chunk_size}"
  --max-gpu-workers "${lmcache_gpu_workers}"
  --max-cpu-workers "${lmcache_cpu_workers}"
  --l1-size-gb "${lmcache_l1_gb}"
  --l1-init-size-gb "${lmcache_l1_init_gb}"
  --l1-write-ttl-seconds 600
  --l1-read-ttl-seconds 300
  --eviction-policy LRU
  --eviction-trigger-watermark 0.90
  --eviction-ratio 0.10
  --l2-store-policy default
  --l2-prefetch-policy retain
  --http-port "${lmcache_http_port}"
  --prometheus-port "${lmcache_prometheus_port}"
)

lmcache_l2_path=disabled
if [[ "${mode}" == "disk" ]]; then
  lmcache_l2_path="${LMCACHE_L2_PATH:-/cache/lmcache/${service_port}}"
  lmcache_l2_gb="${LMCACHE_L2_GB:-256}"
  lmcache_l2_workers="${LMCACHE_L2_WORKERS:-4}"
  mkdir -p "${lmcache_l2_path}"
  l2_config="$(
    LMCACHE_JSON_PATH="${lmcache_l2_path}" \
    LMCACHE_JSON_WORKERS="${lmcache_l2_workers}" \
    LMCACHE_JSON_CAPACITY_GB="${lmcache_l2_gb}" \
      python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "type": "fs_native",
            "base_path": os.environ["LMCACHE_JSON_PATH"],
            "num_workers": int(os.environ["LMCACHE_JSON_WORKERS"]),
            "use_odirect": False,
            "max_capacity_gb": float(os.environ["LMCACHE_JSON_CAPACITY_GB"]),
        },
        separators=(",", ":"),
    )
)
PY
  )"
  server_args+=(--l2-adapter "${l2_config}")
fi

transfer_config="$(
  LMCACHE_JSON_HOST="${lmcache_host}" \
  LMCACHE_JSON_PORT="${lmcache_port}" \
    python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": f"tcp://{os.environ['LMCACHE_JSON_HOST']}",
                "lmcache.mp.port": int(os.environ["LMCACHE_JSON_PORT"]),
                "lmcache.mp.mq_timeout": 60,
                "lmcache.mp.heartbeat_interval": 5,
            },
        },
        separators=(",", ":"),
    )
)
PY
)"

rm -f "${lmcache_log}"
export LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
lmcache "${server_args[@]}" >"${lmcache_log}" 2>&1 &
lmcache_pid=$!
model_pid=""

stop_children() {
  if [[ -n "${model_pid}" ]] && kill -0 "${model_pid}" 2>/dev/null; then
    kill -TERM "${model_pid}" 2>/dev/null || true
  fi
  if kill -0 "${lmcache_pid}" 2>/dev/null; then
    kill -TERM "${lmcache_pid}" 2>/dev/null || true
  fi
}
trap stop_children INT TERM HUP

ready=0
for _ in $(seq 1 "${LMCACHE_START_TIMEOUT:-120}"); do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    break
  fi
  if grep -Fq 'LMCache ZMQ cache server is running' "${lmcache_log}"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "ERROR: LMCache did not become ready; log follows" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${lmcache_pid}" 2>/dev/null || true
  exit 1
fi

printf 'LMCache ready: mode=%s L1=%sGB chunk=%s L2=%s log=%s\n' \
  "${mode}" "${lmcache_l1_gb}" "${lmcache_chunk_size}" \
  "${lmcache_l2_path}" "${lmcache_log}"

"$@" --kv-transfer-config "${transfer_config}" &
model_pid=$!

set +e
completed_pid=""
wait -n -p completed_pid "${lmcache_pid}" "${model_pid}"
first_status=$?
set -e

if [[ "${completed_pid}" == "${lmcache_pid}" ]]; then
  echo "ERROR: LMCache exited while the model server was running" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${model_pid}" 2>/dev/null || true
  exit 1
fi

stop_children
wait "${lmcache_pid}" 2>/dev/null || true
exit "${first_status}"
