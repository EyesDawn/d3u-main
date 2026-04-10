export CUDA_DEVICE_ORDER=PCI_BUS_ID

set -euo pipefail

python3 tools/eval_svq_point_forecast.py \
    --datasets weather \
    --feature-indices 18 \
    --window-indices 36 \
    --residual-space original \
    --no_clip \
    --plot-only

python3 tools/eval_svq_point_forecast.py \
    --datasets weather \
    --feature-indices 13 \
    --window-indices 21 \
    --residual-space original \
    --no_clip \
    --plot-only

python3 tools/eval_svq_point_forecast.py \
    --datasets weather \
    --feature-indices 20 \
    --window-indices 43 \
    --residual-space original \
    --no_clip \
    --plot-only
