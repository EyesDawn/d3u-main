export CUDA_DEVICE_ORDER=PCI_BUS_ID

# python tools/eval_svq_point_forecast.py --datasets ETTm1 weather ECL --gpu 0
python tools/eval_svq_point_forecast.py --plot-only
