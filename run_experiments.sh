#!/usr/bin/env bash
set -e

# Run optionally: ./run_experiments.sh install  -> installs requirements
#                ./run_experiments.sh realtime -> runs only realtime demo
#                ./run_experiments.sh full     -> runs full experiment sequence (long)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$1" = "install" ]; then
  echo "Installing requirements..."
  python -m pip install -r requirements.txt
  exit 0
fi

if [ "$1" = "realtime" ] || [ -z "$1" ]; then
  echo "Running realtime.py (unbuffered)..."
  python -u realtime.py
  exit 0
fi

if [ "$1" = "full" ]; then
  echo "Running full experiment sequence (this may take a long time)..."
  python -u main.py --config config/LCNN.conf --feature_type 1 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best  --data_subset 0.01
  python -u main.py --config config/LCNN_Large.conf --feature_type 1 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/SEResNet.conf --feature_type 1 --dataset 2 --epochs 15 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/EfficientNetB2_Attention.conf --feature_type 1 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/RawNet3.conf --feature_type 0 --dataset 2 --epochs 15 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/AASIST.conf --feature_type 0 --dataset 2 --epochs 30 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/LCNN.conf --feature_type 4 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/LCNN.conf --feature_type 2 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/EfficientNetB2_Attention.conf --feature_type 4 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/SEResNet.conf --feature_type 3 --dataset 2 --epochs 15 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/SimpleCNN.conf --feature_type 1 --dataset 2 --epochs 25 --random_noise --weight_avg --eval_best --data_subset 0.01
  python -u main.py --config config/LCNN_Large.conf --feature_type 2 --dataset 2 --epochs 20 --random_noise --weight_avg --eval_best --data_subset 0.01
  python visualize_results.py --path "exp_result/*/metrics" --compare --show-summary --output ./comparison_plots
  exit 0
fi

echo "Unknown option: $1"
exit 2
