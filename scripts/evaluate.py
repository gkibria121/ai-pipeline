# ============================================================================
# FILE: scripts/evaluate.py
# ============================================================================import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models import AASIST
from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from utils import calculate_tDCF_EER


def evaluate(model, dataloader, device, output_file):
    """Evaluate model and save scores"""
    model.eval()
    
    with open(output_file, 'w') as f_out:
        with torch.no_grad():
            for inputs, keys in tqdm(dataloader, desc='Evaluating'):
                inputs = inputs.to(device)
                _, outputs = model(inputs, freq_aug=False)
                
                # Get scores (difference between bonafide and spoof logits)
                scores = outputs[:, 1] - outputs[:, 0]
                
                for key, score in zip(keys, scores):
                    f_out.write(f'{key} {score.item():.6f}\n')
    
    print(f'Scores saved to: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate AASIST model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model weights (overrides config)')
    parser.add_argument('--eval_set', type=str, default='eval',
                       choices=['dev', 'eval'],
                       help='Which set to evaluate')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    model_path = args.model_path or config['model_path']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model and load weights
    model = AASIST(config['model_config']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Model loaded from: {model_path}')
    
    # Prepare data
    database_path = Path(config['database_path'])
    
    if args.eval_set == 'dev':
        protocol_file = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        data_path = 'ASVspoof2019_LA_dev'
        score_file = 'scores_dev.txt'
    else:
        protocol_file = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
        data_path = 'ASVspoof2019_LA_eval'
        score_file = config.get('eval_output', 'scores_eval.txt')
    
    file_list = genSpoof_list(database_path / protocol_file, is_eval=True)
    print(f'Number of evaluation samples: {len(file_list)}')
    
    # Create dataset and dataloader
    eval_set = Dataset_ASVspoof2019_devNeval(
        file_list,
        database_path / data_path,
        cut=config['model_config']['nb_samp']
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    evaluate(model, eval_loader, device, score_file)
    
    # Calculate metrics if not eval set
    if args.eval_set != 'eval' or config.get('eval_all_best', 'False') == 'True':
        asv_score_file = config.get('asv_score_path')
        if asv_score_file and Path(asv_score_file).exists():
            print('\nCalculating EER and min t-DCF...')
            eer, min_tDCF = calculate_tDCF_EER(
                score_file,
                asv_score_file,
                'evaluation_results.txt',
                printout=True
            )
            print(f'\nResults:')
            print(f'  EER: {eer:.4f}%')
            print(f'  min t-DCF: {min_tDCF:.6f}')
        else:
            print('ASV scores not found. Skipping metric calculation.')


if __name__ == '__main__':
    main()