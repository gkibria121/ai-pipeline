"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

# Feature extraction options
FEATURE_TYPES = {
    0: "raw_audio",
    1: "mel_spectrogram",
    2: "mfcc",
    3: "lfcc",
    4: "cqt"
}


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # Set feature type
    feature_type = args.feature
    config["feature_type"] = feature_type
    
    # Display feature info
    feature_name = FEATURE_TYPES.get(feature_type, 'unknown')
    print(f"{'='*60}")
    print(f"Feature Type: {feature_type} ({feature_name})")
    print(f"{'='*60}")
    
    if args.eval:
        print(f"⚠️  IMPORTANT: Make sure this matches training feature type!")
        print(f"   Training feature: {feature_name}")

    # Override model path with command-line argument if provided
    if args.eval_model_weights is not None:
        config["model_path"] = args.eval_model_weights
        print(f"Using model weights from command-line: {args.eval_model_weights}")

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}_feat{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], 
        config["batch_size"],
        feature_type)
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    use_cuda = torch.cuda.is_available() and not getattr(args, "cpu", False)
    device = "cuda" if use_cuda else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # enable cudnn benchmark to use fast convolution algorithms (optional)
    if str(config.get("cudnn_benchmark_toggle", "True")).lower() in ("true", "1", "yes"):
        torch.backends.cudnn.benchmark = True

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders (pass optional features_path to load cached features)
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config, features_path=getattr(args, "features_path", None))

    # evaluates pretrained model and exit script
    if args.eval:
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
        
        model.load_state_dict(
            torch.load(model_path, map_location=device))
        print("Model loaded : {}".format(model_path))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path, config)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt",
                           output_dir=model_tag / "plots")
        print("DONE.")
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag/"loaded_model_t-DCF_EER.txt",
            output_dir=model_tag / "plots")
        print(f"Evaluation complete. EER: {eval_eer:.4f}%, t-DCF: {eval_tdcf:.5f}")
        sys.exit(0)

    # create loss (keep as 'criterion')
    criterion = nn.CrossEntropyLoss()  # or your configured loss

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, lr_scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # remove mid-training eval: evaluate only after all epochs
    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training loop (DO NOT run eval on eval set when a new best dev model is found)
    for epoch in range(config["num_epochs"]):
        print(f"Start training epoch {epoch+1}/{config['num_epochs']}")
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   criterion)

        # always evaluate on DEV to track progress and save best dev checkpoints
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path, config)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        # check improvement on dev and save best dev checkpoint (no eval set run here)
        if dev_eer < best_dev_eer:
            print("best dev model found at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
            # SWA snapshot on improvement (optional)
            optimizer_swa.update_swa()
            n_swa_update += 1

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    # Final evaluation with plots (run EVAL only once after all epochs)
    print("Start final evaluation")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path, config)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt",
                                             output_dir=model_tag / "plots")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
    database_path: str,
    seed: int,
    config: dict,
    features_path: Union[str, Path, None] = None) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    # Updated paths to include /flac/ subdirectory
    trn_database_path = database_path / "ASVspoof2019_{}_train/flac/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/flac/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/flac/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    # Explicit cache directories for precomputed features
    feat_type = config.get("feature_type", 0)

    # If user provided `features_path`, use it as the root. It may point to
    # either the features root or directly to a feat{n} folder. Otherwise,
    # default to <database_path>/features
    if features_path is not None:
        fp = Path(features_path)
        if fp.name.startswith("feat"):
            cache_base = fp
        else:
            cache_base = fp / f"feat{feat_type}"
    else:
        cache_base = Path(database_path) / "features" / f"feat{feat_type}"

    train_cache = cache_base / "train"
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,
                                           feature_type=feat_type,
                                           cache_dir=train_cache)

    # Print resolved cache paths for sanity checking
    dev_cache = cache_base / "dev"
    eval_cache = cache_base / "eval"
    print("Using feature cache paths:")
    print(f"  train -> {train_cache}")
    print(f"  dev   -> {dev_cache}")
    print(f"  eval  -> {eval_cache}")

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,
                                           feature_type=feat_type,
                                           cache_dir=train_cache)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))
    dev_cache = cache_base / "dev"
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path,
                                            feature_type=feat_type,
                                            cache_dir=dev_cache)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_cache = cache_base / "eval"
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path,
                                             feature_type=feat_type,
                                             cache_dir=eval_cache)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    config: dict) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader, desc="Evaluating"):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(train_loader, model, optimizer, device, criterion, scaler=None, amp_enabled=False, disable_tqdm=False):
    model.train()
    running_loss = 0.0

    it = train_loader
    if not disable_tqdm:
        it = tqdm(train_loader, desc="Training", leave=False)

    for batch_x, batch_y in it:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        if amp_enabled:
            with torch.cuda.amp.autocast():
                _, outputs = model(batch_x, Freq_aug=True)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, outputs = model(batch_x, Freq_aug=True)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    return running_loss / len(train_loader.dataset)


def save_checkpoint(model, config, path):
    """Save model with metadata"""
    checkpoint = {
        'model_state': model.state_dict(),
        'feature_type': config.get('feature_type', 0),
        'config': config
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    """Load model with metadata"""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint['model_state'], checkpoint.get('feature_type', 0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--feature",
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="feature type: 0=raw_audio, 1=mel_spectrogram, 2=mfcc, 3=lfcc, 4=cqt (default: 0)")
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bars to reduce IO overhead')
    parser.add_argument("--features_path", type=str, default=None, help="Optional path to precomputed features (root or feat{n} folder)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate feature type requirement for evaluation
    if args.eval and args.feature is None:
        parser.error("--feature is required when using --eval flag\n"
                    "Feature types: 0=raw_audio, 1=mel_spectrogram, 2=mfcc, 3=lfcc, 4=cqt")
    
    main(args)
