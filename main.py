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
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, FEATURE_TYPES)
from dataset_factory import get_dataset_info, create_dataset_loaders, DATASET_TYPES
from evaluation import calculate_tDCF_EER, calculate_simple_eer_accuracy
from metrics import MetricsTracker, save_all_metrics
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from feature_analysis import analyze_and_visualize_features

warnings.filterwarnings("ignore", category=FutureWarning)


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
    
    # Override num_epochs if provided via command line
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    
    # Override batch_size if provided via command line
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    
    # Set dataset type
    dataset_type = args.dataset if args.dataset is not None else config.get("dataset_type", 1)
    dataset_info = get_dataset_info(dataset_type)
    
    # Display dataset information
    print("\n" + "="*70)
    print(f"DATASET: {dataset_info['name']} (Type {dataset_type})")
    print("="*70)
    
    optim_config["epochs"] = config["num_epochs"]
    track = config.get("track", dataset_info.get("track", "LA"))
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
    
    # Override model_path if eval_model_weights is provided via command line
    if args.eval_model_weights is not None:
        config["model_path"] = args.eval_model_weights
    
    # Set feature_type in config
    if args.feature_type is not None:
        config["feature_type"] = args.feature_type
    elif "feature_type" not in config:
        config["feature_type"] = 0  # Default to raw waveform
    
    # Set random_noise in config
    if args.random_noise:
        config["random_noise"] = True
        print("Random augmentation enabled for training")
    else:
        config["random_noise"] = False

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    
    # Use dataset-specific path or config path
    if dataset_type == 1:
        database_path = Path(config.get("database_path", dataset_info["base_path"]))
        prefix_2019 = "ASVspoof2019.{}".format(track)
    else:
        database_path = Path(dataset_info["base_path"])
        prefix_2019 = None
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    feature_type = config.get("feature_type", 0)
    feature_name = FEATURE_TYPES.get(feature_type, f"feat{feature_type}")
    dataset_name = DATASET_TYPES.get(dataset_type, f"DS{dataset_type}")
    model_tag = "{}_{}_{}_ep{}_bs{}_feat{}".format(
        dataset_name.replace("-", ""),
        track if track else "audio",
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"], feature_type)
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")
    
    # Generate feature analysis visualization
    print(f"\n{'='*50}")
    print(f"Feature Type: {feature_type} ({feature_name.upper()})")
    print(f"Generating feature analysis visualization...")
    print(f"{'='*50}")
    try:
        # Get a sample audio file for visualization
        sample_audio = None
        trn_database_path = database_path / "ASVspoof2019_{}_train/flac".format(track)
        if trn_database_path.exists():
            audio_files = list(trn_database_path.glob("*.flac"))
            if audio_files:
                sample_audio = str(audio_files[0])
        
        if sample_audio:
            feature_viz_dir = model_tag / "feature_analysis"
            analyze_and_visualize_features(
                audio_file=sample_audio,
                feature_type=feature_type,
                save_dir=feature_viz_dir,
                sr=16000
            )
        else:
            print("⚠️  No sample audio found for feature visualization")
    except Exception as e:
        print(f"⚠️  Could not generate feature analysis: {e}")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    if dataset_type == 1:
        trn_loader, dev_loader, eval_loader = get_loader(
            database_path, args.seed, config)
    else:
        trn_loader, dev_loader, eval_loader = create_dataset_loaders(
            dataset_type, database_path, feature_type, 
            config.get("random_noise", False), config["batch_size"], args.seed)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        if dataset_type == 1:
            produce_evaluation_file(eval_loader, model, device,
                                    eval_score_path, eval_trial_path)
        else:
            produce_evaluation_file_simple(eval_loader, model, device,
                                          eval_score_path)
        evaluate_model(dataset_type, eval_score_path, database_path, config,
                      model_tag / "evaluation_results.txt")
        print("DONE.")
        eval_eer, eval_tdcf, eval_acc = evaluate_model(
            dataset_type, eval_score_path, database_path, config,
            model_tag/"loaded_model_results.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

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
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(metric_path)
    
    total_epochs = config["num_epochs"]
    # Training
    for epoch in range(config["num_epochs"]):
        epoch = epoch + 1
        print("\n" + "="*50)
        print(f"Start training epoch {epoch}/{total_epochs}")
        print("="*50)
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        print("\nValidating on development set...")
        if dataset_type == 1:
            produce_evaluation_file(dev_loader, model, device,
                                    metric_path/"dev_score.txt", dev_trial_path)
        else:
            produce_evaluation_file_simple(dev_loader, model, device,
                                          metric_path/"dev_score.txt")
        dev_eer, dev_tdcf, dev_acc = evaluate_model(
            dataset_type, metric_path/"dev_score.txt", database_path, config,
            metric_path/"dev_results_{:03d}epo.txt".format(epoch))
        
        # Handle None t-DCF for non-ASVspoof datasets
        if dev_tdcf is None:
            dev_tdcf = 0.0
            print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_acc: {:.2f}%".format(
                running_loss, dev_eer, dev_acc))
        else:
            print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}, dev_acc: {:.2f}%".format(
                running_loss, dev_eer, dev_tdcf, dev_acc))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        
        # Initialize eval metrics
        current_eval_eer = None
        current_eval_tdcf = None
        current_eval_acc = None
        
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                if dataset_type == 1:
                    produce_evaluation_file(eval_loader, model, device,
                                            eval_score_path, eval_trial_path)
                else:
                    produce_evaluation_file_simple(eval_loader, model, device,
                                                  eval_score_path)
                eval_eer, eval_tdcf, eval_acc = evaluate_model(
                    dataset_type, eval_score_path, database_path, config,
                    metric_path / "eval_results_{:03d}epo.txt".format(epoch))
                
                # Handle None t-DCF
                if eval_tdcf is None:
                    eval_tdcf = 0.0
                
                # Store current eval metrics for tracking
                current_eval_eer = eval_eer
                current_eval_tdcf = eval_tdcf
                current_eval_acc = eval_acc

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
        
        # Log epoch metrics to tracker
        metrics_tracker.add_epoch(epoch, running_loss, dev_eer, dev_tdcf, dev_acc,
                                 current_eval_eer, current_eval_tdcf, current_eval_acc,
                                 best_dev_eer, best_dev_tdcf)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    if dataset_type == 1:
        produce_evaluation_file(eval_loader, model, device, eval_score_path,
                                eval_trial_path)
    else:
        produce_evaluation_file_simple(eval_loader, model, device, eval_score_path)
    eval_eer, eval_tdcf, eval_acc = evaluate_model(
        dataset_type, eval_score_path, database_path, config,
        model_tag / "final_evaluation_results.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    if eval_tdcf is not None:
        f_log.write("EER: {:.3f}, min t-DCF: {:.5f}, Accuracy: {:.2f}%".format(eval_eer, eval_tdcf, eval_acc))
    else:
        f_log.write("EER: {:.3f}, Accuracy: {:.2f}%".format(eval_eer, eval_acc))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        best_eval_acc = eval_acc
    if eval_tdcf is not None and eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    elif eval_tdcf is None:
        # For datasets without t-DCF, save based on best EER
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    
    # Save all metrics and generate visualizations
    save_all_metrics(metrics_tracker.get_metrics(), best_eval_eer, best_eval_tdcf, 
                     metric_path, config)
    
    if best_eval_tdcf is not None:
        print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}, Accuracy: {:.2f}%".format(
            best_eval_eer, best_eval_tdcf, best_eval_acc))
    else:
        print("Exp FIN. EER: {:.3f}, Accuracy: {:.2f}%".format(best_eval_eer, best_eval_acc))


def evaluate_model(dataset_type, cm_scores_file, database_path, config, output_file):
    """
    Evaluate model using appropriate metrics based on dataset type.
    
    Returns:
        eer, metric2, accuracy (metric2 is t-DCF for ASVspoof, None for others)
    """
    if dataset_type == 1:  # ASVspoof2019
        return calculate_tDCF_EER(
            cm_scores_file=cm_scores_file,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=output_file
        )
    else:  # Fake-or-Real or other simple datasets
        eer, acc = calculate_simple_eer_accuracy(
            cm_scores_file=cm_scores_file,
            output_file=output_file
        )
        return eer, None, acc  # No t-DCF for these datasets


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
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    feature_type = config.get("feature_type", 0)
    random_noise = config.get("random_noise", False)
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

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

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,
                                           feature_type=feature_type,
                                           random_noise=random_noise)
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

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path,
                                            feature_type=feature_type)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path,
                                             feature_type=feature_type)
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
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file (ASVspoof format)"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    with tqdm(total=len(data_loader), desc="Evaluation", unit="batch") as pbar:
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            pbar.update(1)

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def produce_evaluation_file_simple(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file (simple format without protocols)"""
    model.eval()
    fname_list = []
    score_list = []
    with tqdm(total=len(data_loader), desc="Evaluation", unit="batch") as pbar:
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            pbar.update(1)

    # Determine labels from file paths (real=1, fake=0)
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            # Extract label from path: real folder = bonafide, fake folder = spoof
            label = "bonafide" if "/real/" in fn else "spoof"
            fh.write("{} {} {}\n".format(fn, label, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with tqdm(total=len(trn_loader), desc="Training", unit="batch") as pbar:
        for batch_x, batch_y in trn_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            ii += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            batch_loss = criterion(batch_out, batch_y)
            running_loss += batch_loss.item() * batch_size
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            elif scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(scheduler))
            
            # Update progress bar with current loss
            current_loss = running_loss / num_total
            pbar.update(1)
            pbar.set_postfix({"loss": f"{current_loss:.5f}"})

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
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
    parser.add_argument("--dataset",
                        type=int,
                        default=None,
                        choices=[1, 2, 3],
                        help="dataset to use: 1=ASVspoof2019, 2=Fake-or-Real, 3=SceneFake (default: None, uses config value or 1)")
    parser.add_argument("--epochs",
                        type=int,
                        default=None,
                        help="number of epochs to override config (default: None, uses config value)")
    parser.add_argument("--batch_size",
                        type=int,
                        default=None,
                        help="batch size to override config (default: None, uses config value)")
    parser.add_argument("--feature_type",
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3],
                        help="feature type: 0=raw, 1=mel_spectrogram, 2=lfcc, 3=mfcc, 4=cqt (default: None, uses config value)")
    parser.add_argument("--random_noise",
                        action="store_true",
                        help="enable random data augmentation (gaussian noise, background noise, reverberation, pitch shift)")
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
                        help="path to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
