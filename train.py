import os
import json
import torch
import random
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from lavis.models import load_model_and_preprocess
from statistics import mean, geometric_mean, harmonic_mean

from data_utils import (
    squarepad_transform, targetpad_transform,
    OACIRRDataset, CIRRDataset, FashionIQDataset
)
from utils import (
    custom_collate_fn, collate_fn, update_train_running_results_dict,
    set_train_bar_description_dict, extract_index_blip_features,
    save_model, generate_randomized_fiq_caption, device
)
from evaluate import (
    compute_oacirr_val_metrics, compute_oacirr_bounding_box_val_metrics,
    compute_cirr_val_metrics, compute_fiq_val_metrics
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def finetune_oacirr(train_variant: str, num_epochs: int, blip_model_name: str, blip_model_weight: str,
                    vit_backbone: str, learning_rate: float, batch_size: int, validation_frequency: int,
                    transform: str, save_training: bool, save_best: bool, save_memory: bool, **kwargs):
    """
        Unified Fine-tuning for the OACIRR Benchmark (Supports both Union and single Subsets).
        :param train_variant: OACIRR setting to train on. Should be in ['Union', 'Fashion', 'Car', 'Product', 'Landmark']
        :param num_epochs: number of fine-tuning epochs
        :param blip_model_name: fine-tuned model to use
        :param blip_model_weight: pre-trained model weight
        :param vit_backbone: BLIP-2 Q-Former vision encoder backbone
        :param learning_rate: fine-tuning leanring rate
        :param batch_size: fine-tuning batch size
        :param validation_frequency: validation frequency expressed in epoch
        :param transform: Image preprocess transform to use. Should be in ['squarepad', 'targetpad']
                          When targetpad is True, also required to provide 'target_ratio' as kwarg
        :param save_training: when True save the weights of the fine-tuned model
        :param save_best: when True save only the weights of the best model wrt the arithmetic_mean metric
        :param save_memory: Save extracted features on CPU while evaluating
        :param kwargs: if you use the 'targetpad' transform, you should prove 'target_ratio' as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = Path(kwargs['save_dir']) / f"OACIR_{train_variant}_finetune_{blip_model_name}_{training_start}"
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # Load Model
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=vit_backbone, is_eval=False, device=device)

    if blip_model_weight:
        blip_model_checkpoint = torch.load(blip_model_weight, map_location=device)
        msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
        print(f"Missing keys when loading weights: {msg.missing_keys}")

    # Setup Image Preprocess Transforms
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
        print(f"Target pad with target_ratio={kwargs['target_ratio']} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # Setup Datasets
    data_root = kwargs['data_root']

    is_visual_baseline = (kwargs.get('bounding_box_width', 0) > 0) or kwargs.get('bounding_box_crop', False)
    actual_highlight_training = False if is_visual_baseline else kwargs.get('highlight_training', False)

    ### Train Dataset
    relative_train_dataset = OACIRRDataset(
        data_root=data_root, variant=train_variant, split='train', mode='relative', preprocess=preprocess,
        highlight_training=actual_highlight_training, text_entity=kwargs['text_entity'],
        bounding_box_width=kwargs['bounding_box_width'], bounding_box_color=kwargs['bounding_box_color'],
        bounding_box_crop=kwargs.get('bounding_box_crop', False)
    )
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset, batch_size=batch_size, num_workers=kwargs['num_workers'],
        pin_memory=False, collate_fn=custom_collate_fn, drop_last=True, shuffle=True
    )

    ### Validation Datasets
    val_variants = ['Fashion', 'Car', 'Product', 'Landmark'] if train_variant == 'Union' else [train_variant]
    val_datasets = {}
    validation_log_frame = {}

    for variant in val_variants:
        relative_val_dataset = OACIRRDataset(data_root, variant, 'val', 'relative', preprocess, highlight_inference=kwargs['highlight_inference'], text_entity=kwargs['text_entity'])
        classic_val_dataset = OACIRRDataset(data_root, variant, 'val', 'classic', preprocess)

        classic_val_dataset_bounding_box = None
        if kwargs['bounding_box_width'] or kwargs['bounding_box_crop']:
            classic_val_dataset_bounding_box = OACIRRDataset(
                data_root, variant, 'val', 'classic', preprocess,
                bounding_box_width=kwargs['bounding_box_width'], bounding_box_color=kwargs['bounding_box_color'], bounding_box_crop=kwargs['bounding_box_crop']
            )
        val_datasets[variant] = (relative_val_dataset, classic_val_dataset, classic_val_dataset_bounding_box)
        validation_log_frame[variant] = pd.DataFrame()

    # Optimizer & Scheduler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_arithmetic = 0
    training_log_frame = pd.DataFrame()

    # Training Loop
    print(f'===== Training loop started =====')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=140, ascii=True)

        for idx, batch_data in enumerate(train_bar):
            # batch_data contains: (reference_images, target_images, reference_names, target_names, modification_texts, reference_bboxes)
            reference_images = batch_data[0].to(device, non_blocking=True)
            target_images = batch_data[1].to(device, non_blocking=True)
            modification_texts = [txt_processors["eval"](text) for text in batch_data[4]]
            reference_bboxes = batch_data[5]
            images_in_batch = reference_images.size(0)

            optimizer.zero_grad()
            blip_model.train()

            with torch.cuda.amp.autocast():
                loss_dict = blip_model({
                    "reference_image": reference_images,
                    "target_image": target_images,
                    "modification_text": modification_texts,
                    "reference_bbox": reference_bboxes if kwargs['highlight_training'] else None
                })

                loss = 0.
                for key in loss_dict.keys():
                    if key in kwargs:
                        loss += kwargs[key] * loss_dict[key]
                    else:
                        print(f"[Warning] Loss weight for '{key}' not found in kwargs. Defaulting to 1.0")
                        loss += 1.0 * loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Log training metrics
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])

        # Training CSV Logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # Validation Loop
        if epoch % validation_frequency == 0:
            blip_model.eval()

            # Iterate over validation subsets
            avg_arithmetics = []

            for variant in val_variants:
                print(f"--- Validating on OACIRR [{variant}] ---")
                relative_val_dataset, classic_val_dataset, classic_val_dataset_bounding_box = val_datasets[variant]

                val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)

                if kwargs['bounding_box_width'] or kwargs['bounding_box_crop']:
                    val_index_features_bbox, val_index_names_bbox = extract_index_blip_features(classic_val_dataset_bounding_box, blip_model, save_memory)
                    results = compute_oacirr_bounding_box_val_metrics(
                        relative_val_dataset, blip_model, val_index_features, val_index_features_bbox,
                        val_index_names, val_index_names_bbox, txt_processors, save_memory=save_memory
                    )
                else:
                    results = compute_oacirr_val_metrics(
                        relative_val_dataset, blip_model, val_index_features, val_index_names, 
                        txt_processors, highlight_inference=kwargs['highlight_inference'], save_memory=save_memory
                    )

                recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5 = results
                recall_results = [recall_at1, recall_at5, recall_at10, recall_at50]

                results_dict = {
                    'R_ID@1': class_recall_at1,
                    'R_ID@3': class_recall_at3,
                    'R_ID@5': class_recall_at5,
                    'R@1': recall_at1,
                    'R@5': recall_at5,
                    'R@10': recall_at10,
                    'R@50': recall_at50,
                    'arithmetic_mean': mean(recall_results),
                    'harmonic_mean': harmonic_mean(recall_results),
                    'geometric_mean': geometric_mean(recall_results),
                }

                print(json.dumps(results_dict, indent=4))
                avg_arithmetics.append(results_dict['arithmetic_mean'])

                # Validation CSV Logging
                log_dict = {'epoch': epoch, **results_dict}
                validation_log_frame[variant] = pd.concat([validation_log_frame[variant], pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame[variant].to_csv(str(training_path / f'{variant.lower()}_validation_metrics.csv'), index=False)

            # Checkpoint Saving
            overall_arithmetic = mean(avg_arithmetics)

            if save_training:
                if save_best and overall_arithmetic > best_arithmetic:
                    best_arithmetic = overall_arithmetic
                    save_model('adafocal_finetune_best', epoch, blip_model, training_path)
                elif not save_best:
                    if train_variant == 'Union' and (epoch >= 20) and (epoch % 2 == 0):
                        save_model(f'adafocal_finetune_union_epoch_{epoch}', epoch, blip_model, training_path)
                    elif train_variant != 'Union':
                        save_model(f'adafocal_finetune_{train_variant.lower()}_epoch_{epoch}', epoch, blip_model, training_path)


def finetune_fiq(train_dress_types: List[str], val_dress_types: List[str], num_epochs: int, blip_model_name: str,
                 blip_model_weight: str, vit_backbone: str, learning_rate: float, batch_size: int, validation_frequency: int,
                 transform: str, save_training: bool, save_best: bool, save_memory: bool, **kwargs):
    """
        Fine-tuning for the FashionIQ dataset.
        :param train_dress_types: FashionIQ categories to train on. Should be in ['dress', 'shirt', 'toptee']
        :param val_dress_types: FashionIQ categories to validate on. Should be in ['dress', 'shirt', 'toptee']
        :param num_epochs: number of fine-tuning epochs
        :param blip_model_name: fine-tuned model to use
        :param blip_model_weight: pre-trained model weight
        :param vit_backbone: BLIP-2 Q-Former vision encoder backbone
        :param learning_rate: fine-tuning leanring rate
        :param batch_size: fine-tuning batch size
        :param validation_frequency: validation frequency expressed in epoch
        :param transform: Image preprocess transform to use. Should be in ['squarepad', 'targetpad']
                          When targetpad is True, also required to provide 'target_ratio' as kwarg
        :param save_training: when True save the weights of the fine-tuned model
        :param save_best: when True save only the weights of the best model the average_recall metric
        :param save_memory: Save extracted features on CPU while evaluating
        :param kwargs: if you use the 'targetpad' transform, you should prove 'target_ratio' as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = Path(kwargs['save_dir']) / f"FIQ_finetune_{blip_model_name}_{training_start}"
    training_path.mkdir(exist_ok=False, parents=True)

    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=vit_backbone, is_eval=False, device=device)

    if blip_model_weight:
        blip_model_checkpoint = torch.load(blip_model_weight, map_location=device)
        msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
        print(f"Missing keys when loading weights: {msg.missing_keys}")

    # Setup Image Preprocess Transforms
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
        print(f"Target pad with target_ratio={kwargs['target_ratio']} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # Setup Datasets
    data_root = kwargs['data_root']
    idx_to_dress_mapping = {}
    relative_val_datasets, classic_val_datasets = [], []

    ### Validation Datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_datasets.append(FashionIQDataset(data_root, 'val',[dress_type], 'relative', preprocess))
        classic_val_datasets.append(FashionIQDataset(data_root, 'val', [dress_type], 'classic', preprocess))

    ### Train Dataset
    relative_train_dataset = FashionIQDataset(data_root, 'train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=kwargs['num_workers'],
                                       pin_memory=False, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Optimizer & Scheduler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_avg_recall = 0
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Training Loop
    print('===== Training loop started =====')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=140, ascii=True)

        for idx, (reference_images, target_images, modification_texts, reference_names, target_names) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            if len(modification_texts) == 2 and isinstance(modification_texts[0], (tuple, list)):
                modification_texts = list(zip(*modification_texts))

            flattened_texts = np.array(modification_texts).flatten().tolist()
            modification_texts = generate_randomized_fiq_caption(flattened_texts)
            modification_texts = [txt_processors["eval"](text) for text in modification_texts]

            optimizer.zero_grad()
            blip_model.train()

            with torch.cuda.amp.autocast():
                loss_dict = blip_model({
                    "reference_image": reference_images,
                    "target_image": target_images,
                    "modification_text": modification_texts
                })

                loss = 0.
                for key in loss_dict.keys():
                    if key in kwargs:
                        loss += kwargs[key] * loss_dict[key]
                    else:
                        print(f"[Warning] Loss weight for '{key}' not found in kwargs. Defaulting to 1.0")
                        loss += 1.0 * loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Log training metrics
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])

        # Training CSV Logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # Validation Loop
        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10, recalls_at50 = [], []

            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets, idx_to_dress_mapping):
                # Extract target image features for the current validation subset
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)

                # Compute retrieval metrics for the current validation subset
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model, index_features, index_names, txt_processors, save_memory)

                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)

                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]

            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))

            # Validation CSV Logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('baseline_finetune_best', epoch, blip_model, training_path)
                elif not save_best:
                    save_model(f'baseline_finetune_epoch_{epoch}', epoch, blip_model, training_path)


def finetune_cirr(num_epochs: int, blip_model_name: str, blip_model_weight: str, vit_backbone: str, learning_rate: float, batch_size: int,
                  validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool, **kwargs):
    """
        Fine-tuning for the CIRR dataset.
        :param num_epochs: number of fine-tuning epochs
        :param blip_model_name: fine-tuned model to use
        :param blip_model_weight: pre-trained model weight
        :param vit_backbone: BLIP-2 Q-Former vision encoder backbone
        :param learning_rate: fine-tuning leanring rate
        :param batch_size: fine-tuning batch size
        :param validation_frequency: validation frequency expressed in epoch
        :param transform: Image preprocess transform to use. Should be in ['squarepad', 'targetpad']
                          When targetpad is True, also required to provide 'target_ratio' as kwarg
        :param save_training: when True save the weights of the fine-tuned model
        :param save_best: when True save only the weights of the best model the average_recall metric
        :param save_memory: Save extracted features on CPU while evaluating
        :param kwargs: if you use the 'targetpad' transform, you should prove 'target_ratio' as kwarg
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = Path(kwargs['save_dir']) / f"CIRR_finetune_{blip_model_name}_{training_start}"
    training_path.mkdir(exist_ok=False, parents=True)

    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=vit_backbone, is_eval=False, device=device)

    if blip_model_weight:
        blip_model_checkpoint = torch.load(blip_model_weight, map_location=device)
        msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
        print(f"Missing keys when loading weights: {msg.missing_keys}")

    # Setup Image Preprocess Transforms
    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(kwargs['target_ratio'], input_dim)
        print(f"Target pad with target_ratio={kwargs['target_ratio']} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # Setup Datasets
    data_root = kwargs['data_root']

    ### Validation Datasets
    relative_val_dataset = CIRRDataset(data_root, 'val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset(data_root, 'val', 'classic', preprocess)

    ### Train Dataset
    relative_train_dataset = CIRRDataset(data_root, 'train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=kwargs['num_workers'],
                                       pin_memory=False, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Optimizer & Scheduler
    optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay': 0.05}])
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_arithmetic = 0
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Training Loop
    print('===== Training loop started =====')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=140, ascii=True)

        for idx, (reference_images, target_images, modification_texts, reference_names, target_names, group_images) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            modification_texts = [txt_processors["eval"](text) for text in modification_texts]

            optimizer.zero_grad()
            blip_model.train()

            with torch.cuda.amp.autocast():
                loss_dict = blip_model({
                    "reference_image": reference_images,
                    "target_image": target_images,
                    "modification_text": modification_texts
                })

                loss = 0.
                for key in loss_dict.keys():
                    if key in kwargs:
                        loss += kwargs[key] * loss_dict[key]
                    else:
                        print(f"[Warning] Loss weight for '{key}' not found in kwargs. Defaulting to 1.0")
                        loss += 1.0 * loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        # Log training metrics
        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(train_running_results[key] / train_running_results['images_in_epoch'])

        # Training CSV Logging
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        # Validation Loop
        if epoch % validation_frequency == 0:
            blip_model.eval()

            # Extract target image features
            val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)

            # Compute retrieval metrics
            results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features, val_index_names, txt_processors)
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }

            print(json.dumps(results_dict, indent=4))

            # Validation CSV Logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('baseline_finetune_best', epoch, blip_model, training_path)
                elif not save_best:
                    save_model(f'baseline_finetune_epoch_{epoch}', epoch, blip_model, training_path)



if __name__ == '__main__':

    parser = ArgumentParser("AdaFocal / Baseline Model Fine-tuning on OACIRR & Standard CIR Datasets")

    parser.add_argument("--dataset", type=str, required=True, choices=['Fashion', 'Car', 'Product', 'Landmark', 'Union', 'CIRR', 'FashionIQ'], help="Dataset to train on")
    parser.add_argument("--data-root", type=str, default="./Datasets/OACIRR", help="Root directory of the dataset")
    parser.add_argument("--blip-model-name", type=str, default="oacir_adafocal", help="Model registry name")
    parser.add_argument("--blip-model-weight", type=str, default="", help="Path to pre-trained model weight")

    parser.add_argument("--vit-backbone", type=str, default="pretrain", help="pretrain for Vit-G, pretrain_vitL for ViT-L")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, choices=['squarepad', 'targetpad'])

    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-epochs", default=50, type=int, help="Number of fine-tuning epochs")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size")
    parser.add_argument("--loss-align", default=1.0, type=float, help="Weight of Contrastive Alignment Loss")

    parser.add_argument("--highlight-training", action='store_true', help="Whether use region highlight strategy during training")
    parser.add_argument("--highlight-inference", action='store_true', help="Whether use region highlight strategy during inference")
    parser.add_argument("--text-entity", action='store_true', help="Add personalized entity prompt into the modification text")
    parser.add_argument("--bounding-box-width", default=0, type=int, help="Reference image bounding box width")
    parser.add_argument("--bounding-box-color", default="red", type=str, help="Reference image bounding box color")
    parser.add_argument("--bounding-box-crop", action='store_true', help="Crop bounding box region")

    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epoch")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-training", action='store_true', help="Whether save the fine-tuning model")
    parser.add_argument("--save-best", action='store_true', help="Save only the best model during fine-tuning")
    parser.add_argument("--save-memory", action='store_true', help="Save extracted features on cpu")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()


    if args.seed:
        set_seed(args.seed)


    training_hyper_params = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "blip_model_name": args.blip_model_name,
        "blip_model_weight": args.blip_model_weight,
        "vit_backbone": args.vit_backbone,
        "target_ratio": args.target_ratio,
        "transform": args.transform,
        "num_workers": args.num_workers,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "loss_align": args.loss_align,
        "highlight_training": args.highlight_training,
        "highlight_inference": args.highlight_inference,
        "text_entity": args.text_entity,
        "bounding_box_width": args.bounding_box_width,
        "bounding_box_color": args.bounding_box_color,
        "bounding_box_crop": args.bounding_box_crop,
        "validation_frequency": args.validation_frequency,
        "save_dir": args.save_dir,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "save_memory": args.save_memory,
        "seed": args.seed
    }


    if args.dataset in ['Fashion', 'Car', 'Product', 'Landmark', 'Union']:
        finetune_oacirr(train_variant=args.dataset, **training_hyper_params)


    elif args.dataset == 'CIRR':
        finetune_cirr(**training_hyper_params)


    elif args.dataset == 'FashionIQ':
        training_hyper_params.update({'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types':['dress', 'toptee', 'shirt']})
        finetune_fiq(**training_hyper_params)


    else:
        raise ValueError("Dataset should be in ['Fashion', 'Car', 'Product', 'Landmark', 'Union', 'CIRR', 'FashionIQ']")
