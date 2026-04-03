import json
import time
from argparse import ArgumentParser
from operator import itemgetter
from typing import List, Tuple
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from lavis.models import load_model_and_preprocess
from statistics import mean, geometric_mean, harmonic_mean
from data_utils import OACIRRDataset, FashionIQDataset, CIRRDataset, squarepad_transform, targetpad_transform
from utils import extract_index_blip_features, collate_fn, custom_collate_fn, device


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_model, index_features: torch.tensor, index_names: List[str],
                            txt_processors, save_memory=False) -> Tuple[float, float]:
    """
        Compute validation metrics on FashionIQ dataset
        :param relative_val_dataset: FashionIQ validation dataset in relative mode
        :param blip_model: BLIP model
        :param index_features: validation index features
        :param index_names: validation index names
        :return: the computed validation metrics
    """

    # Generate predictions
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors, save_memory)

    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics...")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model, relative_val_dataset: FashionIQDataset, index_names: List[str], index_features: torch.tensor,
                                 txt_processors, save_memory=False) -> Tuple[torch.tensor, List[str]]:
    """
        Compute FashionIQ predictions on the validation set
        :param blip_model: BLIP model
        :param relative_val_dataset: FashionIQ validation dataset in relative mode
        :param index_names: validation index names
        :param index_features: validation index features
        :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn, shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[-1]))

    # Initialize predicted features and target names
    target_names = []
    reference_names_all = []
    distance = []
    captions_all = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader, ncols=140, ascii=True):  # Load data

        # Concatenate the captions in a deterministic way
        if len(captions) == 2 and isinstance(captions[0], (tuple, list)):
            captions = list(zip(*captions))
        input_captions = [f"{texts[0].strip('.?, ').capitalize()} and {texts[1].strip('.?, ')}" for texts in captions]
        input_captions = [txt_processors["eval"](caption) for caption in input_captions]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with a single tensor
            if len(input_captions) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(name_to_feat))
                # To avoid unnecessary computation retrieve the reference image features directly from the index features
            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(blip_model.device)
                reference_image_features = reference_image_features.to(blip_model.device)
            batch_distance = blip_model.inference(reference_image_features, feature_curr, input_captions)
            distance.append(batch_distance)
            captions_all += input_captions

        target_names.extend(batch_target_names)
        reference_names_all.extend(reference_names)

    distance = torch.vstack(distance)

    return distance, target_names, reference_names_all, captions_all


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, blip_model, index_features: torch.tensor, index_names: List[str],
                             txt_processors) -> Tuple[float, float, float, float, float, float, float]:
    """
        Compute validation metrics on CIRR dataset
        :param relative_val_dataset: CIRR validation dataset in relative mode
        :param blip_model: BLIP model
        :param index_features: validation index features
        :param index_names: validation index names
        :return: the computed validation metrics
    """
    # Generate predictions
    pred_sim, reference_names, target_names, group_members, captions_all = generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors)

    print("Computing CIRR validation metrics...")
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    sorted_index_names_group = sorted_index_names[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model, relative_val_dataset: CIRRDataset, index_names: List[str], index_features: torch.tensor,
                                  txt_processors) -> Tuple[torch.tensor, List[str], List[str], List[List[str]], List[str]]:
    """
        Compute CIRR predictions on the validation set
        :param blip_model: BLIP model
        :param relative_val_dataset: CIRR validation dataset in relative mode
        :param index_names: validation index names
        :param index_features: validation index features
        :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize predicted features, target_names, group_members and reference_names
    distance = []
    target_names = []
    group_members = []
    reference_names = []
    captions_all = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(relative_val_loader, ncols=140, ascii=True):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with a single tensor
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
                # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions

        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    distance = torch.vstack(distance)

    return distance, reference_names, target_names, group_members, captions_all


def compute_class_recall(sorted_index_names: np.ndarray, target_names: List[str], name_to_relpath: dict) -> Tuple[float, float, float]:
    """
        Compute instance/class-level recall for OACIRR dataset. 
    """
    name_to_folder = {name: path.split('/')[-2] for name, path in name_to_relpath.items()}
    target_folders = np.array([name_to_folder.get(name) for name in target_names])
    folder_mapper = np.vectorize(name_to_folder.get)
    sorted_index_folders = folder_mapper(sorted_index_names)

    class_labels = (sorted_index_folders == target_folders[:, np.newaxis]).astype(int)
    modified_class_labels = ((class_labels == 1) & (np.cumsum(class_labels, axis=1) == 1)).astype(int)

    # Compute the metrics (R_ID@K)
    class_recall_at1 = (np.sum(modified_class_labels[:, :1]) / len(target_names)) * 100
    class_recall_at3 = (np.sum(modified_class_labels[:, :3]) / len(target_names)) * 100
    class_recall_at5 = (np.sum(modified_class_labels[:, :5]) / len(target_names)) * 100

    return class_recall_at1, class_recall_at3, class_recall_at5


def compute_oacirr_val_metrics(relative_val_dataset: OACIRRDataset, blip_model, index_features, index_names: List[str], txt_processors,
                               highlight_inference=False, save_results=False, save_memory=False):
    """
        Unified function to compute validation metrics on OACIRR dataset.
    """
    # Generate predictions
    pred_sim, reference_names, target_names, modification_texts, reference_bboxes, activation_scalars = \
        generate_oacirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors, highlight_inference, save_memory)
    # pred_sim, reference_names, target_names, modification_texts, reference_bboxes, activation_scalars, group_members = \
    #     generate_oacirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors, highlight_inference, save_memory)

    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation metrics...")
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # group_members = np.array(group_members)
    # group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    # group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    # sorted_index_names_group = sorted_index_names[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    # assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    class_recall_metrics = compute_class_recall(sorted_index_names, target_names, relative_val_dataset.name_to_relpath)
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    # group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    # group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    # group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    class_recall_at1, class_recall_at3, class_recall_at5 = class_recall_metrics
    metrics = (recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5)

    if save_results:
        return *metrics, reference_names, modification_texts, target_names, sorted_index_names, reference_bboxes, activation_scalars
        # return *metrics, group_recall_at1, group_recall_at2, group_recall_at3, reference_names, modification_texts, target_names, sorted_index_names, reference_bboxes, activation_scalars
    return metrics
    # return *metrics, group_recall_at1, group_recall_at2, group_recall_at3


def generate_oacirr_val_predictions(blip_model, relative_val_dataset: OACIRRDataset, index_names: List[str], index_features, txt_processors,
                                    highlight_inference=False, save_memory=False):
    """
        Unified function to compute predictions for the OACIRR validation sets (Fashion, Car, Product, Landmark).
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation predictions...")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=custom_collate_fn)

    name_to_feat = dict(zip(index_names, index_features[1]))

    distance, scalar = [], []
    reference_names, target_names = [], []
    captions_all, reference_bboxes_all = [], []

    for batch_data in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_names = batch_data[0]
        batch_target_names = batch_data[1]
        captions = batch_data[2]

        if len(batch_data) == 5:
            # batch_group_members = np.array(batch_data[3]).tolist()
            batch_reference_bbox = batch_data[4]
        else:
            batch_reference_bbox = batch_data[3]

        captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(blip_model.device)
                reference_image_features = reference_image_features.to(blip_model.device)

            reference_bbox = batch_reference_bbox if highlight_inference else None

            model_output = blip_model.inference(reference_image_features, feature_curr, captions, reference_bbox)

            if isinstance(model_output, tuple):
                batch_distance, activation_scalar = model_output
                scalar.append(activation_scalar.cpu())
            else:
                batch_distance = model_output
                scalar.append(torch.zeros(reference_image_features.shape[0], 1))

            distance.append(batch_distance.cpu())
            captions_all += captions

        reference_names.extend(batch_reference_names)
        target_names.extend(batch_target_names)
        reference_bboxes_all.extend(batch_reference_bbox)
        # group_members.extend(batch_group_members)

    return torch.vstack(distance), reference_names, target_names, captions_all, reference_bboxes_all, torch.vstack(scalar)
    # return torch.vstack(distance), reference_names, target_names, captions_all, reference_bboxes_all, torch.vstack(scalar), group_members


def compute_oacirr_bounding_box_val_metrics(relative_val_dataset: OACIRRDataset, blip_model, index_features, index_features_bounding_box, index_names: List[str],
                                            index_names_bounding_box: List[str], txt_processors, save_results=False, save_memory=False):
    """
        Unified function to compute validation metrics on OACIRR dataset with bounding box predictions.
    """
    # Generate predictions
    pred_sim, reference_names, target_names, modification_texts = \
        generate_oacirr_bounding_box_val_predictions(blip_model, relative_val_dataset, index_names_bounding_box, index_features, index_features_bounding_box, txt_processors, save_memory)
    # pred_sim, reference_names, target_names, modification_texts, group_members = \
    #     generate_oacirr_bounding_box_val_predictions(blip_model, relative_val_dataset, index_names_bounding_box, index_features, index_features_bounding_box, txt_processors, save_memory)

    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation metrics...")
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # group_members = np.array(group_members)
    # group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    # group_labels = labels[group_mask].reshape(labels.shape[0], -1)
    # sorted_index_names_group = sorted_index_names[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    # assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    class_recall_metrics = compute_class_recall(sorted_index_names, target_names, relative_val_dataset.name_to_relpath)
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    # group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    # group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    # group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    class_recall_at1, class_recall_at3, class_recall_at5 = class_recall_metrics
    metrics = (recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5)

    if save_results:
        return *metrics, reference_names, modification_texts, target_names, sorted_index_names
        # return *metrics, group_recall_at1, group_recall_at2, group_recall_at3, reference_names, modification_texts, target_names, sorted_index_names
    return metrics
    # return *metrics, group_recall_at1, group_recall_at2, group_recall_at3


def generate_oacirr_bounding_box_val_predictions(blip_model, relative_val_dataset: OACIRRDataset, index_names_bounding_box: List[str], index_features,
                                                 index_features_bounding_box, txt_processors, save_memory=False):
    """
        Unified function to compute predictions for the OACIRR validation sets (Fashion, Car, Product, Landmark) with bounding box predictions.
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation predictions...")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=custom_collate_fn)

    name_to_feat = dict(zip(index_names_bounding_box, index_features_bounding_box[1]))

    distance = []
    reference_names, target_names, captions_all = [], [], []

    for batch_data in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_names = batch_data[0]
        batch_target_names = batch_data[1]
        captions = batch_data[2]
        captions = [txt_processors["eval"](caption) for caption in captions]

        # if len(batch_data) == 5:
            # batch_group_members = np.array(batch_data[3]).tolist()

        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(blip_model.device)
                reference_image_features = reference_image_features.to(blip_model.device)

            batch_distance = blip_model.inference(reference_image_features, feature_curr, captions)
            distance.append(batch_distance.cpu())
            captions_all += captions

        reference_names.extend(batch_reference_names)
        target_names.extend(batch_target_names)
        # group_members.extend(batch_group_members)

    return torch.vstack(distance), reference_names, target_names, captions_all
    # return torch.vstack(distance), reference_names, target_names, captions_all, group_members



if __name__ == '__main__':

    parser = ArgumentParser("Evaluate the model on the OACIRR / Standard CIR Benchmark")

    parser.add_argument("--dataset", type=str, required=True, choices=['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ'], help="Dataset to evaluate on")
    parser.add_argument("--data-root", type=str, default="./Datasets/OACIRR", help="Root directory of the dataset")
    parser.add_argument("--blip-model-name", type=str, default="oacir_adafocal", help="Model registry name")
    parser.add_argument("--blip-model-weight", type=str, required=True, help="Path to the pre-trained model weight")

    parser.add_argument("--vit-backbone", type=str, default="pretrain", help="pretrain for Vit-G, pretrain_vitL for ViT-L")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, choices=['squarepad', 'targetpad'])

    parser.add_argument("--highlight-inference", dest="highlight_inference", action='store_true',
                         help="Whether use region highlight strategy during inference")
    parser.add_argument("--text-entity", dest="text_entity", action='store_true',
                         help="Add personalized entity prompt into the modification text")
    parser.add_argument("--bounding-box-width", default=0, type=int, help="Reference image bounding box width")
    parser.add_argument("--bounding-box-color", default="red", type=str, help="Reference image bounding box color")
    parser.add_argument("--bounding-box-crop",  dest="bounding_box_crop", action='store_true',
                         help="Crop bounding box region (ROI-Crop Baseline)")

    parser.add_argument("--save-results", dest="save_results", action='store_true',
                         help="Whether to save the validation results JSON")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true', help="Save extracted features on cpu")

    args = parser.parse_args()


    # ==================== Load Model ====================
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.vit_backbone, is_eval=False, device=device)

    blip_model_checkpoint = torch.load(args.blip_model_weight, map_location=device)
    blip_msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
    print(f"Missing keys when loading weights: {blip_msg.missing_keys}")

    # ==================== Transforms ====================
    input_dim = 224

    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
        print(f"Target pad with target_ratio={args.target_ratio} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")


    # ==================== OACIRR Evaluation ====================
    if args.dataset in ['Fashion', 'Car', 'Product', 'Landmark']:

        # Initialize the unified OACIRR Dataset
        relative_val_dataset = OACIRRDataset(
            data_root=args.data_root, variant=args.dataset, split='val', mode='relative', preprocess=preprocess,
            highlight_inference=args.highlight_inference, text_entity=args.text_entity
        )
        classic_val_dataset = OACIRRDataset(data_root=args.data_root, variant=args.dataset, split='val', mode='classic', preprocess=preprocess)

        if args.bounding_box_width:
            classic_val_dataset_bounding_box = OACIRRDataset(data_root=args.data_root, variant=args.dataset, split='val', mode='classic', preprocess=preprocess, bounding_box_width=args.bounding_box_width, bounding_box_color=args.bounding_box_color)
        elif args.bounding_box_crop:
            classic_val_dataset_bounding_box = OACIRRDataset(data_root=args.data_root, variant=args.dataset, split='val', mode='classic', preprocess=preprocess, bounding_box_crop=True)

        # Extract Gallery Image Features
        val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model, args.save_memory)

        if args.bounding_box_width or args.bounding_box_crop:
            val_index_features_bounding_box, val_index_names_bounding_box = extract_index_blip_features(classic_val_dataset_bounding_box, blip_model, args.save_memory)

            start_time = time.perf_counter()
            results = compute_oacirr_bounding_box_val_metrics(
                relative_val_dataset, blip_model, val_index_features, val_index_features_bounding_box,
                val_index_names, val_index_names_bounding_box, txt_processors, args.save_results, args.save_memory
            )
            duration = time.perf_counter() - start_time

            if args.save_results:
                # recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, group_recall_at1, group_recall_at2, group_recall_at3, \
                recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, \
                reference_names, modification_texts, target_names, sorted_index_names = results
            else:
                # recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, group_recall_at1, group_recall_at2, group_recall_at3 = results
                recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5 = results

        else:
            start_time = time.perf_counter()
            results = compute_oacirr_val_metrics(
                relative_val_dataset, blip_model, val_index_features, val_index_names, txt_processors, 
                highlight_inference=args.highlight_inference, save_results=args.save_results, save_memory=args.save_memory
            )
            duration = time.perf_counter() - start_time

            if args.save_results:
                # recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, group_recall_at1, group_recall_at2, group_recall_at3, \
                recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, \
                reference_names, modification_texts, target_names, sorted_index_names, reference_bboxes, activation_scalars = results
            else:
                # recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, group_recall_at1, group_recall_at2, group_recall_at3 = results
                recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5 = results

        results_dict = {
            # 'R_sub@1': group_recall_at1,
            # 'R_sub@2': group_recall_at2,
            # 'R_sub@3': group_recall_at3,
            'R_ID@1': class_recall_at1,
            'R_ID@3': class_recall_at3,
            'R_ID@5': class_recall_at5,
            'R@1': recall_at1,
            'R@5': recall_at5,
            'R@10': recall_at10,
            'R@50': recall_at50,
            'inference_time': duration
        }

        print("\n" + "="*40)
        print(f"Results for OACIRR [{args.dataset}]:")
        print(json.dumps(results_dict, indent=4))
        print("="*40 + "\n")

        # Save JSON result for qualitative analysis
        if args.save_results:
            base_dir = '/'.join(args.blip_model_weight.split('/')[:-2])
            save_path = Path(base_dir) / 'saved_results'
            save_path.mkdir(exist_ok=True, parents=True)

            Results_Dict = {
                'reference_names': reference_names,
                'modification_texts': modification_texts,
                'target_names': target_names,
                'sorted_index_names': sorted_index_names.tolist()
            }

            if args.bounding_box_width or args.bounding_box_crop:
                file_name = f'validation_results_bbox_{args.dataset.lower()}.json'
            else:
                Results_Dict['reference_bboxes'] = reference_bboxes
                Results_Dict['activation_scalars'] = [s.item() if s.numel() == 1 else s.numpy().tolist() for s in activation_scalars]
                file_name = f'validation_results_{args.dataset.lower()}.json'

            with open(save_path / file_name, 'w', encoding='utf-8') as file:
                json.dump(Results_Dict, file, indent=4)

            print(f"Results successfully saved to {save_path / file_name}")


    # ==================== CIRR Evaluation ====================
    elif args.dataset == 'CIRR':

        # Define the validation datasets
        relative_val_dataset = CIRRDataset(args.data_root, 'val', 'relative', preprocess)
        classic_val_dataset = CIRRDataset(args.data_root, 'val', 'classic', preprocess)

        # Extract target image features
        val_index_features, val_index_names = extract_index_blip_features(classic_val_dataset, blip_model)

        # Compute retrieval metrics
        start_time = time.perf_counter()

        results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features, val_index_names, txt_processors)

        end_time = time.perf_counter()
        duration = end_time - start_time

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
            'geometric_mean': geometric_mean(results),
            'inference_time': duration
        }

        print("\n" + "="*40)
        print(f"Results for CIRR:")
        print(json.dumps(results_dict, indent=4))
        print("="*40 + "\n")


    # ==================== FashionIQ Evaluation ====================
    elif args.dataset == 'FashionIQ':

        idx_to_dress_mapping = {0: 'dress', 1: 'toptee', 2: 'shirt'}
        recalls_at10 = []
        recalls_at50 = []
        duration = 0

        # Compute and log validation metrics for each FashionIQ category
        for idx, dress_type in idx_to_dress_mapping.items():
            # Define the validation datasets
            relative_val_dataset = FashionIQDataset(args.data_root, 'val', [dress_type], 'relative', preprocess)
            classic_val_dataset = FashionIQDataset(args.data_root, 'val', [dress_type], 'classic', preprocess)

            # Extract target image features
            index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, args.save_memory)

            # Compute retrieval metrics
            start_time = time.perf_counter()

            recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model, index_features, index_names, txt_processors, args.save_memory)

            end_time = time.perf_counter()
            duration += end_time - start_time

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)

            torch.cuda.empty_cache()

        results_dict = {}
        for i in range(len(recalls_at10)):
            results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
            results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]

        results_dict.update({
            'average_recall_at10': mean(recalls_at10),
            'average_recall_at50': mean(recalls_at50),
            'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
            'inference_time': duration
        })

        print("\n" + "="*40)
        print(f"Results for FashionIQ:")
        print(json.dumps(results_dict, indent=4))
        print("="*40 + "\n")


    else:
        raise ValueError("Dataset should be in ['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ']")
