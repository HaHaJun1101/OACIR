import os
import sys
import json
import argparse
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from tqdm import tqdm
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_utils import CIRRDataset, targetpad_transform, squarepad_transform
from utils import device, extract_index_blip_features, collate_fn
from lavis.models import load_model_and_preprocess


def generate_cirr_test_submissions(file_name: str, data_root: str, blip_model, preprocess, txt_processors, index_features, index_names):
    """
        Generate and save CIRR test submission files to be submitted to evaluation server
    """

    # Define the CIRR test dataset
    relative_test_dataset = CIRRDataset(data_root, 'test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, blip_model, index_features, index_names, txt_processors)

    submission = {'version': 'rc2', 'metric': 'recall'}
    group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = Path('./submission/CIRR')
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions to {submissions_folder_path}")

    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, blip_model, index_features: torch.tensor, index_names: List[str], txt_processors):
    """
        Compute test prediction dicts for CIRR dataset
        :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_sim, reference_names, group_members, pairs_id, _, _ = generate_cirr_test_predictions(blip_model, relative_test_dataset, index_names, index_features, txt_processors)

    # Compute the distances and sort the results
    print(f"Compute CIRR prediction dicts")
    distances = 1 - predicted_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)

    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_model, relative_test_dataset: CIRRDataset, index_names: List[str], index_features: torch.tensor, txt_processors):
    """
        Compute CIRR predictions on the test set
    """

    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features[1]))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    distance = []
    reference_names = []
    group_members = []
    pairs_id = []
    captions_all = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(relative_test_loader, ncols=140, ascii=True):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]

        # Compute the predicted features
        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
                # To avoid unnecessary computation retrieve the reference image features directly from the index features

            batch_distance = blip_model.inference(reference_image_features, index_features[0], captions)
            distance.append(batch_distance)
            captions_all += captions

        reference_names.extend(batch_reference_names)
        group_members.extend(batch_group_members)
        pairs_id.extend(batch_pairs_id)

    distance = torch.vstack(distance)

    return distance, reference_names, group_members, pairs_id, captions_all, name_to_feat


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':

    parser = ArgumentParser("Generate Test Submissions for CIRR Dataset")

    parser.add_argument("--data-root", type=str, default="../Datasets/cirr_dataset", help="Root directory of the dataset")
    parser.add_argument("--blip-model-name", type=str, default="oacir_baseline", help="Model registry name")
    parser.add_argument("--blip-model-weight", type=str, required=True, help="Path to pre-trained model weight for submission generation")

    parser.add_argument("--vit-backbone", type=str, default="pretrain", help="pretrain for Vit-G, pretrain_vitL for ViT-L")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, choices=['squarepad', 'targetpad'])
    
    parser.add_argument("--file-name", type=str, default="cirr_test_submission", help="Suffix name for the generated submission files")

    args = parser.parse_args()


    # Load Model
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.vit_backbone, is_eval=False, device=device)

    blip_model_checkpoint = torch.load(args.blip_model_weight, map_location=device)
    blip_msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
    print(f"Missing keys when loading weights: {blip_msg.missing_keys}")

    # Transforms
    input_dim = 224

    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
        print(f"Target pad with target_ratio={args.target_ratio} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # Extract CIRR test dataset index features
    classic_test_dataset = CIRRDataset(args.data_root, 'test1', 'classic', preprocess)
    index_features, index_names = extract_index_blip_features(classic_test_dataset, blip_model)

    generate_cirr_test_submissions(args.file_name, args.data_root, blip_model, preprocess, txt_processors, index_features, index_names)
