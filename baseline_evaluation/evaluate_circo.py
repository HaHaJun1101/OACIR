import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lavis.models import load_model_and_preprocess
from data_utils import squarepad_transform, targetpad_transform, CIRCODataset
from utils import collate_fn, extract_index_blip_features_circo, device


def compute_circo_val_metrics(relative_val_dataset: CIRCODataset, blip_model, index_features, index_names: List[str], txt_processors, save_dir: Path) -> None:
    """
        Compute predictions on CIRCO dataset and save them as JSON for submission / evaluation.
    """

    # Generate predictions
    pred_sim, reference_names, target_names, captions_all, ids = generate_circo_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors)

    print("Formatting CIRCO predictions...")

    # Compute the distances and sort the results
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(reference_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)

    rst_dict = {}
    for i, l in enumerate(sorted_index_names):
        rst_list = l[:50].tolist()
        rst_dict.update({str(ids[i]): rst_list})

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"circo_retrieval_results_{relative_val_dataset.split}.json"

    with open(out_path, "w") as f:
        json.dump(rst_dict, f)
    print(f"Results successfully saved to {out_path}")


def generate_circo_val_predictions(blip_model, relative_val_dataset: CIRCODataset, index_names: List[str], index_features, txt_processors) \
                                -> Tuple[torch.tensor, List[str], List[str], List[List[str]], List[str]]:
    """
        Compute CIRCO predictions on the validation set
    """

    print(f"Computing CIRCO {relative_val_dataset.split} predictions...")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=4, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    distance = []
    target_names = []
    reference_names = []
    captions_all = []
    ids = []

    for item in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_images = item['reference_img']
        batch_reference_names = item['reference_imd_id']
        captions = [txt_processors["eval"](caption) for caption in item['relative_caption']]
        qid = item['query_id']

        # Compute the predicted features
        with torch.no_grad():
            batch_reference_images = batch_reference_images.to(device, non_blocking=True)
            _, reference_image_features = blip_model.extract_target_features(batch_reference_images, mode="mean")
            feature_curr = index_features[0].to(blip_model.device)
            reference_image_features = reference_image_features.to(blip_model.device)
            
            batch_distance = blip_model.inference(reference_image_features, feature_curr, captions)
            distance.append(batch_distance.cpu())
            captions_all += captions

        ids.extend(qid)
        reference_names.extend(batch_reference_names)

    distance = torch.vstack(distance)

    return distance, reference_names, target_names, captions_all, ids



if __name__ == '__main__':

    parser = ArgumentParser("Evaluate the Baseline model on the CIRCO dataset")

    parser.add_argument("--data-root", type=str, default="../Datasets/CIRCO", help="Path to CIRCO dataset root")
    parser.add_argument("--save-dir", type=str, default="./submission/CIRCO", help="Directory to save the prediction JSON")

    parser.add_argument("--blip-model-name", type=str, default="oacir_baseline", help="Model registry name")
    parser.add_argument("--blip-model-weight", type=str, required=True, help="Path to the pre-trained model weight")
    parser.add_argument("--vit-backbone", type=str, default="pretrain", help="pretrain for Vit-G, pretrain_vitL for ViT-L")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, choices=['squarepad', 'targetpad'])
    parser.add_argument("--ntype", type=str, required=True, choices=['val', 'test'], help="Dataset split type")

    args = parser.parse_args()


    # Create temporary directory for feature files
    temp_dir = tempfile.mkdtemp(prefix="CIRCO_val_temp_")
    print(f"Created temporary directory for memory-mapped feature files: {temp_dir}")

    # Load Model
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.vit_backbone, is_eval=False, device=device)

    blip_model_checkpoint = torch.load(args.blip_model_weight, map_location=device)
    blip_msg = blip_model.load_state_dict(blip_model_checkpoint[blip_model.__class__.__name__], strict=False)
    print(f"Missing keys when loading weights: {blip_msg.missing_keys}")

    # Define image preprocess pipeline
    input_dim = 224

    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
        print(f"Target pad with target_ratio={args.target_ratio} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # Define the evaluation datasets
    data_root_path = Path(args.data_root)
    classic_val_dataset = CIRCODataset(data_root_path, args.ntype, 'classic', preprocess)
    relative_val_dataset = CIRCODataset(data_root_path, args.ntype, 'relative', preprocess)

    # Extract target image features
    val_index_features, val_index_names = extract_index_blip_features_circo(classic_val_dataset, blip_model, output_dir=temp_dir, save_memory=True)

    # Compute retrieval metrics (generates JSON file)
    compute_circo_val_metrics(relative_val_dataset, blip_model, val_index_features, val_index_names, txt_processors, Path(args.save_dir))

    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
