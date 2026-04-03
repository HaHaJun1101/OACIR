import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple
from tqdm import tqdm

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from statistics import mean, harmonic_mean, geometric_mean
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classic_collate_fn(batch):
    image_names, images = zip(*batch)
    return list(image_names), list(images)

def oacirr_relative_collate_fn(batch):
    ref_names, target_names, mod_texts, ref_images = zip(*batch)
    return list(ref_names), list(target_names), list(mod_texts), list(ref_images)

def cirr_relative_collate_fn(batch):
    ref_names, target_names, mod_texts, group_members, ref_images = zip(*batch)
    return list(ref_names), list(target_names), list(mod_texts), list(group_members), list(ref_images)

def fiq_relative_collate_fn(batch):
    ref_names, target_names, mod_texts, ref_images = zip(*batch)
    return list(ref_names), list(target_names), list(mod_texts), list(ref_images)


OACIRR_CONFIGS = {
    'Fashion': {'dir': 'OACIRR-Fashion', 'anno': 'oacirr-fashion'},
    'Car': {'dir': 'OACIRR-Car', 'anno': 'oacirr-car'},
    'Product': {'dir': 'OACIRR-Product', 'anno': 'oacirr-product'},
    'Landmark': {'dir': 'OACIRR-Landmark', 'anno': 'oacirr-landmark'}
}


class BaseOACIRRDataset(Dataset):
    """
        Base dataset class for the OACIRR benchmark.
    """
    def __init__(self, dataset_name: str, data_root_path: Path, mode: str,
                 text_entity: bool = False, bounding_box_width: int = 0, bounding_box_color: str = 'red'):

        self.dataset_name = dataset_name
        self.data_root_path = Path(data_root_path)
        self.config = OACIRR_CONFIGS[dataset_name]
        self.mode = mode
        self.text_entity = text_entity
        self.bounding_box_width = bounding_box_width
        self.bounding_box_color = bounding_box_color

        # Instruction templates for UMR evaluation
        self.prompts = ["Same {entity}", "With the same {entity}", "Identical {entity}", "{entity} unchanged",
                        "Preserving the {entity}", "Invariant {entity}", "Keep the {entity}", "Fixed {entity}"]
        self.num_prompts = len(self.prompts)

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        anno_dir = self.data_root_path / self.config['dir'] / self.config['anno']

        with open(anno_dir / 'quadruple_captions' / 'caption_full.val.json') as f:
            self.quadruples = json.load(f)
        with open(anno_dir / 'image_splits' / 'split.val.json') as f:
            self.name_to_relpath = json.load(f)

        self.classic_image_names = list(self.name_to_relpath.keys())

        print(f"OACIRR [{self.dataset_name}] validation dataset in {mode} mode initialized.")

    def __getitem__(self, index):
        if self.mode == 'relative':
            quadruple = self.quadruples[index]
            reference_name = quadruple['reference']
            target_name = quadruple['target']
            modification_text = quadruple['modification_text_mllm']

            if self.text_entity:
                entity = quadruple['object_category']
                prompt_template = self.prompts[index % self.num_prompts]
                final_prompt = prompt_template.format(entity=entity)
                if self.bounding_box_width > 0:
                    bbox_prompt = f" in the {self.bounding_box_color} bounding box"
                    final_prompt += bbox_prompt
                modification_text = f"{final_prompt}, {modification_text}"

            reference_image_path = self.data_root_path / self.config['dir'] / self.name_to_relpath[reference_name]
            reference_image = Image.open(reference_image_path).convert("RGB")

            # Draw Visual Bounding Box
            if self.bounding_box_width > 0:
                bounding_box = quadruple['reference_bounding_box']
                draw = ImageDraw.Draw(reference_image)
                draw.rectangle(bounding_box, outline=self.bounding_box_color, width=self.bounding_box_width)

            return reference_name, target_name, modification_text, reference_image

        elif self.mode == 'classic':
            image_name = self.classic_image_names[index]
            image_path = self.data_root_path / self.config['dir'] / self.name_to_relpath[image_name]
            image = Image.open(image_path).convert("RGB")

            return image_name, image

    def __len__(self):
        return len(self.quadruples) if self.mode == 'relative' else len(self.classic_image_names)


class CIRRDataset(Dataset):
    """
        CIRR dataset class.
    """
    def __init__(self, data_root_path: str, mode: str):
        self.mode = mode
        self.data_path = Path(data_root_path)

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        with open(self.data_path / 'cirr' / 'captions' / 'cap.rc2.val.json') as f:
            self.triplets = json.load(f)
        with open(self.data_path / 'cirr' / 'image_splits' / 'split.rc2.val.json') as f:
            self.name_to_relpath = json.load(f)

        self.classic_image_names = list(self.name_to_relpath.keys())

        print(f"CIRR validation dataset in {mode} mode initialized.")

    def __getitem__(self, index):
        if self.mode == 'relative':
            reference_name = self.triplets[index]['reference']
            target_name = self.triplets[index]['target_hard']
            rel_caption = self.triplets[index]['caption']
            group_members = self.triplets[index]['img_set']['members']

            reference_image_path = self.data_path / self.name_to_relpath[reference_name]
            reference_image = Image.open(reference_image_path).convert("RGB")

            return reference_name, target_name, rel_caption, group_members, reference_image

        elif self.mode == 'classic':
            image_name = self.classic_image_names[index]
            image_path = self.data_path / self.name_to_relpath[image_name]
            image = Image.open(image_path).convert("RGB")

            return image_name, image

    def __len__(self):
        return len(self.triplets) if self.mode == 'relative' else len(self.classic_image_names)


class FashionIQDataset(Dataset):
    """
        FashionIQ dataset class.
    """
    def __init__(self, data_root_path: str, dress_types: List[str], mode: str):
        self.mode = mode
        self.dress_types = dress_types
        self.data_path = Path(data_root_path)

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.data_path / 'captions' / f'cap.{dress_type}.val.json') as f:
                self.triplets.extend(json.load(f))

        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.data_path / 'image_splits' / f'split.{dress_type}.val.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ validation - {dress_types} dataset in {mode} mode initialized.")

    def __getitem__(self, index):
        if self.mode == 'relative':
            image_captions = self.triplets[index]['captions']
            reference_name = self.triplets[index]['candidate']
            target_name = self.triplets[index]['target']

            reference_image_path = self.data_path / 'images' / f"{reference_name}.png"
            reference_image = Image.open(reference_image_path).convert("RGB")

            return reference_name, target_name, image_captions, reference_image

        elif self.mode == 'classic':
            image_name = self.image_names[index]
            image_path = self.data_path / 'images' / f"{image_name}.png"
            image = Image.open(image_path).convert("RGB")

            return image_name, image

    def __len__(self):
        return len(self.triplets) if self.mode == 'relative' else len(self.image_names)


def compute_class_recall(sorted_index_names: np.ndarray, target_names: List[str], name_to_relpath: dict) -> Tuple[float, float, float]:
    """
        Compute instance-level recall.
    """
    name_to_folder = {name: path.split('/')[-2] for name, path in name_to_relpath.items()}
    target_folders = np.array([name_to_folder.get(name) for name in target_names])
    folder_mapper = np.vectorize(name_to_folder.get)
    sorted_index_folders = folder_mapper(sorted_index_names)

    class_labels = (sorted_index_folders == target_folders[:, np.newaxis])
    modified_class_labels = ((class_labels == 1) & (np.cumsum(class_labels, axis=1) == 1)).astype(int)

    class_recall_at1 = (np.sum(modified_class_labels[:, :1]) / len(target_names)) * 100
    class_recall_at3 = (np.sum(modified_class_labels[:, :3]) / len(target_names)) * 100
    class_recall_at5 = (np.sum(modified_class_labels[:, :5]) / len(target_names)) * 100

    return class_recall_at1, class_recall_at3, class_recall_at5


def load_instructions(instruction_path: str, dataset_name: str) -> list[str]:
    try:
        df = pd.read_csv(instruction_path, sep='\t')
        prompts = df[df['dataset_name'] == dataset_name][['prompt_1', 'prompt_2', 'prompt_3', 'prompt_4']].values.flatten().tolist()
        return prompts
    except Exception as e:
        print(f"Loading instruction failed: {e}. Using the default instruction.")
        return ["Retrieve a image that reflects the described transformation from the provided image."]


def extract_index_features(dataset, model, batch_size: int, num_workers: int) -> tuple[torch.Tensor, list[str]]:
    candidate_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=classic_collate_fn, pin_memory=False)
    index_features, index_names = [], []

    for names, images in tqdm(candidate_dataloader, desc="Extracting Gallery Features", ncols=140, ascii=True):
        candidates = [{'img': img} for img in images]
        with torch.no_grad():
            embeddings = model.encode(candidates, max_length=4096)['hidden_states']
            index_features.append(embeddings.cpu())
            index_names.extend(names)

    return torch.vstack(index_features), index_names


def compute_oacirr_metrics(relative_val_dataset, model, index_features, index_names, instructions, batch_size: int, num_workers: int, save_results: bool):
    relative_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=oacirr_relative_collate_fn, pin_memory=False)
    all_pred_sims, all_ref_names, all_target_names, all_mod_texts = [], [], [], []
    index_features_gpu = index_features.to(device)

    for ref_names, target_names, mod_texts, ref_images in tqdm(relative_dataloader, ncols=140, ascii=True):
        queries = [{'txt': mod_texts[i], 'img': ref_images[i]} for i in range(len(ref_images))]
        batch_instructions = [random.choice(instructions) for _ in range(len(queries))]

        with torch.no_grad():
            query_embeddings = model.encode(queries, is_query=True, instruction=batch_instructions, max_length=4096)['hidden_states']

        batch_sims = query_embeddings.to(device) @ index_features_gpu.T

        all_pred_sims.append(batch_sims.cpu())
        all_ref_names.extend(ref_names)
        all_target_names.extend(target_names)
        all_mod_texts.extend(mod_texts)

    pred_sim = torch.vstack(all_pred_sims)
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    ref_names_array = np.array(all_ref_names)[:, np.newaxis]
    mask = (sorted_index_names != ref_names_array)
    sorted_index_names_masked = sorted_index_names[mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)

    target_names_array = np.array(all_target_names)[:, np.newaxis]
    labels = (sorted_index_names_masked == target_names_array)

    assert np.all(np.sum(labels, axis=1) == 1)

    recall_at1 = np.sum(labels[:, :1]) / len(labels) * 100
    recall_at5 = np.sum(labels[:, :5]) / len(labels) * 100
    recall_at10 = np.sum(labels[:, :10]) / len(labels) * 100
    recall_at50 = np.sum(labels[:, :50]) / len(labels) * 100

    class_recall_at1, class_recall_at3, class_recall_at5 = compute_class_recall(
        sorted_index_names_masked, all_target_names, relative_val_dataset.name_to_relpath
    )

    if save_results:
        return recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, all_ref_names, all_mod_texts, all_target_names, sorted_index_names_masked
    else:
        return recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5


def compute_cirr_metrics(relative_val_dataset, model, index_features, index_names, instructions, batch_size: int, num_workers: int, save_results: bool):
    relative_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=cirr_relative_collate_fn, pin_memory=False)
    all_pred_sims, all_ref_names, all_target_names, all_mod_texts, all_group_members = [], [], [], [], []
    index_features_gpu = index_features.to(device)

    for ref_names, target_names, mod_texts, group_members, ref_images in tqdm(relative_dataloader, ncols=140, ascii=True):
        queries = [{'txt': mod_texts[i], 'img': ref_images[i]} for i in range(len(ref_images))]
        batch_instructions = [random.choice(instructions) for _ in range(len(queries))]

        with torch.no_grad():
            query_embeddings = model.encode(queries, is_query=True, instruction=batch_instructions, max_length=4096)['hidden_states']

        batch_sims = query_embeddings.to(device) @ index_features_gpu.T

        all_pred_sims.append(batch_sims.cpu())
        all_ref_names.extend(ref_names)
        all_target_names.extend(target_names)
        all_mod_texts.extend(mod_texts)
        all_group_members.extend(group_members)

    pred_sim = torch.vstack(all_pred_sims)
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    ref_names_array = np.array(all_ref_names)[:, np.newaxis]
    mask = (sorted_index_names != ref_names_array)
    sorted_index_names_masked = sorted_index_names[mask].reshape(sorted_index_names.shape[0], -1)

    target_names_array = np.array(all_target_names)[:, np.newaxis]
    labels = (sorted_index_names_masked == target_names_array)

    recall_at1 = np.sum(labels[:, :1]) / len(labels) * 100
    recall_at5 = np.sum(labels[:, :5]) / len(labels) * 100
    recall_at10 = np.sum(labels[:, :10]) / len(labels) * 100
    recall_at50 = np.sum(labels[:, :50]) / len(labels) * 100

    group_members = np.array(all_group_members)
    group_mask = (sorted_index_names_masked[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    group_recall_at1 = np.sum(group_labels[:, :1]) / len(group_labels) * 100
    group_recall_at2 = np.sum(group_labels[:, :2]) / len(group_labels) * 100
    group_recall_at3 = np.sum(group_labels[:, :3]) / len(group_labels) * 100

    if save_results:
        return recall_at1, recall_at5, recall_at10, recall_at50, group_recall_at1, group_recall_at2, group_recall_at3, all_ref_names, all_mod_texts, all_target_names, sorted_index_names_masked
    else:
        return recall_at1, recall_at5, recall_at10, recall_at50, group_recall_at1, group_recall_at2, group_recall_at3


def compute_fiq_metrics(relative_val_dataset, model, index_features, index_names, instructions, batch_size: int, num_workers: int, save_results: bool):
    relative_dataloader = DataLoader(dataset=relative_val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=fiq_relative_collate_fn, pin_memory=False)
    all_pred_sims, all_ref_names, all_target_names, all_mod_texts = [], [], [], []
    index_features_gpu = index_features.to(device)

    for ref_names, target_names, mod_texts, ref_images in tqdm(relative_dataloader, ncols=140, ascii=True):
        processed_mod_texts = [f"{texts[0].strip('.?, ').capitalize()} and {texts[1].strip('.?, ')}" for texts in mod_texts]
        queries = [{'txt': txt, 'img': img} for txt, img in zip(processed_mod_texts, ref_images)]
        batch_instructions = [random.choice(instructions) for _ in range(len(queries))]

        with torch.no_grad():
            query_embeddings = model.encode(queries, is_query=True, instruction=batch_instructions, max_length=4096)['hidden_states']

        batch_sims = query_embeddings.to(device) @ index_features_gpu.T
        all_pred_sims.append(batch_sims.cpu())
        all_ref_names.extend(ref_names)
        all_target_names.extend(target_names)
        all_mod_texts.extend(processed_mod_texts)

    pred_sim = torch.vstack(all_pred_sims)
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    target_names_array = np.array(all_target_names)
    labels = (sorted_index_names == target_names_array[:, None])

    recall_at10 = np.sum(labels[:, :10]) / len(labels) * 100
    recall_at50 = np.sum(labels[:, :50]) / len(labels) * 100

    if save_results:
        return recall_at10, recall_at50, all_ref_names, all_mod_texts, all_target_names, sorted_index_names
    else:
        return recall_at10, recall_at50


if __name__ == '__main__':

    parser = ArgumentParser("Validate the MM-Embed model on the OACIRR benchmark and standard CIR datasets")

    parser.add_argument("--dataset", type=str, required=True, choices=['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ'], help="Dataset to evaluate on")
    parser.add_argument("--dataset-root", type=str, default="Datasets/OACIRR/OACIRR-Subset", help="Root directory of the datasets (OACIRR Subsets or standard CIR)")
    parser.add_argument("--model-id", type=str, default="nvidia/MM-Embed", help="The model ID of MM-Embed on Hugging Face")
    parser.add_argument("--batch-size", type=int, default=16, help="The batch size during the validation")
    parser.add_argument("--num-workers", type=int, default=6, help="The number of worker processes used by DataLoader")
    parser.add_argument("--instruction-path", type=str, default="Datasets/M-BEIR/instructions/query_instructions.tsv", help="Path of the instruction TSV file")

    parser.add_argument("--text-entity", action='store_true', help="Add personalized entity prompt to the modification text")
    parser.add_argument("--bounding-box-width", default=0, type=int, help="Reference image bounding box width")
    parser.add_argument("--bounding-box-color", default="red", type=str, help="Reference image bounding box color")

    parser.add_argument("--save-results", action='store_true', help="Whether save the full validation results")
    parser.add_argument("--results-dir", type=str, default='./umr_results/MM-Embed', help="Directory to save the results")
    parser.add_argument("--results-name", type=str, default='validation_metrics', help="File name prefix to save the result")

    args = parser.parse_args()


    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()


    if args.dataset == 'CIRR':
        relative_val_dataset = CIRRDataset(args.dataset_root, 'relative')
        classic_val_dataset = CIRRDataset(args.dataset_root, 'classic')
        instructions = load_instructions(args.instruction_path, args.dataset)

        val_index_features, val_index_names = extract_index_features(classic_val_dataset, model, args.batch_size, args.num_workers)
        results = compute_cirr_metrics(relative_val_dataset, model, val_index_features, val_index_names, instructions, args.batch_size, args.num_workers, args.save_results)

        if args.save_results:
            recall_at1, recall_at5, recall_at10, recall_at50, group_recall_at1, group_recall_at2, group_recall_at3, ref_names, mod_texts, target_names, sorted_names = results
        else:
            recall_at1, recall_at5, recall_at10, recall_at50, group_recall_at1, group_recall_at2, group_recall_at3 = results

        recall_results = [recall_at1, recall_at5, recall_at10, recall_at50]
        group_recall_results = [group_recall_at1, group_recall_at2, group_recall_at3]

        results_dict = {
            'group_recall_at1': group_recall_at1,
            'group_recall_at2': group_recall_at2,
            'group_recall_at3': group_recall_at3,
            'recall_at1': recall_at1,
            'recall_at5': recall_at5,
            'recall_at10': recall_at10,
            'recall_at50': recall_at50,
            'mean(R@5+R_s@1)': (recall_at5 + group_recall_at1) / 2,
            'arithmetic_mean': mean(recall_results + group_recall_results),
            'harmonic_mean': harmonic_mean(recall_results + group_recall_results),
            'geometric_mean': geometric_mean(recall_results + group_recall_results)
        }


    elif args.dataset == 'FashionIQ':
        idx_to_dress_mapping = {0: 'dress', 1: 'toptee', 2: 'shirt'}
        recalls_at10, recalls_at50 = [], []
        instructions = load_instructions(args.instruction_path, args.dataset)

        if args.save_results:
            ref_names, mod_texts, target_names, sorted_names = [], [], [], []

        for idx, dress_type in idx_to_dress_mapping.items():
            relative_val_dataset = FashionIQDataset(args.dataset_root, [dress_type], 'relative')
            classic_val_dataset = FashionIQDataset(args.dataset_root, [dress_type], 'classic')

            val_index_features, val_index_names = extract_index_features(classic_val_dataset, model, args.batch_size, args.num_workers)
            results = compute_fiq_metrics(relative_val_dataset, model, val_index_features, val_index_names, instructions, args.batch_size, args.num_workers, args.save_results)

            if args.save_results:
                recall_at10, recall_at50, batch_ref_names, batch_mod_texts, batch_target_names, batch_sorted_names = results
                ref_names.extend(batch_ref_names)
                mod_texts.extend(batch_mod_texts)
                target_names.extend(batch_target_names)
                sorted_names.append(batch_sorted_names)
            else:
                recall_at10, recall_at50 = results

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)

        results_dict = {}
        for i in range(len(recalls_at10)):
            results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
            results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]

        results_dict.update({
            'average_recall_at10': mean(recalls_at10),
            'average_recall_at50': mean(recalls_at50),
            'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2,
        })


    else:
        # Evaluate on OACIRR Subsets
        relative_val_dataset = BaseOACIRRDataset(args.dataset, args.dataset_root, 'relative', args.text_entity, args.bounding_box_width, args.bounding_box_color)
        classic_val_dataset = BaseOACIRRDataset(args.dataset, args.dataset_root, 'classic')

        instructions = load_instructions(args.instruction_path, args.dataset)

        val_index_features, val_index_names = extract_index_features(classic_val_dataset, model, args.batch_size, args.num_workers)
        results = compute_oacirr_metrics(relative_val_dataset, model, val_index_features, val_index_names, instructions, args.batch_size, args.num_workers, args.save_results)

        if args.save_results:
            recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5, ref_names, mod_texts, target_names, sorted_names = results
        else:
            recall_at1, recall_at5, recall_at10, recall_at50, class_recall_at1, class_recall_at3, class_recall_at5 = results

        recall_results = [recall_at1, recall_at5, recall_at10, recall_at50]

        results_dict = {
            'class_recall_at1': class_recall_at1,
            'class_recall_at3': class_recall_at3,
            'class_recall_at5': class_recall_at5,
            'recall_at1': recall_at1,
            'recall_at5': recall_at5,
            'recall_at10': recall_at10,
            'recall_at50': recall_at50,
            'arithmetic_mean': mean(recall_results),
            'harmonic_mean': harmonic_mean(recall_results),
            'geometric_mean': geometric_mean(recall_results)
        }

    print(json.dumps(results_dict, indent=4))

    result_dir = Path(args.results_dir) / args.dataset
    result_dir.mkdir(exist_ok=True, parents=True)
    result_filepath = result_dir / f"{args.results_name}.txt"

    with open(result_filepath, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        for key, value in results_dict.items():
            f.write(f"{key:<30}: {value:.4f}\n")

    if args.save_results:
        Full_Results_Dict = {
            'reference_names': ref_names,
            'modification_texts': mod_texts,
            'target_names': target_names,
            'sorted_index_names': sorted_names.tolist() if isinstance(sorted_names, np.ndarray) else sorted_names
        }
        json_filepath = result_dir / f"{args.results_name}.json"
        with open(json_filepath, 'w', encoding='utf-8') as file:
            json.dump(Full_Results_Dict, file, indent=4)
