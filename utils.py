import os
import random
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import OACIRRDataset, FashionIQDataset, CIRRDataset, Ipr2prDataset, CIRCODataset


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def collate_fn(batch: list):
    """
        Discard None images in a batch when using torch DataLoader
        :param batch: input_batch
        :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def custom_collate_fn(batch: list):

    transposed_batch = list(zip(*batch))

    if len(transposed_batch) == 4:
        ref_names = transposed_batch[0]
        tgt_names = transposed_batch[1]
        mod_texts = transposed_batch[2]
        ref_bboxes = transposed_batch[3]

        return (
            list(ref_names),
            list(tgt_names),
            list(mod_texts),
            list(ref_bboxes)
        )

    elif len(transposed_batch) == 5:
        ref_names = transposed_batch[0]
        tgt_names = transposed_batch[1]
        mod_texts = transposed_batch[2]
        group_names_list = transposed_batch[3]
        ref_bboxes = transposed_batch[4]

        return (
            list(ref_names),
            list(tgt_names),
            list(mod_texts),
            list(group_names_list),
            list(ref_bboxes)
        )

    elif len(transposed_batch) == 6:
        ref_images_tuple = transposed_batch[0]
        tgt_images_tuple = transposed_batch[1]
        ref_images_batch = torch.stack(ref_images_tuple, dim=0)
        tgt_images_batch = torch.stack(tgt_images_tuple, dim=0)
        ref_names = transposed_batch[2]
        tgt_names = transposed_batch[3]
        mod_texts = transposed_batch[4]
        ref_bboxes = transposed_batch[5]

        return (
            ref_images_batch,
            tgt_images_batch,
            list(ref_names),
            list(tgt_names),
            list(mod_texts),
            list(ref_bboxes)
        )

    else:
        raise ValueError("Batch data cannot be parsed. ")


def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset, OACIRRDataset], clip_model) -> Tuple[torch.tensor, List[str]]:
    """
        Extract OACIRR, FashionIQ or CIRR index features using CLIP image encoder
    """
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []

    if isinstance(dataset, CIRRDataset):
        print(f"Extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"Extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    elif isinstance(dataset, OACIRRDataset):
        print(f"Extracting OACIRR [{dataset.variant}] {dataset.split} index features")

    for names, images in tqdm(classic_val_loader, ncols=140, ascii=True):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)

    return index_features, index_names


def extract_index_blip_features(dataset: Union[CIRRDataset, FashionIQDataset, OACIRRDataset], blip_model, save_memory=False) -> Tuple[Tuple[torch.tensor, torch.tensor], List[str]]:
    """
        Extract OACIRR, FashionIQ or CIRR index features using Blip-2 Q-Former model
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_features_raw = []
    index_names = []

    if isinstance(dataset, CIRRDataset):
        print(f"Extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"Extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    elif isinstance(dataset, OACIRRDataset):
        print(f"Extracting OACIRR [{dataset.variant}] {dataset.split} index features")
    elif isinstance(dataset, CIRCODataset):
        print(f"Extracting CIRCO {dataset.split} index features")

    for names, images in tqdm(classic_val_loader, ncols=140, ascii=True):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features, image_embeds_frozen = blip_model.extract_target_features(images, mode="mean")
            if save_memory:
                image_features = image_features.cpu()
                image_embeds_frozen = image_embeds_frozen.cpu()
            index_features.append(image_features)
            index_features_raw.append(image_embeds_frozen)
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    index_features_raw = torch.vstack(index_features_raw)

    return (index_features, index_features_raw), index_names


def extract_index_blip_features_circo(dataset: CIRCODataset, blip_model, output_dir: str, save_memory=False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[str]]:

    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=6, pin_memory=True)
    index_names = []

    blip_model.eval()
    with torch.no_grad():
        _, sample_images = next(iter(classic_val_loader))
        sample_proj, sample_raw = blip_model.extract_target_features(sample_images.to(device), mode="mean")

    proj_shape = (len(dataset), sample_proj.shape[1], sample_proj.shape[2])
    if sample_raw.ndim == 3:
        raw_shape = (len(dataset), sample_raw.shape[1], sample_raw.shape[2])
    else:
        raw_shape = (len(dataset), sample_raw.shape[1])

    sample_proj_np = sample_proj.cpu().numpy()
    sample_raw_np = sample_raw.cpu().numpy()

    proj_features_path = os.path.join(output_dir, "proj_features.mmap")
    raw_features_path = os.path.join(output_dir, "raw_features.mmap")

    index_features_memmap = np.memmap(proj_features_path, dtype=sample_proj_np.dtype, mode='w+', shape=proj_shape)
    index_features_raw_memmap = np.memmap(raw_features_path, dtype=sample_raw_np.dtype, mode='w+', shape=raw_shape)

    current_pos = 0
    for names, images in tqdm(classic_val_loader, ncols=140, ascii=True, desc="Extracting Index Features"):
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            proj_batch, raw_batch = blip_model.extract_target_features(images, mode="mean")

        batch_len = len(names)

        index_features_memmap[current_pos : current_pos + batch_len] = proj_batch.cpu().numpy()
        index_features_raw_memmap[current_pos : current_pos + batch_len] = raw_batch.cpu().numpy()

        index_names.extend(names)
        current_pos += batch_len

    index_features_memmap.flush()
    index_features_raw_memmap.flush()

    index_features = torch.from_numpy(index_features_memmap)
    index_features_raw = torch.from_numpy(index_features_raw_memmap)

    return (index_features, index_features_raw), index_names


def extract_index_fuse_features(dataset: Union[CIRRDataset, FashionIQDataset, OACIRRDataset], fuse_model) -> Tuple[torch.tensor, List[str]]:
    """
        Extract OACIRR, FashionIQ or CIRR index features using the image encoder of the fusion model
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []

    if isinstance(dataset, CIRRDataset):
        print(f"Extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"Extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    elif isinstance(dataset, OACIRRDataset):
        print(f"Extracting OACIRR [{dataset.variant}] {dataset.split} index features")

    for names, images in tqdm(classic_val_loader, ncols=140, ascii=True):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features = fuse_model.retrieval_transformer.encode_image(images)
            index_features.append(image_features)
            index_names.extend(names)
    
    index_features = torch.vstack(index_features)

    return (index_features), index_names


def extract_index_target_features(dataset, blip_model, save_memory=False) -> Tuple[torch.tensor, List[str]]:
    """
        Extract OACIRR, FashionIQ or CIRR training set index features
    """
    classic_train_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=6, pin_memory=True, collate_fn=collate_fn)

    train_index_features = []
    train_index_names = []

    if isinstance(dataset, CIRRDataset):
        print(f"Extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"Extracting FashionIQ {dataset.dress_types} - {dataset.split} index features")
    elif isinstance(dataset, OACIRRDataset):
        print(f"Extracting OACIRR [{dataset.variant}] {dataset.split} index features")

    for names, images in tqdm(classic_train_loader, ncols=140, ascii=True):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features, _ = blip_model.extract_target_features(images, mode="mean")
            if save_memory:
                image_features = image_features.cpu()
            train_index_features.append(image_features)
            train_index_names.extend(names)

    train_index_features = torch.vstack(train_index_features)

    return train_index_features, train_index_names


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
        Normalized element-wise sum of image features and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
        Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
    """
    captions = []

    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")

    return captions


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
        Update `train_running_results` dict during training
        :param train_running_results: logging training dict
        :param loss: computed loss for batch
        :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu', non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
        Update tqdm train bar during training
        :param train_bar: tqdm training bar
        :param epoch: current epoch
        :param num_epochs: numbers of epochs
        :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def update_train_running_results_dict(train_running_results: dict, loss_dict: dict, images_in_batch: int):
    """
        Update `train_running_results` dict during training
        :param train_running_results: logging training dict
        :param loss: computed loss for batch
        :param images_in_batch: num images in the batch
    """
    for key in loss_dict.keys():
        if key not in train_running_results:
            train_running_results[key] = 0
        train_running_results[key] += loss_dict[key].to('cpu', non_blocking=True).detach().item() * images_in_batch

    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description_dict(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
        Update tqdm train bar during training
        :param train_bar: tqdm training bar
        :param epoch: current epoch
        :param num_epochs: numbers of epochs
        :param train_running_results: logging training dict
    """ 
    images_in_epoch = train_running_results['images_in_epoch']

    bar_content = ''

    for key in train_running_results:
        if key != 'images_in_epoch':
            bar_content += f'{key}: {train_running_results[key] / images_in_epoch:.3f}, '

    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] {bar_content}"
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
        Save the weights of the model during training
        :param name: name of the file
        :param cur_epoch: current epoch
        :param model_to_save: pytorch model to be saved
        :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))
