import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Literal

import PIL
import PIL.Image
import torchvision.transforms.functional as F
import torch
from PIL import ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def collate_fn(batch):
    """
        function which discard None images in a batch when using torch DataLoader
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
        Square pad the input image with zero padding
    """
    def __init__(self, size: int):
        """
            :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
        Pad the image if its aspect ratio is above a target ratio
        Pad the image to match such target ratio
    """
    def __init__(self, target_ratio: float, size: int):
        """
            :param target_ratio: target ratio
            :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(w, h) / self.target_ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
        CLIP-like preprocessing transform on a square padded image
        :param dim: image output dimension
        :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
        CLIP-like preprocessing transform computed after using TargetPad pad
        :param target_ratio: target ratio for TargetPad
        :param dim: image output dimension
        :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def mask_image(image_1, image_2, patch_size: int, mask_ratio: float):
    """
        Random mask patches of the input image

        Parameters:
        - image_1/2: Input image as a Numpy array of shape (C, H, W)
        - patch_size: Size of each image patch (assuming square patches)
        - mask_ratio: Ratio of image patches to be masked (between 0 and 1)

        Returns:
        - masked_image_1/2: Image with masked patches
    """
    C_1, H_1, W_1 = image_1.shape
    C_2, H_2, W_2 = image_2.shape
    assert H_1 == W_1, "Image_1 must be square"
    assert H_2 == W_2, "Image_2 must be square"
    assert H_1 == H_2, "Image_1 must be the same size as Image_2"
    assert H_1 % patch_size == 0, "Image size must be divisible by patch size"

    num_patches = (H_1 // patch_size) ** 2
    num_masked_patches = int(num_patches * mask_ratio)

    # Create a mask for the patches
    mask = np.zeros(num_patches, dtype=bool)
    mask[ : num_masked_patches] = True
    np.random.shuffle(mask)

    # Apply the mask to the image
    masked_image_1 = np.array(image_1).copy()
    masked_image_2 = np.array(image_2).copy()
    patch_idx = 0
    for i in range(0, H_1, patch_size):
        for j in range(0, W_1, patch_size):
            if mask[patch_idx]:
                masked_image_1[:, i:i+patch_size, j:j+patch_size] = 0  # Mask with black patch
                # masked_image_1[:, i:i+patch_size, j:j+patch_size] = 1  # Mask with white patch
                masked_image_2[:, i:i+patch_size, j:j+patch_size] = 0  # Mask with black patch
                # masked_image_2[:, i:i+patch_size, j:j+patch_size] = 1  # Mask with white patch
            patch_idx += 1

    return torch.tensor(masked_image_1), torch.tensor(masked_image_2)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> str:
    """
        Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
        :param flattened_captions: the list of caption to randomize, note that the length of such list is 2 * batch_size since to each triplet are associated two captions
        :return: the randomized caption list (with length = batch_size)
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


def transform_bbox_targetpad(bbox, original_size, target_ratio=1.25, final_size=224):
    """
        Perform an affine transformation on the bounding box that align with the targetpad_transform

        :param bbox (list or tuple): The [x1, y1, x2, y2] coordinates on the original image
        :param original_size (tuple): The original image's (width, height)
        :param target_ratio (float): The aspect ratio of the TargetPad
        :param final_size (int): The final target size of the square image

        :return: The [x1, y1, x2, y2] coordinates in the final_size x final_size coordinate system.
             If the transformed bounding box is completely outside the cropping area, then return None.
    """
    original_w, original_h = original_size
    x1, y1, x2, y2 = bbox

    actual_ratio = max(original_w, original_h) / min(original_w, original_h)
    padded_w, padded_h = original_w, original_h
    hp, vp = 0, 0

    if actual_ratio >= target_ratio:
        scaled_max_wh = max(original_w, original_h) / target_ratio
        hp = max(int((scaled_max_wh - original_w) / 2), 0)
        vp = max(int((scaled_max_wh - original_h) / 2), 0)

        padded_w = original_w + 2 * hp
        padded_h = original_h + 2 * vp

    x1_padded, y1_padded = x1 + hp, y1 + vp
    x2_padded, y2_padded = x2 + hp, y2 + vp

    if padded_w < padded_h:
        scale = final_size / padded_w
        resized_w = final_size
        resized_h = int(padded_h * scale)
    else:
        scale = final_size / padded_h
        resized_h = final_size
        resized_w = int(padded_w * scale)

    x1_resized, y1_resized = x1_padded * scale, y1_padded * scale
    x2_resized, y2_resized = x2_padded * scale, y2_padded * scale

    crop_x_start = (resized_w - final_size) / 2
    crop_y_start = (resized_h - final_size) / 2

    x1_cropped = x1_resized - crop_x_start
    x2_cropped = x2_resized - crop_x_start
    y1_cropped = y1_resized - crop_y_start
    y2_cropped = y2_resized - crop_y_start

    final_x1 = max(0, x1_cropped)
    final_y1 = max(0, y1_cropped)
    final_x2 = min(final_size, x2_cropped)
    final_y2 = min(final_size, y2_cropped)

    if final_x1 >= final_x2 or final_y1 >= final_y2:
        return None

    return [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]


# ==========================================================================================
# The Unified OACIRR Dataset Class
# ==========================================================================================

OACIRR_VARIANT_CONFIGS = {
    # 1. Joint Training Union
    'Union': {'img_dir': 'OACIRR-Union', 'anno_dir': 'OACIRR-Union/oacirr-union'},
    # 2. Single Domain Subsets
    'Fashion': {'img_dir': 'OACIRR-Subset/OACIRR-Fashion', 'anno_dir': 'OACIRR-Subset/OACIRR-Fashion/oacirr-fashion'},
    'Car': {'img_dir': 'OACIRR-Subset/OACIRR-Car', 'anno_dir': 'OACIRR-Subset/OACIRR-Car/oacirr-car'},
    'Product': {'img_dir': 'OACIRR-Subset/OACIRR-Product', 'anno_dir': 'OACIRR-Subset/OACIRR-Product/oacirr-product'},
    'Landmark': {'img_dir': 'OACIRR-Subset/OACIRR-Landmark', 'anno_dir': 'OACIRR-Subset/OACIRR-Landmark/oacirr-landmark'},
    # 3. Cross-Domain "Leave-One-Out" Training Sets
    'WO_Fashion': {'img_dir': 'OACIRR-CrossDomain/WO-Fashion', 'anno_dir': 'OACIRR-CrossDomain/WO-Fashion/oacirr-wo-fashion'},
    'WO_Car': {'img_dir': 'OACIRR-CrossDomain/WO-Car', 'anno_dir': 'OACIRR-CrossDomain/WO-Car/oacirr-wo-car'},
    'WO_Product': {'img_dir': 'OACIRR-CrossDomain/WO-Product', 'anno_dir': 'OACIRR-CrossDomain/WO-Product/oacirr-wo-product'},
    'WO_Landmark': {'img_dir': 'OACIRR-CrossDomain/WO-Landmark', 'anno_dir': 'OACIRR-CrossDomain/WO-Landmark/oacirr-wo-landmark'},
}

class OACIRRDataset(Dataset):
    """
        The Unified Dataset Class for the OACIRR Benchmark
    """

    def __init__(self, data_root: str, variant: str, split: str, mode: str, preprocess: callable,
                 highlight_training: bool = False, highlight_inference: bool = False, text_entity: bool = False,
                 bounding_box_width: int = 0, bounding_box_color: str = 'red', bounding_box_crop: bool = False):

        self.data_root = Path(data_root)
        self.variant = variant
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        if variant not in OACIRR_VARIANT_CONFIGS:
            raise ValueError(f"Variant '{variant}' is not supported. Choose from: {list(OACIRR_VARIANT_CONFIGS.keys())}")

        self.config = OACIRR_VARIANT_CONFIGS[variant]
        self.img_root = self.data_root / self.config['img_dir']
        self.anno_root = self.data_root / self.config['anno_dir']

        self.text_entity = text_entity
        self.prompts = ["Same {entity}", "With the same {entity}", "Identical {entity}", "{entity} unchanged",
                        "Preserving the {entity}", "Invariant {entity}", "Keep the {entity}", "Fixed {entity}"]
        self.num_prompts = len(self.prompts)

        self.bounding_box_width = bounding_box_width
        self.bounding_box_color = bounding_box_color
        self.bounding_box_crop = bounding_box_crop
        self.highlight_training = highlight_training
        self.highlight_inference = highlight_inference

        self.pad_ratio = None
        self.pad_size = None

        if hasattr(self.preprocess, 'transforms'):
            for t in self.preprocess.transforms:
                if type(t).__name__ == 'TargetPad':
                    self.pad_ratio = t.target_ratio
                    self.pad_size = t.size
                    break
                elif type(t).__name__ == 'SquarePad':
                    self.pad_ratio = 1.0
                    self.pad_size = t.size
                    break

        if self.pad_ratio is None:
            self.pad_ratio = 1.0
            self.pad_size = 224

        if split not in ['train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in['relative', 'classic']")

        # Load annotations dynamically based on variant and split
        with open(self.anno_root / 'quadruple_captions' / f'caption_full.{split}.json') as f:
            self.quadruples = json.load(f)

        with open(self.anno_root / 'image_splits' / f'split.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        with open(self.anno_root / 'image_bounding_box' / f'bounding_box.{split}.json') as f:
            self.name_to_bounding_box = json.load(f)

        print(f"OACIRR [{variant}] {split} dataset in {mode} mode initialized.")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                quadruple = self.quadruples[index]
                reference_name = quadruple['reference']
                target_name = quadruple['target']
                modification_text = quadruple['modification_text_mllm']

                if self.text_entity:
                    entity = quadruple['object_category']
                    prompt_index = index % self.num_prompts
                    final_prompt = self.prompts[prompt_index].format(entity=entity)
                    modification_text = f"{final_prompt}, {modification_text}"

                if self.split == 'train':
                    reference_image_path = self.img_root / self.name_to_relpath[reference_name]
                    raw_reference_image = PIL.Image.open(reference_image_path).convert("RGB")

                    if self.bounding_box_width:
                        bounding_box = quadruple['reference_bounding_box']
                        bounding_box_draw = ImageDraw.Draw(raw_reference_image)
                        bounding_box_draw.rectangle(bounding_box, outline=self.bounding_box_color, width=self.bounding_box_width)

                    reference_image = self.preprocess(raw_reference_image)
                    target_image_path = self.img_root / self.name_to_relpath[target_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path)).convert("RGB")

                    reference_bbox = None
                    if self.highlight_training:
                        bounding_box = quadruple['reference_bounding_box']
                        original_w, original_h = raw_reference_image.size
                        reference_bbox = transform_bbox_targetpad(
                            bbox=bounding_box,
                            original_size=(original_w, original_h),
                            target_ratio=self.pad_ratio,
                            final_size=self.pad_size
                        )

                    return reference_image, target_image, reference_name, target_name, modification_text, reference_bbox

                elif self.split == 'val':
                    group_names = quadruple.get('target_subset_vitG', None)
                    reference_bbox = None

                    if self.highlight_inference:
                        bounding_box = quadruple['reference_bounding_box']
                        reference_image_path = self.img_root / self.name_to_relpath[reference_name]
                        raw_reference_image = PIL.Image.open(reference_image_path).convert("RGB")
                        original_w, original_h = raw_reference_image.size

                        reference_bbox = transform_bbox_targetpad(
                            bbox=bounding_box,
                            original_size=(original_w, original_h),
                            target_ratio=self.pad_ratio,
                            final_size=self.pad_size
                        )

                    if group_names is not None:
                        return reference_name, target_name, modification_text, group_names, reference_bbox
                    else:
                        return reference_name, target_name, modification_text, reference_bbox

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.img_root / self.name_to_relpath[image_name]
                raw_image = PIL.Image.open(image_path).convert("RGB")

                if self.bounding_box_width:
                    bounding_box = self.name_to_bounding_box[image_name]
                    bounding_box_draw = ImageDraw.Draw(raw_image)
                    bounding_box_draw.rectangle(bounding_box, outline=self.bounding_box_color, width=self.bounding_box_width)
                    image = self.preprocess(raw_image)
                elif self.bounding_box_crop:
                    bounding_box = self.name_to_bounding_box[image_name]
                    cropped_image = raw_image.crop(bounding_box)
                    image = self.preprocess(cropped_image)
                else:
                    image = self.preprocess(raw_image)

                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception at index {index}: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.quadruples)
        elif self.mode == 'classic':
            if self.bounding_box_width or self.bounding_box_crop:
                return len(self.name_to_bounding_box)
            else:
                return len(self.name_to_relpath)


# ==========================================================================================
# Standard CIR Benchmark Datasets
# ==========================================================================================

class FashionIQDataset(Dataset):
    """
        ****************************************************************************************
        * [MODIFICATION NOTICE for OACIR Codebase]: 
        * To ensure cross-platform compatibility and avoid hardcoded absolute paths,
        * the 'data_root' parameter MUST be explicitly provided during instantiation.
        * Example: dataset = FashionIQDataset(data_root="./Datasets/FashionIQ", split="val", ...)
        ****************************************************************************************

        The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions, reference_name, target_name) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
        The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, data_root: str, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
            :param data_root: path to the FashionIQ dataset root directory
            :param split: dataset split, should be in ['test', 'train', 'val']
            :param dress_types: list of fashionIQ category
            :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield tuples made of (image_name, image)
                - In 'relative' mode the dataset yield tuples made of:
                    - (reference_image, target_image, image_captions, reference_name, target_name) when split == train
                    - (reference_name, target_name, image_captions) when split == val
                    - (reference_name, reference_image, image_captions) when split == test
            :param preprocess: function which preprocesses the image
        """
        self.data_path = Path(data_root)
        self.split = split
        self.dress_types = dress_types
        self.mode = mode
        self.preprocess = preprocess

        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.data_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.data_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = self.data_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert("RGB"))
                    target_name = self.triplets[index]['target']
                    target_image_path = self.data_path / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert("RGB"))
                    return reference_image, target_image, image_captions, reference_name, target_name

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = self.data_path / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert("RGB"))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.data_path / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path).convert("RGB"))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
        ****************************************************************************************
        * [MODIFICATION NOTICE for OACIR Codebase]: 
        * To ensure cross-platform compatibility and avoid hardcoded absolute paths,
        * the `data_root` parameter MUST be explicitly provided during instantiation.
        * Example: dataset = CIRRDataset(data_root="./Datasets/CIRR", split="val", ...)
        ****************************************************************************************

        The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, rel_caption, reference_name, target_hard_name, group_images) when split == train
            - (reference_name, target_name, rel_caption, group_members) when split == val
            - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, data_root: str, split: str, mode: str, preprocess: callable):
        """
            :param data_root: path to the CIRR dataset root directory
            :param split: dataset split, should be in ['test', 'train', 'val']
            :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield tuples made of (image_name, image)
                - In 'relative' mode the dataset yield tuples made of:
                    - (reference_image, target_image, rel_caption, reference_name, target_hard_name, group_images) when split == train
                    - (reference_name, target_name, rel_caption, group_members) when split == val
                    - (pair_id, reference_name, rel_caption, group_members) when split == test1
            :param preprocess: function which preprocesses the image
        """
        self.data_path = Path(data_root)
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(self.data_path / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(self.data_path / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = self.data_path / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert("RGB"))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.data_path / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path).convert("RGB"))

                    # Delete the target image from the group members
                    target_mask = group_members != np.repeat(target_hard_name, len(group_members))
                    group_names = np.array(group_members)[target_mask]

                    # Load group images
                    group_images_list = []
                    for group_name in group_names:
                        group_image_path = self.data_path / self.name_to_relpath[group_name]
                        group_image = self.preprocess(PIL.Image.open(group_image_path).convert("RGB"))
                        group_images_list.append(group_image)
                    group_images = torch.stack(group_images_list)

                    return reference_image, target_image, rel_caption, reference_name, target_hard_name, group_images

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.data_path / self.name_to_relpath[image_name]
                image = self.preprocess(PIL.Image.open(image_path).convert("RGB"))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")



# ==========================================================================================
# Additional CIR / Cross-Task Generalization Datasets
# ==========================================================================================

class Ipr2prDataset(Dataset):
    """
        Ipr2pr dataset class which manages Ipr2pr data (derived from MGIE)
        The dataset yield tuples made of (reference_image, target_image, modified_text)
    """

    def __init__(self, data_root: str, preprocess: callable, patch_size: int, mask_ratio: float):
        """
            :param data_root: path to the Ipr2pr dataset root directory
            :param preprocess: function which preprocesses the image
            :param patch_size: Size of each image patch (assuming square patches)
            :param mask_ratio: Ratio of image patches to be masked (between 0 and 1)
        """
        self.data_root = Path(data_root)
        self.preprocess = preprocess
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # get triplets made by (reference_image, target_image, modified_text)
        with open(self.data_root / f'mgie_triplets.json') as f:
            self.triplets = json.load(f)

        if mask_ratio:
            print(f"Ipr2pr dataset with mask ratio {mask_ratio} initialized")
        else:
            print(f"Ipr2pr dataset initialized")

    def __getitem__(self, index):
        try:
            reference_name = self.triplets[index]['reference']
            target_name = self.triplets[index]['target']
            modified_text = self.triplets[index]['text']

            reference_image_path = self.data_root / 'mgie_dataset' / f"{reference_name}.jpg"
            reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert("RGB"))

            target_image_path = self.data_root / 'mgie_dataset' / f"{target_name}.jpg"
            target_image = self.preprocess(PIL.Image.open(target_image_path).convert("RGB"))

            if self.mask_ratio:
                reference_image, target_image = mask_image(
                    reference_image, target_image, patch_size=self.patch_size, mask_ratio=self.mask_ratio
                )

            return reference_image, target_image, modified_text

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.triplets)


class CIRCODataset(Dataset):
    """
        CIRCO dataset class which manages CIRCO data
        The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_id, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, relative_caption, shared_concept, reference_img_id, target_img_id, gt_img_ids, query_id) when split == val
            - (reference_image, relative_caption, shared_concept, reference_img_id, query_id) when split == test
    """

    def __init__(self, data_root: Union[str, Path], split: Literal['val', 'test'], mode: Literal['relative', 'classic'], preprocess: callable):
        """
            :param data_root: path to the CIRCO dataset root directory
            :param split: dataset split, should be in ['val', 'test']
            :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield tuples made of (image_id, image)
                - In 'relative' mode the dataset yield tuples made of:
                    - (reference_image, target_image, relative_caption, shared_concept, reference_img_id, target_img_id, gt_img_ids, query_id) when split == val
                    - (reference_image, relative_caption, shared_concept, reference_img_id, query_id) when split == test
            :param preprocess: function which preprocesses the image
        """
        self.data_path = Path(data_root)
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        with open(self.data_path / "COCO2017_unlabeled" / "annotations" / "image_info_unlabeled2017.json", "r") as f:    # Load COCO images information
            imgs_info = json.load(f)

        self.img_paths = [self.data_path  / "COCO2017_unlabeled" / "unlabeled2017" / img_info["file_name"] for img_info in imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        with open(self.data_path / 'annotations' / f'{split}.json', "r") as f:    # get CIRCO annotations
            self.annotations: List[dict] = json.load(f)

        self.max_num_gts = 23    # Get maximum number of ground truth images (for padding when loading the images)

        print(f"CIRCO {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        if self.mode == 'relative':
            query_id = str(self.annotations[index]['id'])
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path).convert("RGB"))

            if self.split == 'val':
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path).convert("RGB"))

                # Pad ground truth image IDs with empty strings for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]
            img = self.preprocess(PIL.Image.open(img_path).convert("RGB"))
            
            return img_id, img

        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class COCODataset(Dataset):
    """
        Base dataset class for GeneCIS (COCO)
    """
    def __init__(self, root_dir: str, preprocess: callable) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.preprocess = preprocess

    def load_sample(self, sample):
        val_img_id = sample['val_image_id']
        fpath = self.root_dir / f'{val_img_id:012d}.jpg'
        image = self.preprocess(PIL.Image.open(fpath).convert("RGB"))
        return image


class COCOValSubset(COCODataset):
    """
        Validation Subset class for GeneCIS (COCO)
    """
    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            self.val_samples = json.load(f)

    def __getitem__(self, index):
        sample = self.val_samples[index]
        reference = sample['reference']
        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        return reference, caption, gallery_and_target, 0  

    def __len__(self):
        return len(self.val_samples)


class VAWDataset(Dataset):
    """
        Base dataset class for GeneCIS (VAW)
    """
    def __init__(self, root_dir: str, preprocess: callable) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.preprocess = preprocess

    def load_cropped_image(self, img):

        image_id = img['image_id']
        path = self.root_dir / f'{image_id}.jpg'
        image = self.preprocess(PIL.Image.open(path).convert("RGB"))
        return image


class VAWValSubset(VAWDataset):
    """
        Validation Subset class for GeneCIS (VAW)
    """
    def __init__(self, val_split_path: str, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            self.val_samples = json.load(f)

    def __getitem__(self, index):
        sample = self.val_samples[index]
        reference = sample['reference']
        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_cropped_image(i) for i in (reference, target)]
        gallery = [self.load_cropped_image(i) for i in gallery]

        if self.preprocess is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        return reference, caption, gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)
