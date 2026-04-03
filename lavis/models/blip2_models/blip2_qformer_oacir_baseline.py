"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)


def bbox_to_patch_mask(bboxes: list, image_size=224, patch_size=14, device='cpu') -> torch.Tensor:
    """
        Convert a list of bounding boxes into a 1D patch mask for Q-Former attention bias.
    """
    batch_masks = []
    num_patches_per_side = image_size // patch_size
    num_total_patches = 1 + num_patches_per_side**2

    for bbox in bboxes:
        mask_1d = torch.zeros(num_total_patches, device=device)
        if bbox is None:
            batch_masks.append(mask_1d)
            continue

        # Convert the pixel coordinates to the patch grid coordinates
        patch_x1 = max(0, bbox[0] // patch_size)
        patch_y1 = max(0, bbox[1] // patch_size)
        patch_x2 = min(num_patches_per_side, (bbox[2] + patch_size - 1) // patch_size)
        patch_y2 = min(num_patches_per_side, (bbox[3] + patch_size - 1) // patch_size)

        for i in range(patch_y1, patch_y2):
            for j in range(patch_x1, patch_x2):
                patch_index = i * num_patches_per_side + j + 1
                if patch_index < num_total_patches:
                    mask_1d[patch_index] = 1.0
        batch_masks.append(mask_1d)

    return torch.stack(batch_masks)


@registry.register_model("oacir_baseline")
class Blip2QformerOacirBaseline(Blip2Base):
    """
    Standard & Fixed-Beta CIR Baseline for the OACIR task.

    Supported model types:
        - pretrain: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("oacir_baseline", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",    # Default ViT-G Model
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",    # ViT-L Model
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        fixed_beta=0.0
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # Frozen Image Encoder E_I
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        # Multimodal Encoder E_M
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # Contrastive Alignment Head (Linear_I and Linear_M)
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.fixed_beta = fixed_beta
        self.num_query_token = num_query_token

        # self.patch_size = self.visual_encoder.conv1.kernel_size[0]    # For Vit-L
        self.patch_size = self.visual_encoder.patch_embed.patch_size[0]    # For ViT-G


    def forward(self, samples):
        reference_image = samples["reference_image"]
        modification_text = samples["modification_text"]
        target_image = samples["target_image"]
        reference_bbox = samples.get("reference_bbox", None)

        # Determine whether to apply Fixed Beta attention highlighting
        is_highlighting = reference_bbox is not None and any(b is not None for b in reference_bbox) and self.fixed_beta > 0
        attention_bias = None

        if is_highlighting:
            batch_patch_mask = bbox_to_patch_mask(reference_bbox, patch_size=self.patch_size, device=reference_image.device)
            attention_bias = (self.fixed_beta * batch_patch_mask).unsqueeze(1).unsqueeze(1)
            # Apply Fixed Beta and format as [batch_size, 1, 1, seq_len] for Cross-Attention broadcasting

        ### =============== Query Branch =============== ###
        reference_image_embeds = self.ln_vision(self.visual_encoder(reference_image))
        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)

        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image.device)

        modification_text_tokens = self.tokenizer(
            modification_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_image.device)

        fusion_atts = torch.cat([query_atts, modification_text_tokens.attention_mask], dim=1)

        fusion_output = self.Qformer.bert(
            modification_text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=fusion_atts,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            return_dict=True,
            attention_bias=attention_bias,
        )

        # Extract [CLS] token and map to shared space via Linear_M
        fusion_features = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), dim=-1)

        ### =============== Target Branch =============== ###
        target_image_embeds = self.ln_vision(self.visual_encoder(target_image))
        target_image_atts = torch.ones(target_image_embeds.size()[:-1], dtype=torch.long).to(reference_image.device)

        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=target_image_embeds,
            encoder_attention_mask=target_image_atts,
            use_cache=True,
            return_dict=True,
        )

        target_features = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)

        ### =============== Contrastive Alignment Loss =============== ###
        sim_f2t = torch.matmul(
            fusion_features.unsqueeze(1).unsqueeze(1), target_features.permute(0, 2, 1)
        ).squeeze()

        sim_f2t, _ = sim_f2t.max(-1)
        sim_f2t = sim_f2t / self.temp

        bs = reference_image.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=torch.long).to(reference_image.device)
        loss_align = F.cross_entropy(sim_f2t, targets)

        return {'loss_align': loss_align}


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_features = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        return image_features, image_embeds_frozen


    @torch.no_grad()
    def inference(self, reference_image_embeds, target_features, modification_text, reference_bbox=None):
        is_highlighting = reference_bbox is not None and any(b is not None for b in reference_bbox) and self.fixed_beta > 0
        attention_bias = None

        if is_highlighting:
            batch_patch_mask = bbox_to_patch_mask(reference_bbox, patch_size=self.patch_size, device=reference_image_embeds.device)
            attention_bias = (self.fixed_beta * batch_patch_mask).unsqueeze(1).unsqueeze(1)
            
        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(reference_image_embeds.device)
        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image_embeds.device)

        modification_text_tokens = self.tokenizer(
            modification_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_image_embeds.device)

        fusion_atts = torch.cat([query_atts, modification_text_tokens.attention_mask], dim=1)

        fusion_output = self.Qformer.bert(
            modification_text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=fusion_atts,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            return_dict=True,
            attention_bias=attention_bias,
        )

        fusion_features = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), dim=-1)
        sim_f2t = torch.matmul(fusion_features.unsqueeze(1).unsqueeze(1), target_features.permute(0, 2, 1)).squeeze()
        sim_f2t, _ = sim_f2t.max(-1)

        return sim_f2t


    @torch.no_grad()
    def inference_features(self, reference_image_embeds, modification_text, reference_bbox=None):
        is_highlighting = reference_bbox is not None and any(b is not None for b in reference_bbox) and self.fixed_beta > 0
        attention_bias = None

        if is_highlighting:
            batch_patch_mask = bbox_to_patch_mask(reference_bbox, patch_size=self.patch_size, device=reference_image_embeds.device)
            attention_bias = (self.fixed_beta * batch_patch_mask).unsqueeze(1).unsqueeze(1)
            
        reference_image_atts = torch.ones(reference_image_embeds.size()[:-1], dtype=torch.long).to(reference_image_embeds.device)
        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(reference_image_embeds.device)

        modification_text_tokens = self.tokenizer(
            modification_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_image_embeds.device)

        fusion_atts = torch.cat([query_atts, modification_text_tokens.attention_mask], dim=1)

        fusion_output = self.Qformer.bert(
            modification_text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=fusion_atts,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            return_dict=True,
            attention_bias=attention_bias,
        )

        fusion_features = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]), dim=-1)

        return fusion_features


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
