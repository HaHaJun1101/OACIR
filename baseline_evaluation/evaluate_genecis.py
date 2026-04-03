import os
import sys
import torch
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from functools import partial

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lavis.models import load_model_and_preprocess
from data_utils import squarepad_transform, targetpad_transform, VAWValSubset, COCOValSubset
from utils import device


def get_recall(indices, targets):
    """
        Calculates the recall score for the given predictions and targets
    """
    # One hot label branch
    if len(targets.size()) == 1:
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero(as_tuple=False)
        if len(hits) == 0:
            return 0
        n_hits = (targets == indices).nonzero(as_tuple=False)[:, :-1].size(0)
        return float(n_hits) / targets.size(0)

    # Multi hot label branch
    else:
        recall = []
        for preds, gt in zip(indices, targets):
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)

            success = (preds_binary * gt_binary).sum() > 0
            recall.append(1 if success else 0)
        return torch.Tensor(recall).float().mean()


class AverageMeter(object):
    """
        Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def validate_genecis(blip_model, valloader, topk=(1, 2, 3), save_path=None, txt_processors=None):

    blip_model.eval()
    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    print(f"Computing GeneCIS validation metrics...")
    with torch.no_grad():
        for batch in tqdm(valloader, ncols=140, ascii=True):
            ref_img, captions, gallery_set, target_rank = [x for x in batch[:4]]
            ref_img = ref_img.cuda(non_blocking=True)
            gallery_set = gallery_set.cuda(non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]

            bsz, n_gallery, _, h, w = gallery_set.size()
            imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            all_img_feats0, all_img_feats = blip_model.extract_target_features(imgs_,  mode="mean")

            # L2 normalize and view into correct shapes
            _, gallery_feats = all_img_feats0.split((bsz, bsz * n_gallery), dim=0)
            ref_feats, _ = all_img_feats.split((bsz, bsz * n_gallery), dim=0)

            # Using model's inference_features logic. Assuming shape [B, M, D]
            combined_feats = blip_model.inference_features(ref_feats, captions)
            gallery_feats = gallery_feats.view(bsz, n_gallery, 32, -1)

            # Compute similarity
            similarities = combined_feats[:, None, :].unsqueeze(1) * gallery_feats 
            similarities, _ = similarities.max(dim=2)    # B x N x D
            similarities = similarities.sum(dim=-1)

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)    # B x N
            sort_idxs = sort_idxs.to(device, non_blocking=True)
            target_rank = target_rank.to(device, non_blocking=True)

            # Compute recall at K
            for k in topk:
                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)

            sims_to_save.append(similarities.cpu())

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving predictions to: {save_path}')
            torch.save(sims_to_save, save_path)

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters



if __name__ == '__main__':

    parser = ArgumentParser("Evaluate the Baseline model on the GeneCIS dataset")

    parser.add_argument("--data-root", type=str, default="../Datasets/GeneCIS", help="Path to GeneCIS dataset root")
    parser.add_argument("--blip-model-name", type=str, default="oacir_baseline", help="Model registry name")
    parser.add_argument("--blip-model-weight", type=str, required=True, help="Path to the pre-trained model weight")
    parser.add_argument("--vit-backbone", type=str, default="pretrain", help="pretrain for Vit-G, pretrain_vitL for ViT-L")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str, choices=['squarepad', 'targetpad'])
    parser.add_argument("--ntype", type=str, required=True, choices=['change_attribute', 'change_object', 'focus_attribute', 'focus_object'], help="GeneCIS condition type")

    args = parser.parse_args()


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

    # Define the validation datasets
    data_root = Path(args.data_root)
    genecis_split_path = data_root / "genecis" / f'{args.ntype}.json'

    if not genecis_split_path.exists():
        raise FileNotFoundError(f"Cannot find split file: {genecis_split_path}")

    if 'attribute' in args.ntype:
        val_dataset_subset = VAWValSubset(val_split_path=genecis_split_path, preprocess=preprocess, image_dir=data_root / "VG_100K")
    elif 'object' in args.ntype:
        val_dataset_subset = COCOValSubset(val_split_path=genecis_split_path, preprocess=preprocess, root_dir=data_root / "val2017")

    get_dataloader = partial(torch.utils.data.DataLoader, sampler=None, batch_size=32, num_workers=6, pin_memory=True, shuffle=False)
    valloader_subset = get_dataloader(dataset=val_dataset_subset)

    # Compute retrieval metrics
    validate_genecis(blip_model, valloader_subset, topk=(1, 2, 3), save_path=None, txt_processors=txt_processors)
