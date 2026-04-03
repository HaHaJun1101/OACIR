import os
import json
import base64
import PIL.Image
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser


# OACIRR Structure Configuration
OACIRR_VARIANT_CONFIGS = {
    'Fashion': {'img_dir': 'OACIRR-Fashion', 'anno_dir': 'OACIRR-Fashion/oacirr-fashion'},
    'Car': {'img_dir': 'OACIRR-Car', 'anno_dir': 'OACIRR-Car/oacirr-car'},
    'Product': {'img_dir': 'OACIRR-Product', 'anno_dir': 'OACIRR-Product/oacirr-product'},
    'Landmark': {'img_dir': 'OACIRR-Landmark', 'anno_dir': 'OACIRR-Landmark/oacirr-landmark'},
}


# --- Helper Function to generate HTML for an image (Base64 version) ---
def get_image_html_base64(image_name, image_path, alt_text="image", is_gt_target=False, is_gt_in_retrieved=False, **kwargs):
    """
        Generates an HTML <img> tag for a given image.
        Handles cases where the image might not exist.
    """
    style = f"max-width:{kwargs['image_display_width']}px; max-height:{kwargs['image_display_height']}px; margin:2px; display:block;"
    if is_gt_target:    # Ground Truth Target Image
        style += "border: 3px solid blue;"
    elif is_gt_in_retrieved:    # Retrieved Ground Truth Target Image
        style += "border: 4px solid green;"
    else:
        style += "border: 2px solid #ddd;"

    if os.path.exists(image_path):
        try:
            buffered = BytesIO()
            image = PIL.Image.open(image_path)

            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG")

            image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f'<img src="data:image/jpeg;base64,{image_str}" alt="{alt_text}" title="{image_name}" style="{style}">'

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            return f'<div style="{style} text-align:center; background-color:#f0f0f0; display:flex; align-items:center; justify-content:center;">Error<br>loading:<br>{image_name}</div>'
    else:
        return f'<div style="{style} text-align:center; background-color:#f0f0f0; display:flex; align-items:center; justify-content:center;">Image<br>not found:<br>{image_name}</div>'


# --- Main Visualization Function (generates HTML string) ---
def generate_retrieval_results_html(dataset_name, img_root, reference_names, modification_texts, target_names, sorted_index_names,
                                    name_to_relpath, top_k_display=10, num_queries_to_display=None, query_sampling_step=None, **kwargs):
    """
        Generates and displays an HTML table for image retrieval results.
    """
    if not (len(reference_names) == len(modification_texts) == len(target_names) == len(sorted_index_names)):
        raise ValueError("Input lists must have the same length.")

    html_content = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>OACIRR Retrieval Results</title>"
    html_content += "<style> table { border-collapse: collapse; width: auto; font-family: sans-serif; } th, td { padding: 8px; border: 1px solid #ddd; text-align: left; vertical-align: top; } thead tr { background-color: #f2f2f2; } img { object-fit: contain; } </style>"
    html_content += f"</head><body><h1>OACIRR [{dataset_name}] Retrieval Results</h1><table>"

    # Header Row 
    header_cols = [
        "Query ID", "Reference Image", "Modification Text", 
        "Target Image", f"Top-{top_k_display} Retrieved Images"
    ]

    html_content += "<thead><tr>"
    html_content += f"<th style='text-align: center;'>{header_cols[0]}</th>"
    html_content += f"<th style='text-align: center;'>{header_cols[1]}</th>"
    html_content += f"<th style='text-align: center; min-width: 200px;'>{header_cols[2]}</th>"
    html_content += f"<th style='text-align: center;'>{header_cols[3]}</th>"
    html_content += f"<th colspan='{top_k_display}'>{header_cols[4]}</th>"
    html_content += "</tr></thead>"
    html_content += "<tbody>"

    queries_to_process = list(range(len(reference_names)))

    if query_sampling_step is not None and query_sampling_step > 0:
        queries_to_process = queries_to_process[::query_sampling_step]
    elif num_queries_to_display is not None:
        queries_to_process = queries_to_process[:min(num_queries_to_display, len(reference_names))]

    for i in tqdm(queries_to_process, ncols=140, ascii=True, desc="Generating HTML"):

        # Query ID
        html_content += "<tr style='border-bottom: 1px solid #eee;'>"
        html_content += f"<td style='text-align: center;'><b>Q{i+1}</b></td>"

        # Reference Image
        reference_name = reference_names[i]
        reference_path = img_root / name_to_relpath[reference_name]

        html_content += "<td style='text-align: center;'>"
        html_content += f"<b>{reference_name}</b><br>"
        html_content += get_image_html_base64(
            reference_name,
            reference_path,
            alt_text=f"Reference: {reference_name}",
            image_display_width=kwargs['image_display_width'],
            image_display_height=kwargs['image_display_height']
        )
        html_content += "</td>"

        # Modification Text
        modification_text = modification_texts[i]
        html_content += f"<td style='word-wrap: break-word;'>{modification_text}</td>"

        # Ground Truth Target Image
        gt_target_name = target_names[i]
        target_path = img_root / name_to_relpath[gt_target_name]

        html_content += "<td style='text-align: center;'>"
        html_content += f"<b>{gt_target_name}</b><br>"
        html_content += get_image_html_base64(
            gt_target_name,
            target_path,
            alt_text=f"GT Target: {gt_target_name}",
            is_gt_target=True,
            image_display_width=kwargs['image_display_width'],
            image_display_height=kwargs['image_display_height']
        )
        html_content += "</td>"

        # For Top-K Retrieved Images, create individual cells for each image
        retrieved_names = sorted_index_names[i]

        for k in range(top_k_display):
            html_content += "<td style='text-align: center;'>" # Each top-k image in its own cell
            if k < len(retrieved_names):
                candidate_name = retrieved_names[k]
                candidate_path = img_root / name_to_relpath[candidate_name]
                is_correct_retrieval = (candidate_name == gt_target_name)

                html_content += f"<b>Rank {k+1}</b><br>{candidate_name}<br>"
                html_content += get_image_html_base64(
                    candidate_name,
                    candidate_path,
                    alt_text=f"Retrieved {k+1}: {candidate_name}",
                    is_gt_in_retrieved=is_correct_retrieval,
                    image_display_width=kwargs['image_display_width'],
                    image_display_height=kwargs['image_display_height']
                )
            else:
                html_content += f"Rank {k+1}<br>N/A"
            html_content += "</td>"

        html_content += "</tr>"

    html_content += "</tbody></table></body></html>"
    return html_content



if __name__ == '__main__':

    parser = ArgumentParser("Visualize OACIRR Validation Results in HTML")

    parser.add_argument("--dataset", type=str, required=True, choices=['Fashion', 'Car', 'Product', 'Landmark'], help="OACIRR subset to visualize")
    parser.add_argument("--data-root", type=str, default="./OACIRR/OACIRR-Subset", help="Root directory of the datasets")
    parser.add_argument("--results-file", type=str, required=True, help="Path to the validation_results.json file")

    parser.add_argument("--top-k-display", default=10, type=int)
    parser.add_argument("--num-queries-to-display", type=int)
    parser.add_argument("--query-sampling-step", type=int, default=None)
    parser.add_argument("--image-display-width", default=100, type=int)
    parser.add_argument("--image-display-height", default=100, type=int)

    args = parser.parse_args()


    # Dataset Paths
    config = OACIRR_VARIANT_CONFIGS[args.dataset]
    data_root = Path(args.data_root)
    img_root = data_root / config['img_dir']
    anno_root = data_root / config['anno_dir']

    # Get a mapping from image name to relative path
    with open(anno_root / 'image_splits' / 'split.val.json') as f:
        name_to_relpath = json.load(f)

    # Get validation results
    results_file_path = Path(args.results_file)
    if not results_file_path.exists():
        raise FileNotFoundError(f"Cannot find results file: {results_file_path}")
    with open(results_file_path) as f:
        validation_results = json.load(f)

    reference_names = validation_results['reference_names']
    modification_texts = validation_results['modification_texts']
    target_names = validation_results['target_names']
    sorted_index_names = validation_results['sorted_index_names']

    # Generate HTML content
    html_output = generate_retrieval_results_html(
        dataset_name=args.dataset,
        img_root=img_root,
        reference_names=reference_names,
        modification_texts=modification_texts,
        target_names=target_names,
        sorted_index_names=sorted_index_names,
        name_to_relpath=name_to_relpath,
        top_k_display=args.top_k_display,
        num_queries_to_display=args.num_queries_to_display,
        query_sampling_step=args.query_sampling_step,
        image_display_width=args.image_display_width,
        image_display_height=args.image_display_height
    )

    # Write to an HTML file in the same directory as the results JSON
    if html_output:
        output_html_path = results_file_path.with_suffix('.html')
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"Successfully generated visualization at: {output_html_path}")
    else:
        print("HTML generation failed.")
