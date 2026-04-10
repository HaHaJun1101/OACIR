<div align="center">
  <img src='assets/OACIR_logo.png' style="height:120px" alt="OACIR Logo"></img>
</div>

<h2 align="center"><strong>Beyond Semantic Search:
  <br>
  Towards Referential Anchoring in Composed Image Retrieval</strong></h2>

<p align="center">
  <span style="font-size: 16px;">
    Yuxin Yang<sup>1,2</sup>, &nbsp;
    Yinan Zhou<sup>3,4</sup>, &nbsp;
    Yuxin Chen<sup>4</sup>, &nbsp;
    Ziqi Zhang<sup>1</sup>, &nbsp;
    Zongyang Ma<sup>1</sup>,<br>
    Chunfeng Yuan<sup>1,2</sup>, &nbsp;
    Bing Li<sup>1,2,5</sup>, &nbsp;
    Jun Gao<sup>6</sup>, &nbsp;
    Weiming Hu<sup>1,2,7</sup>
  </span>
  <br><br>
  <span style="font-size: 15px; font-style: italic; color: #555;">
    <sup>1</sup> Institute of Automation, Chinese Academy of Sciences (CASIA) <br>
    <sup>2</sup> University of Chinese Academy of Sciences (UCAS) <br>
    <sup>3</sup> Xi'an Jiaotong University (XJTU) &nbsp;&nbsp;&nbsp;
    <sup>4</sup> Tencent &nbsp;&nbsp;&nbsp;
    <sup>5</sup> PeopleAI &nbsp;&nbsp;&nbsp;
    <sup>6</sup> HelloGroup &nbsp;&nbsp;&nbsp;
    <sup>7</sup> ShanghaiTech University
  </span>
</p>

<h4 align="center">🎉 <b>Accepted by CVPR 2026 (Highlight)</b> 🎉</h4>

<br>

<div align="center">
  <a href='https://arxiv.org/abs/2604.05393'><img src='https://img.shields.io/badge/Paper-arXiv:2604.05393-B31B1B.svg?logo=arxiv&logoColor=white'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://hahajun1101.github.io/OACIR/'><img src='https://img.shields.io/badge/Project-Page-FF9900.svg?logo=Google-Chrome&logoColor=white'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://huggingface.co/datasets/HaHaJun1101/OACIRR'><img src='https://img.shields.io/badge/Dataset-OACIRR-0070D0.svg?logo=HuggingFace&logoColor=white'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href='https://huggingface.co/HaHaJun1101/AdaFocal'><img src='https://img.shields.io/badge/Weights-AdaFocal-1FA050.svg?logo=HuggingFace&logoColor=white'></a>
  <br>
  <br>
</div>

<p align="center">
  <img src="assets/OACIR_task_overview.png" width="95%" alt="OACIR Task Overview"/>
</p>

> Traditional Composed Image Retrieval (CIR) enables flexible multi-modal search but inherently prioritizes broad semantic matching, often failing to retrieve a user-specified instance across contexts. To bridge this gap, we introduce **Object-Anchored Composed Image Retrieval (OACIR)**, a novel fine-grained retrieval task that mandates strict *instance-level consistency*. To advance research on this challenging task, we construct **OACIRR**, the first large-scale, multi-domain benchmark comprising over 160K quadruples. We further propose **AdaFocal**, an efficient framework featuring a Context-Aware Attention Modulator (CAAM) that dynamically intensifies attention on the anchored instance region.

<br>

<div align="center">
  ⭐ <b>If you find our OACIR codebase helpful to your research, please give this repo a star! ⭐ <br>
  💖 It really encourages our open-source journey. Thank you! 💖</b>
</div>

<br>

---

## 🔔 News
- **🌟 [2026-04-09]: Our paper has been selected as a ✨*Highlight*✨ at CVPR 2026!**
- **🔥 [2026-04-07]: The *AdaFocal* model checkpoints are officially released on HuggingFace!**
- **🔥 [2026-04-03]: The full Training/Evaluation code are officially released and is now available for use!**
- **🔥 [2026-03-25]: The OACIRR Benchmark is officially released on HuggingFace!**
- **🎉 [2026-02-21]: Our paper "Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval" has been accepted to CVPR 2026!**

---

## 💡 Dataset Overview

**OACIRR** ( **O**bject-**A**nchored **C**omposed **I**mage **R**etrieval on **R**eal-world images ) is the first large-scale, multi-domain benchmark tailored for the OACIR task.

Unlike traditional Composed Image Retrieval (CIR), which inherently prioritizes broad semantic matching, **OACIRR** mandates strict **instance-level fidelity**. By anchoring a specific object via a bounding box in the reference image, it requires models to retrieve a target image that semantically satisfies the textual modification while **strictly preserving the identical anchored instance**.

**OACIRR** comprises a **unified training set** of **127K quadruples** covering **2,647 instances**, along with an extensive **evaluation benchmark** containing **33.4K queries** across **1,238 instances** from four diverse domains: **<font color=#990000>Fashion</font>, <font color=#CC3300>Car</font>, <font color=#003399>Product</font>, and <font color=#006633>Landmark</font>.** The benchmark is enriched with over **26.6K curated distractor instances** to form challenging galleries.

**Collectively, OACIRR encompasses 160K+ quadruples, providing both a high-quality foundational dataset and a rigorous, comprehensive benchmark for the OACIR task.**

<p align="center">
  <img src="assets/OACIRR_statistics.png" width="95%" alt="OACIRR Dataset Statistics"/>
</p>

---

## ⚙️ AdaFocal Framework

To address the core challenges of the OACIR task, we propose **AdaFocal**, an effective framework that dynamically modulates visual attention for precise, instance-level retrieval. Our approach augments a multimodal fusion backbone with a lightweight **Context-Aware Attention Modulator (CAAM)**, enabling a nuanced balance between instance fidelity and compositional reasoning.

<p align="center">
  <img src="assets/AdaFocal_framework.png" width="92%" alt="AdaFocal Framework Overview"/>
</p>

Specifically, **AdaFocal** employs a two-stage reasoning process: *Contextual Perception* and *Adaptive Focus*. It first perceives the query's compositional context to predict a modulation scalar ($\beta$). This learned signal then drives an Attention Activation Mechanism, which explicitly and adaptively intensifies the visual focus on the user-specified instance region ( provided via bounding box ) during multimodal feature fusion.

By dynamically re-weighting the attention distribution, **AdaFocal** seamlessly synthesizes the anchored instance, the global visual scene, and the textual modification into a coherent representation, establishing a robust and flexible baseline for identity-preserving retrieval.

---

## 🏆 Benchmark & Results

Our extensive evaluation demonstrates that the **OACIR** task presents a profound challenge to existing models. While current Universal Multimodal Retrieval (UMR) and Composed Image Retrieval (CIR) paradigms struggle with instance-level fidelity, our proposed **AdaFocal** establishes a robust and effective baseline.

### 📈 Benchmark Results

<div align="center">
  <table>
    <thead>
      <tr style="background-color: #f2f2f2;">
        <th rowspan="2" align="center" valign="middle">Domain</th>
        <th rowspan="2" align="center" valign="middle">Method</th>
        <th rowspan="2" align="center" valign="middle">Pretraining Data</th>
        <th colspan="3" align="center"><font color="#990000">Fashion</font></th>
        <th colspan="3" align="center"><font color="#CC3300">Car</font></th>
        <th colspan="3" align="center"><font color="#003399">Product</font></th>
        <th colspan="3" align="center"><font color="#006633">Landmark</font></th>
        <th rowspan="2" align="center" valign="middle">Avg.</th>
      </tr>
      <tr style="background-color: #f2f2f2;">
        <th align="center">R<sub>ID</sub>@1</th><th align="center">R@1</th><th align="center">R@5</th>
        <th align="center">R<sub>ID</sub>@1</th><th align="center">R@1</th><th align="center">R@5</th>
        <th align="center">R<sub>ID</sub>@1</th><th align="center">R@1</th><th align="center">R@5</th>
        <th align="center">R<sub>ID</sub>@1</th><th align="center">R@1</th><th align="center">R@5</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="6" align="center" valign="middle"><b>UMR</b></td>
        <td>UniIR-CLIP<sub>SF</sub></td>
        <td align="center">M-BEIR</td>
        <td align="center">17.33</td><td align="center">12.26</td><td align="center">24.76</td>
        <td align="center">32.67</td><td align="center">16.95</td><td align="center">41.89</td>
        <td align="center">33.71</td><td align="center">18.22</td><td align="center">40.10</td>
        <td align="center">29.47</td><td align="center">15.51</td><td align="center">43.24</td>
        <td align="center">27.18</td>
      </tr>
      <tr>
        <td>UniIR-BLIP<sub>FF</sub></td>
        <td align="center">M-BEIR</td>
        <td align="center">28.53</td><td align="center">22.41</td><td align="center">39.63</td>
        <td align="center">37.21</td><td align="center">19.97</td><td align="center">46.51</td>
        <td align="center">37.76</td><td align="center">20.98</td><td align="center">43.19</td>
        <td align="center">31.71</td><td align="center">17.14</td><td align="center">52.12</td>
        <td align="center">33.10</td>
      </tr>
      <tr>
        <td>LamRA-Ret</td>
        <td align="center">M-BEIR+NLI</td>
        <td align="center">27.45</td><td align="center">21.63</td><td align="center">37.10</td>
        <td align="center">61.03</td><td align="center">35.44</td><td align="center">74.51</td>
        <td align="center">69.45</td><td align="center">39.53</td><td align="center">70.25</td>
        <td align="center">58.64</td><td align="center">32.58</td><td align="center">68.74</td>
        <td align="center">49.70</td>
      </tr>
      <tr>
        <td>MM-Embed</td>
        <td align="center">M-BEIR+MTEB</td>
        <td align="center">41.38</td><td align="center">34.55</td><td align="center">52.50</td>
        <td align="center">53.21</td><td align="center">30.06</td><td align="center">62.80</td>
        <td align="center">71.03</td><td align="center">41.47</td><td align="center">71.15</td>
        <td align="center">78.85</td><td align="center">38.88</td><td align="center">79.32</td>
        <td align="center">54.60</td>
      </tr>
      <tr>
        <td>GME (2B)</td>
        <td rowspan="2" align="center" valign="middle">UMRB</td>
        <td align="center">38.13</td><td align="center">32.14</td><td align="center">51.50</td>
        <td align="center">58.84</td><td align="center">31.60</td><td align="center">66.03</td>
        <td align="center">76.89</td><td align="center">44.11</td><td align="center">74.20</td>
        <td align="center">73.86</td><td align="center">38.99</td><td align="center">75.61</td>
        <td align="center">55.16</td>
      </tr>
      <tr>
        <td>GME (7B)</td>
        <td align="center">44.98</td><td align="center">39.24</td><td align="center">60.18</td>
        <td align="center">63.11</td><td align="center">38.34</td><td align="center">75.38</td>
        <td align="center">83.44</td><td align="center">54.60</td><td align="center">84.15</td>
        <td align="center">77.11</td><td align="center">47.09</td><td align="center">82.69</td>
        <td align="center">62.53</td>
      </tr>
      <tr>
        <td rowspan="2" align="center" valign="middle"><b>CIR</b></td>
        <td>SPRC (ViT-G)</td>
        <td align="center">CIRR</td>
        <td align="center">28.62</td><td align="center">25.79</td><td align="center">44.48</td>
        <td align="center">25.13</td><td align="center">15.92</td><td align="center">37.06</td>
        <td align="center">54.39</td><td align="center">34.85</td><td align="center">62.31</td>
        <td align="center">40.41</td><td align="center">26.29</td><td align="center">52.39</td>
        <td align="center">37.30</td>
      </tr>
      <tr>
        <td>SPRC (ViT-G)</td>
        <td align="center"><b>OACIRR (Ours)</b></td>
        <td align="center">65.25</td><td align="center">58.51</td><td align="center">80.89</td>
        <td align="center">72.87</td><td align="center">49.82</td><td align="center">89.57</td>
        <td align="center">86.05</td><td align="center">70.61</td><td align="center">93.68</td>
        <td align="center">76.32</td><td align="center">56.04</td><td align="center">89.00</td>
        <td align="center">74.05</td>
      </tr>
      <tr>
        <td rowspan="2" align="center" valign="middle"><b>OACIR</b></td>
        <td>Baseline (ViT-G)</td>
        <td align="center" rowspan="3" valign="middle"><b>OACIRR (Ours)</b></td>
        <td align="center">69.07</td><td align="center">58.76</td><td align="center">81.44</td>
        <td align="center">74.59</td><td align="center">49.78</td><td align="center">89.46</td>
        <td align="center">87.48</td><td align="center">69.53</td><td align="center">93.66</td>
        <td align="center">79.80</td><td align="center">55.49</td><td align="center">89.87</td>
        <td align="center">74.91</td>
      </tr>
      <tr>
        <td><b>AdaFocal (ViT-G)</b></td>
        <td align="center"><b>77.15</b></td><td align="center"><b>65.31</b></td><td align="center"><b>86.88</b></td>
        <td align="center"><b>78.42</b></td><td align="center"><b>53.63</b></td><td align="center"><b>92.22</b></td>
        <td align="center"><b>91.86</b></td><td align="center"><b>74.11</b></td><td align="center"><b>95.39</b></td>
        <td align="center"><b>82.92</b></td><td align="center"><b>58.47</b></td><td align="center"><b>91.63</b></td>
        <td align="center"><b>79.00</b></td>
      </tr>
    </tbody>
  </table>
</div>
<br>

### 📉 Open-Source Checkpoints & Reproducible Performance

> **A Note on the Open-Source Models:**
> The benchmark results reported in the paper reflect the absolute peak performance achieved through extensive hyperparameter search and complex joint-training strategies across varying data splits. For this open-source release, we prioritize a **clean, unified, and highly reproducible codebase**. The checkpoints provided below are trained using our standardized pipeline exclusively on the `OACIRR Union` set. This streamlined approach results in a negligible performance variance (~0.5%) while offering a much more elegant and accessible foundation for the community to build upon.

We provide two variants of the **AdaFocal** weights on HuggingFace. You can instantly evaluate them using our provided `evaluate.sh` script.

| Model Variant | Component Type | R<sub>ID</sub>@1 (Avg) | R@1 (Avg) | R@5 (Avg) | Overall Avg | Weights Download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AdaFocal (Scalar $\beta$)** | Default Configuration | 81.52 | 63.08 | 90.98 | **78.53** | [🤗 Download](https://huggingface.co/HaHaJun1101/AdaFocal/blob/main/adafocal_scalar.pt) |
| **AdaFocal (Vector $\vec{\beta}$)** | Vector Ablation | 81.99 | 63.06 | 91.35 | **78.80** | [🤗 Download](https://huggingface.co/HaHaJun1101/AdaFocal/blob/main/adafocal_vector.pt) |

---

## 🚀 Quick Start

### 🛠️ 1. Installation

**1.1 Clone the repository**
```bash
git clone https://github.com/HaHaJun1101/OACIR.git
cd OACIR
```

**1.2 Create a fresh Conda environment**
```bash
conda create -n oacir python=3.9 -y
conda activate oacir
```

**1.3 Install PyTorch & Dependencies**

*( Note: Please adjust the index-url or PyTorch version according to your machine's CUDA version. )*
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

---

### 🗂️ 2. Data Preparation

To fully reproduce our experiments ( including OACIRR benchmarks and cross-task generalization ), please download the datasets and organize them into a unified `./Datasets/` directory under the project root.

**2.1 Download the OACIRR Dataset**

Download **OACIRR** from [HuggingFace](https://huggingface.co/datasets/HaHaJun1101/OACIRR#downloading-the-oacirr-dataset), and unzip the compressed image files.
```bash
# Recommended: Use Git LFS
git lfs install
git clone https://huggingface.co/datasets/HaHaJun1101/OACIRR ./Datasets/OACIRR
```

**2.2 Overall Dataset Directory Structure**

Ensure your workspace matches the following structure.
*( Note: For standard CIR benchmarks like CIRR, FashionIQ, CIRCO, and GeneCIS, please refer to their official repositories for download instructions ).*
```text
OACIR/
├── assets/
├── baseline_evaluation/             <-- (Cross-task evaluation scripts for CIRCO, GeneCIS, CIRR test)
├── lavis/                           <-- (Core model architectures)
├── umr_evaluation/                  <-- (UMR model evaluation scripts)
├── train.sh & evaluate.sh           <-- (Execution scripts)
├── train.py, evaluate.py            <-- (Main pipelines)
├── data_utils.py, utils.py          <-- (Dataset & helper functions)
├── visualize_results.py             <-- (HTML qualitative visualization)
│
└── Datasets/                        <-- (Place downloaded data here)
    ├── OACIRR/
    │   ├── OACIRR-Union/            <-- (Joint Training Set & Unzipped train images)
    │   └── OACIRR-Subset/           <-- (Domain-specific Subsets & Unzipped val images)
    ├── cirr_dataset/
    ├── fashionIQ_dataset/
    ├── CIRCO/
    └── GeneCIS/
```

---

### 🔥 3. Inference & Evaluation

Our codebase provides a highly flexible evaluation pipeline. You can reproduce all the results reported in our paper.

#### 3.1 Evaluating AdaFocal Checkpoints on OACIRR
Download our pre-trained **AdaFocal** weights from [HuggingFace](https://huggingface.co/HaHaJun1101/AdaFocal).
Open `evaluate.sh` and configure the following:
- `DATASET="Fashion" (or "Car", "Product", "Landmark")`
- `MODEL_WEIGHT="/path/to/adafocal_weight.pt"`
- **For Scalar Version:** Set `MODEL_NAME="oacir_adafocal"` and *uncomment* `HIGHLIGHT_INFERENCE`.
- **For Vector Version:** Set `MODEL_NAME="oacir_adafocal_vector"` and *uncomment* `HIGHLIGHT_INFERENCE`.

Run the script:
```bash
bash evaluate.sh
```

#### 3.2 Evaluating Baselines & Ablation Settings
You can easily switch between AdaFocal and different ablated baselines by modifying variables inside `evaluate.sh`:
- **Standard Baseline:** Set `MODEL_NAME="oacir_baseline"` and *comment out* all highlighting / bbox flags.
- **Visual Anchor Baseline:** Set `MODEL_NAME="oacir_baseline"`, *uncomment* `BBOX_WIDTH=3`, and *comment out* highlighting flags.
- **ROI-Cropped Baseline:** Set `MODEL_NAME="oacir_baseline"`, *uncomment* `BBOX_CROP="--bounding-box-crop"`, and *comment out* highlighting flags.
- **Fixed $\beta$ Ablation:** Set `MODEL_NAME="oacir_baseline"`, *uncomment* `HIGHLIGHT_INFERENCE`, and manually adjust the `fixed_beta` value inside `lavis/models/blip2_models/blip2_qformer_oacir_baseline.py`.

#### 3.3 Zero-Shot Cross-Task Generalization
To evaluate the zero-shot generalizability of the OACIRR-trained model on standard CIR benchmarks:
- **For CIRR / FashionIQ:** Simply change `DATASET="CIRR"` or `DATASET="FashionIQ"` in `evaluate.sh` and run it.
- **For CIRCO / GeneCIS / CIRR Test Submission:** Navigate to the `baseline_evaluation/` directory and use the standalone scripts:
  ```bash
  cd baseline_evaluation
  
  # Evaluate on CIRCO validation set
  python evaluate_circo.py --blip-model-weight /path/to/weight.pt --ntype val
  
  # Evaluate on GeneCIS (e.g., change_attribute condition)
  python evaluate_genecis.py --blip-model-weight /path/to/weight.pt --ntype change_attribute
  ```

#### 3.4 Benchmarking Third-Party Models on OACIRR
OACIRR serves as a rigorous benchmark for existing Universal Multimodal Retrieval (UMR) and Zero-Shot CIR paradigms.
- **Ready-to-use UMR Scripts:** Inside the `umr_evaluation/` folder, we provide scripts to evaluate [MM-Embed](https://huggingface.co/nvidia/MM-Embed) and [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct).
    ```bash
    cd umr_evaluation

    # Evaluate GME-7B with Visual Anchor
    python evaluate_gme.py \
        --dataset Fashion \
        --data-root ../Datasets/OACIRR \
        --model-path Alibaba-NLP/gme-Qwen2-VL-7B-Instruct \
        --bounding-box-width 3 \
        --text-entity
    ```
- **Other Baselines:** For other prominent UMR or ZS-CIR models evaluated in our paper, please adapt our `OACIRR Dataset` to their respective codebases:
  - *UMR Models*: [UniIR](https://github.com/TIGER-AI-Lab/UniIR) | [LamRA](https://github.com/Code-kunkun/LamRA) | [U-MARVEL](https://github.com/chaxjli/U-MARVEL) | [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
  - *ZS-CIR Models*: [Pic2Word](https://github.com/google-research/composed_image_retrieval) | [SEARLE](https://github.com/miccunifi/SEARLE) | [LinCIR](https://github.com/navervision/lincir)

---

### 🌟 4. Training Your Own Model

Training from scratch, fine-tuning, or performing ablation studies is fully automated via the `train.sh` script.

**4.1 Basic Configuration**

Open `train.sh` and customize your hyperparameters. Set `DATASET="Union"` to train jointly on all 4 domains, or select a specific subset (e.g., `"Fashion"`). Adjust `LR`, `BATCH_SIZE`, and `EPOCHS` as needed.

**4.2 Flexible Initialization**

- **Train from BLIP-2:** Leave `MODEL_WEIGHT=""`. The script will automatically download the pre-trained Q-Former weights from LAVIS.
- **Transfer Learning / Resume Training:** Provide the path to a previous checkpoint (e.g., a model pre-trained on CIRR/FashionIQ) in `MODEL_WEIGHT="/path/to/checkpoint.pt"`.

**4.3 Model Selection**

Set `MODEL_NAME` to `"oacir_adafocal"` for our proposed method ( ensure `HIGHLIGHT_TRAINING` is uncommented ), or `"oacir_baseline"` for standard CIR training.

**4.4 Start Training**

Run the script:
```bash
bash train.sh
```
*( Note: If `DATASET="Union"` is selected, the script will automatically perform validation on all 4 OACIRR domains, saving the best checkpoints globally under the `./checkpoints/` directory. )*

---

### 👀 5. Qualitative Visualization

To deeply analyze the retrieval results and visually understand how **AdaFocal** overcomes semantic drift and fine-grained confusion, we provide an HTML visualization tool.

After running `evaluate.sh` with the `SAVE_RESULTS="--save-results"` flag enabled, run the visualization script:
```bash
python visualize_results.py \
  --dataset Fashion \
  --data-root ./Datasets/OACIRR/OACIRR-Subset \
  --results-file ./checkpoints/saved_results/validation_results_fashion.json \
  --top-k-display 10
```
This will generate an intuitive `retrieval_results.html` in the same directory as your JSON file, allowing you to visually compare ground truths, queries, and Top-K retrieved candidates side-by-side.

---

## ✒️ Citation

If you find our work, codebase, dataset, or models useful, please consider citing our paper:
```bibtex
@article{yang2026beyond,
  title={Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval},
  author={Yang, Yuxin and Zhou, Yinan and Chen, Yuxin and Zhang, Ziqi and Ma, Zongyang and Yuan, Chunfeng and Li, Bing and Gao, Jun and Hu, Weiming},
  journal={arXiv preprint arXiv:2604.05393},
  year={2026}
}
```

---

## 🤝 Acknowledgements

Our codebase is built upon [LAVIS](https://github.com/salesforce/LAVIS) and [SPRC](https://github.com/chunmeifeng/SPRC). We deeply thank the authors for their excellent open-source contributions!
