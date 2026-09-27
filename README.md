# 3D-MedSAM2-LoRA

Adapting Meta's Segment Anything Model 2(SAM 2) video predictor for 3D volumetric medical segmentation on abdominal CT scans(BTCV dataset).

By mapping the 3D volume's z-axis(depth slices) as sequential video frames, SAM 2's streaming memory mechanism tracks and segments continuous organs across slices. We use LoRA and Space-Depth adapters to fine-tune the model efficiently without retraining the full backbone.

## Setup

1. **Clone the repository:**

   git clone https://github.com/scamy-io/3D-MedSAM2-LoRA.git
   cd 3D-MedSAM2-LoRA


2. **Install dependencies:**

   pip install -r requirements.txt


3. **Data Preparation:**
   Place the BTCV NIfTI files into the data/` folder:
   - CT volumes: data/imagesTr/img0001.nii.gz ... img0040.nii.gz
   - Ground truth labels: data/labelsTr/label0001.nii.gz ... label0040.nii.gz`

---

## How to Run

### 1. Preprocessing
Preprocesses raw NIfTI files with soft-tissue HU windowing [-150, 250] and extracts normalized axial PNG slices

python scripts/01_preprocess.py


### 2. Training (LoRA + SD-Adapters)
Fine-tune on a specific target organ (e.g. Organ 6 = Liver,11 = Pancreas)

python scripts/03_train_lora.py --organ 6 --sd-adapter

Checkpoints are saved under checkpoints/`.

### 3. Inference
Run bidirectional tracking on a validation case using a trained LoRA checkpoint

python scripts/02_run_inference.py --case img0035 --organ 6 --lora checkpoints/lora_organ6.pt


### 4. Validation Evaluation
Compute DSC, HD95, and VPE

python scripts/04_evaluate.py --organ 6 --lora checkpoints/lora_organ6.pt


## Method Summary

- **Depth as Time**: Axial CT slices are streamed through SAM 2's video memory module.
- **Dual-Memory Bank**: Short-term memory (preceding 3 slices) captures local anatomical continuity, while long-term memory anchors to the initial prompt slice to prevent drift.
- **Heuristic Start & Early Halting**: Automatically initializes on the slice with the maximum organ area, propagating bidirectionally, and halts when cross-sectional area falls below threshold.
- **Efficient Fine-Tuning**: LoRA (rank 8, alpha 16) targeting attention layers combined with Space-Depth transpose adapters (d=64).
- **Loss**: Combined Volumetric Soft Dice + Binary Cross-Entropy loss.
