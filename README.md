# Machine Learning-Assisted Timing Prediction Framework for Faster Timing Closure in VLSI Physical Design

## Overview

Timing closure is one of the most challenging stages of modern VLSI physical design. Static Timing Analysis (STA) provides accurate timing information, but repeatedly analyzing large numbers of timing endpoints can become computationally expensive.

This project develops a **machine-learning-assisted timing prediction and risk-guided verification framework** for pre-routing timing analysis.

The framework uses a Graph Neural Network (GNN) based on **TimingPredict** to predict timing information from circuit graph representations. The predicted timing is then used to identify potentially high-risk endpoints. Instead of treating every endpoint equally, the framework prioritizes candidates for accurate STA-based verification.

### Proposed flow

```text
Circuit Graph
     │
     ▼
TimingPredict GNN
     │
     ▼
Timing Prediction
     │
     ▼
Predicted Slack
     │
     ▼
Risk Assessment
     │
     ▼
High-Risk Endpoint Selection
     │
     ▼
Selective STA Verification
     │
     ▼
OpenSTA
```

The main contribution of this project is the **risk-guided selective verification methodology** built on top of the existing GNN timing prediction framework.

---

## Key Result

At the selected **0.05 ns predicted-slack risk threshold**:

| Metric                              |     Result |
| ----------------------------------- | ---------: |
| Endpoint candidates evaluated       |     86,200 |
| Candidates flagged for verification |     18,164 |
| Verification workload               | **21.07%** |
| Actual timing violations            |     12,239 |
| Violations covered                  |     11,656 |
| Violation coverage                  | **95.24%** |
| Precision                           |     64.17% |
| F1-score                            |     0.7668 |

The **21.07% value represents the fraction of candidates selected for verification**. It is a verification-workload reduction proxy and should not be interpreted as a measured STA runtime reduction.

---

## Dataset

The project uses the graph-based dataset associated with the original TimingPredict framework.

The complete dataset is not included in this repository. Please follow the original TimingPredict/Dataset instructions to obtain and prepare the required data.

The evaluation uses seven test circuits:

* `jpeg_encoder`
* `usbf_device`
* `aes192`
* `xtea`
* `spm`
* `y_huff`
* `synth_ram`

The original TimingPredict dataset contains additional benchmark circuits used for training and evaluation.

---

## Model

The timing prediction model is based on the **TimingGCN** architecture from TimingPredict.

The model contains:

* Three graph convolution stages
* Signal propagation module
* Net-delay prediction
* Cell/timing feature prediction

The pretrained checkpoint used in this project is:

```text
checkpoints/08_atcd_specul/15799.pth
```

The original model implementation is retained in:

```text
model.py
```

---

## Risk-Guided Verification

The predicted timing information is converted into endpoint slack estimates.

For each timing endpoint:

```text
Predicted Slack = Required Arrival Time − Predicted Arrival Time
```

An endpoint is flagged for verification when:

```text
Predicted Slack < 0.05 ns
```

The reference violation condition is:

```text
Actual Slack < 0 ns
```

The selected threshold was obtained through a threshold sweep over the evaluation circuits.

This allows the framework to prioritize potentially critical endpoints instead of requiring every endpoint to be treated as equally high priority.

---

## OpenSTA Validation

OpenSTA is used as the accurate STA engine for validation.

The project includes a dataset-faithful OpenSTA setup for the `jpeg_encoder` circuit using:

* Sky130 standard-cell libraries
* Synthesis netlist
* SPEF parasitic data
* CTS-stage SDC constraints

The validated setup produced:

```text
Setup WNS : -14.31 ns
Setup TNS : -10288.65 ns
```

The first 100 synchronous setup paths were also extracted for further analysis.

OpenSTA version used:

```text
OpenSTA 3.1.0
```

---

## Results

### Timing prediction

The final evaluation includes:

* R²
* Mean Absolute Error (MAE)
* Median absolute error
* Maximum absolute error
* Net-delay MAE
* Endpoint vs. non-endpoint error

Representative results:

| Circuit      |     R² | MAE (ns) | Median Error (ns) | Max Error (ns) |
| ------------ | -----: | -------: | ----------------: | -------------: |
| jpeg_encoder | 0.8632 |   1.8449 |            0.9138 |       188.4875 |
| usbf_device  | 0.9360 |   0.8201 |            0.5784 |         5.6524 |
| aes192       | 0.8223 |   2.0959 |            1.0023 |        19.0469 |
| xtea         | 0.8861 |   0.7138 |            0.2515 |        48.8342 |
| spm          | 0.8505 |   0.3097 |            0.1556 |         2.4602 |
| y_huff       | 0.9255 |   0.4690 |            0.2786 |         8.3306 |
| synth_ram    | 0.8678 |   1.5774 |            0.8190 |         9.0078 |

### Risk-guided verification

The global operating point achieves:

```text
Verification workload : 21.07%
Violation coverage    : 95.24%
Precision             : 64.17%
F1-score              : 0.7668
```

This demonstrates that a relatively small subset of predicted high-risk endpoints can capture the majority of actual timing violations.

---

## Figures

The main generated figures are available under:

```text
results/figures/
```

### Timing prediction accuracy

```text
results/figures/timing_r2.png
results/figures/timing_mae.png
```

### Risk-guided verification trade-off

```text
results/figures/risk_coverage_vs_workload.png
```

The risk-coverage figure illustrates the trade-off between the percentage of endpoints selected for verification and the percentage of actual timing violations covered.

---

## Repository Structure

```text
TimingPredict/
│
├── data/
│
├── checkpoints/
│   └── 08_atcd_specul/
│       └── 15799.pth
│
├── results/
│   ├── baseline_results.txt
│   └── risk_analysis_results.txt
│
├── data_graph.py
├── data_homograph.py
├── data_stat.py
├── data_stat_cpu.py
│
├── model.py
├── train_gnn.py
├── train_deepgnn.py
├── train_stat.py
├── train_stat_rf.py
│
├── run_project.py
├── risk_analysis.py
├── risk_threshold_sweep.py
│
├── uncertainty_analysis.py
└── uncertainty_calibration.py
```

Additional project-level results are stored in:

```text
../results/
```

including:

```text
../results/project_pipeline_results.csv
../results/final_results.csv
../results/risk_global_operating_point.txt
../results/risk_threshold_sweep.txt
../results/uncertainty_calibration_results.txt
../results/figures/
../results/opensta/
```

---

## Environment

The experiments were performed using:

```text
Python       3.10
PyTorch      2.2.1 + CUDA 12.1
DGL          2.2.0 + CUDA 12.1
NumPy        1.26.4
scikit-learn
```

GPU used:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
8 GB VRAM
```

---

## Running the Original TimingPredict Evaluation

After preparing the dataset and downloading the pretrained checkpoint:

```bash
python train_gnn.py --test_iter 15799 --checkpoint 08_atcd_specul
```

This reproduces the original TimingPredict evaluation flow.

---

## Running the Project Pipeline

The complete project evaluation can be run using:

```bash
python run_project.py
```

This performs:

1. GNN inference
2. Timing prediction
3. Timing error calculation
4. Slack calculation
5. Endpoint risk identification
6. Violation detection evaluation
7. Precision/recall/F1 calculation
8. Verification workload calculation
9. Result generation

The final per-circuit results are written to:

```text
../results/project_pipeline_results.csv
```

---

## Risk Threshold Sweep

To evaluate different risk thresholds:

```bash
python risk_threshold_sweep.py
```

The resulting analysis is stored in:

```text
../results/risk_threshold_sweep.txt
```

The selected operating point is:

```text
Predicted slack threshold = 0.05 ns
```

---

## Uncertainty Analysis

The project also includes exploratory uncertainty and calibration analysis:

```bash
python uncertainty_analysis.py
```

and:

```bash
python uncertainty_calibration.py
```

These experiments investigate prediction errors and the relationship between prediction uncertainty and timing error.

The calibration results are stored in:

```text
../results/uncertainty_calibration_results.txt
```

---

## Limitations

The current implementation has several limitations:

1. The evaluation uses a pretrained GNN rather than training a new architecture from scratch.
2. The current risk score is derived from predicted slack and does not represent a probabilistically calibrated uncertainty estimate.
3. The reported 21.07% workload represents the fraction of candidates selected for verification, not a directly measured STA runtime reduction.
4. OpenSTA validation has currently been demonstrated using the dataset-faithful `jpeg_encoder` setup.
5. Prediction errors can be significant for certain endpoints, particularly in circuits with larger timing-error distributions.

---

## Future Work

Possible extensions include:

* Training the model on larger and more diverse circuit datasets.
* Developing a calibrated probabilistic risk estimator.
* Integrating selective OpenSTA verification directly into the automated flow.
* Measuring actual STA runtime savings.
* Extending validation to additional circuits and technology nodes.
* Investigating uncertainty-aware GNN architectures.
* Exploring post-routing timing prediction and timing-closure optimization.

---

## Original TimingPredict Work

This project builds upon the open-source **TimingPredict** framework:

> Zizheng Guo, Mingjie Liu, Jiaqi Gu, Shuhan Zhang, David Z. Pan, and Yibo Lin,
> *A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction*,
> Proceedings of the 59th Annual Design Automation Conference, 2022.

If you use the original TimingPredict implementation, please cite:

```bibtex
@inproceedings{mltimerdac22,
  author = {Guo, Zizheng and Liu, Mingjie and Gu, Jiaqi and Zhang, Shuhan and Pan, David Z. and Lin, Yibo},
  booktitle = {Proceedings of the 59th Annual Design Automation Conference 2022},
  organization = {ACM},
  title = {A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction},
  year = {2022}
}
```

## Project Summary

This work demonstrates a practical research pipeline in which **GNN-based timing prediction is combined with risk-guided selective verification**.

The final evaluation shows that, at the selected operating point, **21.07% of endpoint candidates are flagged while 95.24% of actual timing violations are covered**, providing a basis for reducing the amount of timing verification required while preserving high violation-detection coverage.
