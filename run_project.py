import csv
from pathlib import Path

import torch
from sklearn.metrics import r2_score

from data_graph import data_test
from model import TimingGCN


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "./checkpoints/08_atcd_specul/15799.pth"

TEST_KEYS = [
    "jpeg_encoder",
    "usbf_device",
    "aes192",
    "xtea",
    "spm",
    "y_huff",
    "synth_ram",
]

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "project_pipeline_results.csv"


# ============================================================
# Load model
# ============================================================

print("========================================")
print("TimingPredict Risk-Guided Verification")
print("========================================")

print("\nLoading pretrained TimingPredict model...")

model = TimingGCN()
model.cuda()

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location="cuda"
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Run pipeline
# ============================================================

results = []

print("\n===== END-TO-END TEST PIPELINE =====")

for test_key in TEST_KEYS:

    print("\n----------------------------------------")
    print("Circuit:", test_key)
    print("----------------------------------------")

    g, ts = data_test[test_key]

    # --------------------------------------------------------
    # One model inference only
    # --------------------------------------------------------

    with torch.no_grad():

        net_delays, _, timing_features = model(
            g,
            ts,
            groundtruth=False
        )

        pred = timing_features[:, :4]
        truth = g.ndata["n_ats"][:, :4]

    # --------------------------------------------------------
    # Timing prediction error
    # --------------------------------------------------------

    valid = torch.isfinite(pred) & torch.isfinite(truth)

    pred_valid = pred[valid]
    truth_valid = truth[valid]

    error = torch.abs(pred - truth)
    error_valid = error[valid]

    mae = error_valid.mean().item()
    median_error = error_valid.median().item()
    max_error = error_valid.max().item()

    # R2
    pred_cpu = pred_valid.detach().cpu().numpy()
    truth_cpu = truth_valid.detach().cpu().numpy()

    r2 = r2_score(truth_cpu, pred_cpu)

    # --------------------------------------------------------
    # Net-delay error
    # --------------------------------------------------------

    net_delay_error = torch.abs(
        net_delays - g.ndata["n_net_delays_log"]
    )

    valid_net = torch.isfinite(net_delay_error)

    net_delay_mae = (
        net_delay_error[valid_net].mean().item()
    )

    # --------------------------------------------------------
    # Endpoint selection
    # --------------------------------------------------------

    endpoint_mask = (
        g.ndata["n_is_timing_endpt"]
        .bool()
        .unsqueeze(1)
        .expand_as(valid)
    )

    endpoint_valid = valid & endpoint_mask

    # --------------------------------------------------------
    # Slack prediction
    # --------------------------------------------------------

    rats = g.ndata["n_rats"][:, :4]

    predicted_slack = rats - pred
    actual_slack = rats - truth

    valid_slack = (
        torch.isfinite(predicted_slack) &
        torch.isfinite(actual_slack)
    )

    valid_endpoint = valid_slack & endpoint_mask

    predicted_slack_endpoint = predicted_slack[
        valid_endpoint
    ]

    actual_slack_endpoint = actual_slack[
        valid_endpoint
    ]

    # --------------------------------------------------------
    # Timing violation classification
    # --------------------------------------------------------

    predicted_violation = (
        predicted_slack_endpoint < 0.05
    )

    actual_violation = (
        actual_slack_endpoint < 0
    )

    tp = (
        predicted_violation &
        actual_violation
    ).sum().item()

    fp = (
        predicted_violation &
        ~actual_violation
    ).sum().item()

    fn = (
        ~predicted_violation &
        actual_violation
    ).sum().item()

    tn = (
        ~predicted_violation &
        ~actual_violation
    ).sum().item()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total = tp + fp + fn + tn

    flagged = tp + fp

    flag_rate = (
        flagged / total
        if total > 0
        else 0.0
    )

    violation_coverage = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"R2:                  {r2:.5f}")
    print(f"Timing MAE:          {mae:.5f}")
    print(f"Median error:        {median_error:.5f}")
    print(f"Maximum error:       {max_error:.5f}")
    print(f"Net-delay MAE:       {net_delay_mae:.5f}")

    print("\nViolation detection:")
    print(f"TP:                  {tp}")
    print(f"FP:                  {fp}")
    print(f"FN:                  {fn}")
    print(f"TN:                  {tn}")
    print(f"Precision:           {precision:.5f}")
    print(f"Recall:              {recall:.5f}")
    print(f"F1:                  {f1:.5f}")
    print(f"Verification workload: {flag_rate:.5%}")
    print(f"Violation coverage:     {violation_coverage:.5%}")

    results.append({
        "circuit": test_key,
        "r2": r2,
        "timing_mae": mae,
        "median_error": median_error,
        "max_error": max_error,
        "net_delay_mae": net_delay_mae,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "verification_workload": flag_rate,
        "violation_coverage": violation_coverage,
    })


# ============================================================
# Save CSV
# ============================================================

fieldnames = list(results[0].keys())

with open(
    CSV_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# Global summary
# ============================================================

total_tp = sum(r["tp"] for r in results)
total_fp = sum(r["fp"] for r in results)
total_fn = sum(r["fn"] for r in results)
total_tn = sum(r["tn"] for r in results)

global_precision = (
    total_tp / (total_tp + total_fp)
)

global_recall = (
    total_tp / (total_tp + total_fn)
)

global_f1 = (
    2 * global_precision * global_recall /
    (global_precision + global_recall)
)

global_flag_rate = (
    (total_tp + total_fp) /
    (
        total_tp +
        total_fp +
        total_fn +
        total_tn
    )
)


print("\n========================================")
print("GLOBAL SUMMARY")
print("========================================")

print(f"Total TP:              {total_tp}")
print(f"Total FP:              {total_fp}")
print(f"Total FN:              {total_fn}")
print(f"Total TN:              {total_tn}")
print(f"Precision:             {global_precision:.5%}")
print(f"Recall:                {global_recall:.5%}")
print(f"F1:                    {global_f1:.5f}")
print(f"Verification workload: {global_flag_rate:.5%}")

print("\nResults saved to:")
print(CSV_FILE)
