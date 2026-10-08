import torch
from data_graph import data_test
from model import TimingGCN

print("Loading pretrained TimingPredict model...")
model = TimingGCN()
model.cuda()
model.load_state_dict(torch.load("./checkpoints/08_atcd_specul/15799.pth"))
model.eval()

test_keys = [
    "jpeg_encoder",
    "usbf_device",
    "aes192",
    "xtea",
    "spm",
    "y_huff",
    "synth_ram"
]

thresholds = [0.0, 0.25, 0.5, 1.0, 2.0]

with open("../results/risk_threshold_sweep.txt", "w") as f:

    for key in test_keys:
        g, ts = data_test[key]

        with torch.no_grad():
            pred = model(g, ts, groundtruth=False)[2][:, :4]
            truth = g.ndata["n_ats"][:, :4]

        rats = g.ndata["n_rats"][:, :4]

        endpoint = g.ndata["n_is_timing_endpt"].bool().unsqueeze(1)
        valid = torch.isfinite(pred) & torch.isfinite(truth) & torch.isfinite(rats) & endpoint

        pred_slack = rats - pred
        actual_slack = rats - truth

        valid_pred = pred_slack[valid]
        valid_actual = actual_slack[valid]

        actual_violation = valid_actual < 0

        print(f"\n===== {key} =====")
        f.write(f"\n===== {key} =====\n")
        f.write(f"Total valid endpoint samples: {valid_pred.numel()}\n")

        for threshold in thresholds:
            flagged = valid_pred < threshold

            tp = (flagged & actual_violation).sum().item()
            fp = (flagged & ~actual_violation).sum().item()
            fn = (~flagged & actual_violation).sum().item()
            tn = (~flagged & ~actual_violation).sum().item()

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            flag_rate = (tp + fp) / valid_pred.numel()
            violation_coverage = recall

            line = (
                f"threshold={threshold:.2f} ns | "
                f"flagged={tp+fp} | "
                f"flag_rate={flag_rate*100:.2f}% | "
                f"TP={tp} | FP={fp} | FN={fn} | TN={tn} | "
                f"precision={precision*100:.2f}% | "
                f"recall={recall*100:.2f}% | "
                f"F1={f1:.4f}"
            )

            print(line)
            f.write(line + "\n")

print("\nSaved: ../results/risk_threshold_sweep.txt")
