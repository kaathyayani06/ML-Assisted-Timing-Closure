import torch

from data_graph import data_test
from model import TimingGCN

print("Loading pretrained TimingPredict model...")

model = TimingGCN()
model.cuda()

checkpoint = "./checkpoints/08_atcd_specul/15799.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()

print("Model loaded successfully.")

test_keys = [
    "jpeg_encoder",
    "usbf_device",
    "aes192",
    "xtea",
    "spm",
    "y_huff",
    "synth_ram"
]

print("\n===== Risk Analysis Across Test Circuits =====")

for test_key in test_keys:

    print("\n----------------------------------------")
    print("Circuit:", test_key)
    print("----------------------------------------")

    g, ts = data_test[test_key]

    with torch.no_grad():
        pred = model(g, ts, groundtruth=False)[2][:, :4]
        truth = g.ndata["n_ats"][:, :4]

    valid = torch.isfinite(pred) & torch.isfinite(truth)

    error = torch.abs(pred - truth)

    error_valid = error[valid]

    print("Valid timing samples:", error_valid.numel())
    print("Timing MAE:", error_valid.mean().item())
    print("Timing median error:", error_valid.median().item())
    print("Timing maximum error:", error_valid.max().item())

    with torch.no_grad():
        net_delays = model(g, ts, groundtruth=False)[0]

    net_delay_error = torch.abs(net_delays - g.ndata["n_net_delays_log"])
    valid_net = torch.isfinite(net_delay_error)

    print("Net delay MAE:", net_delay_error[valid_net].mean().item())
    print("Net delay median error:", net_delay_error[valid_net].median().item())
    print("Net delay maximum error:", net_delay_error[valid_net].max().item())

    endpoint = (
        g.ndata["n_is_timing_endpt"]
        .bool()
        .unsqueeze(1)
        .expand_as(valid)
    )

    endpoint_error = error[valid & endpoint]
    non_endpoint_error = error[valid & ~endpoint]

    print("\nEndpoint MAE:", endpoint_error.mean().item())
    print("Non-endpoint MAE:", non_endpoint_error.mean().item())

    rats = g.ndata["n_rats"][:, :4]

    predicted_slack = rats - pred

    valid_slack = (
        torch.isfinite(rats) &
        torch.isfinite(pred)
    )

    valid_endpoint_slack = (
        valid_slack &
        g.ndata["n_is_timing_endpt"].bool().unsqueeze(1)
    )

    slack = predicted_slack[valid_endpoint_slack]
    slack_error = error[valid_endpoint_slack]

    risk_low = slack < 0
    risk_medium = (slack >= 0) & (slack < 1)
    risk_high = slack >= 1

    print("\nPredicted Slack Risk Analysis:")

    if risk_low.any():
        print(
            "Predicted violation MAE:",
            slack_error[risk_low].mean().item()
        )
    else:
        print("Predicted violation MAE: N/A")

    if risk_medium.any():
        print(
            "Near-critical MAE:",
            slack_error[risk_medium].mean().item()
        )
    else:
        print("Near-critical MAE: N/A")

    if risk_high.any():
        print(
            "Comfortable MAE:",
            slack_error[risk_high].mean().item()
        )
    else:
        print("Comfortable MAE: N/A")

    print("Predicted violation samples:", risk_low.sum().item())
    print("Near-critical samples:", risk_medium.sum().item())
    print("Comfortable samples:", risk_high.sum().item())

    actual_slack = rats - truth
    actual_violation = (actual_slack < 0) & torch.isfinite(actual_slack)
    predicted_violation = (predicted_slack < 0) & torch.isfinite(predicted_slack)

    actual_violation = actual_violation[valid_endpoint_slack]
    predicted_violation = predicted_violation[valid_endpoint_slack]

    tp = (predicted_violation & actual_violation).sum().item()
    fp = (predicted_violation & ~actual_violation).sum().item()
    fn = (~predicted_violation & actual_violation).sum().item()
    tn = (~predicted_violation & ~actual_violation).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\nTiming Violation Detection:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    total = tp + fp + fn + tn
    flagged = tp + fp
    actual_violations = tp + fn

    flag_rate = flagged / total if total > 0 else 0.0
    violation_coverage = tp / actual_violations if actual_violations > 0 else 0.0

    print("Flag rate:", flag_rate)
    print("Violation coverage:", violation_coverage)
