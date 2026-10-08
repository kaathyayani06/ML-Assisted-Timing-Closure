import torch
import gc

from data_graph import data_train, data_test
from model import TimingGCN


print("Loading pretrained TimingPredict model...")

model = TimingGCN()
model.cuda()

checkpoint = "./checkpoints/08_atcd_specul/15799.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()

print("Model loaded successfully.")


# ============================================================
# STEP 1: Calibration on training benchmarks
# ============================================================

print("\n===== Calibration =====")

calibration_keys = [
    "blabla",
    "usb_cdc_core",
    "BM64",
    "salsa20"
]

calibration_errors = []

for key in calibration_keys:

    print("Calibrating on:", key)

    g, ts = data_train[key]

    with torch.no_grad():
        pred = model(g, ts, groundtruth=False)[2][:, :4]
        truth = g.ndata["n_ats"][:, :4]

    valid = torch.isfinite(pred) & torch.isfinite(truth)

    error = torch.abs(pred - truth)

    valid_error = error[valid]

    calibration_errors.append(valid_error.cpu())

    print("Valid samples:", valid.sum().item())
    print("Mean error:", valid_error.mean().item())


all_calibration_errors = torch.cat(calibration_errors)

threshold_95 = torch.quantile(
    all_calibration_errors,
    0.95
).item()

print("\nTotal calibration samples:",
      all_calibration_errors.numel())

print("95th percentile error threshold:",
      threshold_95)


# ============================================================
# STEP 2: Evaluate interval coverage on all test benchmarks
# ============================================================

print("\n===== Test Interval Coverage =====")

test_keys = [
    "jpeg_encoder",
    "usbf_device",
    "aes192",
    "xtea",
    "spm",
    "y_huff",
    "synth_ram"
]

for test_key in test_keys:

    print("\nTesting on:", test_key)

    g, ts = data_test[test_key]

    with torch.no_grad():
        test_pred = model(
            g,
            ts,
            groundtruth=False
        )[2][:, :4]

        test_truth = g.ndata["n_ats"][:, :4]

    lower_bound = test_pred - threshold_95
    upper_bound = test_pred + threshold_95

    valid_interval = (
        torch.isfinite(test_truth) &
        torch.isfinite(lower_bound) &
        torch.isfinite(upper_bound)
    )

    covered = (
        (test_truth >= lower_bound) &
        (test_truth <= upper_bound) &
        valid_interval
    )

    coverage = (
        covered.sum().item() /
        valid_interval.sum().item()
    ) * 100

    test_error = torch.abs(
        test_pred - test_truth
    )

    valid_error = test_error[valid_interval]

    print("Valid samples:",
          valid_interval.sum().item())

    print("Covered samples:",
          covered.sum().item())

    print("Interval coverage:",
          coverage)

    print("Mean absolute error:",
          valid_error.mean().item())

    print("Median absolute error:",
          valid_error.median().item())

    print("Maximum absolute error:",
          valid_error.max().item())

    del (
        g,
        ts,
        test_pred,
        test_truth,
        lower_bound,
        upper_bound,
        valid_interval,
        covered,
        test_error,
        valid_error
    )

    gc.collect()
    torch.cuda.empty_cache()


print("\nCalibration experiment completed successfully.")
