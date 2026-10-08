import torch
import numpy as np
import gc

from data_graph import data_train,data_test
from model import TimingGCN

print("Loading pretrained TimingPredict model...")

model = TimingGCN()
model.cuda()

checkpoint = "./checkpoints/08_atcd_specul/15799.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()
# Save the original model parameters for uncertainty analysis
original_state = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
}

print("Original model parameters saved.")
print("Model loaded successfully.")
print("Loading jpeg_encoder test graph...")

g, ts = data_test["jpeg_encoder"]

with torch.no_grad():
    pred = model(g, ts, groundtruth=False)[2][:, :4]
    truth = g.ndata["n_ats"][:, :4]

print("Prediction shape:", pred.shape)
print("Baseline prediction mean:", pred.mean().item())
print("Baseline prediction std:", pred.std().item())
with torch.no_grad():
    net_delays, cell_delays, predicted_timing = model(g, ts, groundtruth=False)

print("Net delays shape:", net_delays.shape)
print("Cell delays shape:", cell_delays.shape)
print("Predicted timing shape:", predicted_timing.shape)
print("Predicted timing first 4 mean:",
      predicted_timing[:, :4].mean().item())
print("Predicted timing first 4 std:",
      predicted_timing[:, :4].std().item())

print("Predicted timing last 4 mean:",
      predicted_timing[:, 4:].mean().item())
print("Predicted timing last 4 std:",
      predicted_timing[:, 4:].std().item())
print("Cell delays finite:", torch.isfinite(cell_delays).all().item())
print("Cell delay min:", cell_delays.min().item())
print("Cell delay max:", cell_delays.max().item())
net_delay_error = torch.abs(net_delays - g.ndata["n_net_delays_log"])

valid_net = torch.isfinite(net_delay_error)

print("Net delay mean absolute error:",
      net_delay_error[valid_net].mean().item())

print("Net delay median absolute error:",
      net_delay_error[valid_net].median().item())

print("Net delay maximum absolute error:",
      net_delay_error[valid_net].max().item())
# Predicted vs actual endpoint slack
rats = g.ndata["n_rats"][:, :4]
actual_ats = g.ndata["n_ats"][:, :4]
predicted_ats = pred[:, :4]
endpoint = g.ndata["n_is_timing_endpt"].bool()

valid_slack = (
    torch.isfinite(rats) &
    torch.isfinite(actual_ats) &
    torch.isfinite(predicted_ats)
)

actual_slack = rats - actual_ats
predicted_slack = rats - predicted_ats

valid_endpoint = valid_slack & endpoint.unsqueeze(1)

actual_slack_valid = actual_slack[valid_endpoint]
predicted_slack_valid = predicted_slack[valid_endpoint]

print("Valid endpoint slack samples:", actual_slack_valid.numel())
print("Actual slack mean:", actual_slack_valid.mean().item())
print("Predicted slack mean:", predicted_slack_valid.mean().item())
print("Actual slack min:", actual_slack_valid.min().item())
print("Predicted slack min:", predicted_slack_valid.min().item())
print("Ground truth shape:", truth.shape)
valid = torch.isfinite(pred) & torch.isfinite(truth)

error = torch.abs(pred - truth)

valid_error = error[valid]

print("Valid timing samples:", valid.sum().item())
print("Mean absolute error:", valid_error.mean().item())
print("Maximum absolute error:", valid_error.max().item())

print("Median absolute error:", valid_error.median().item())
print("\n===== Endpoint Error Analysis =====")

endpoint = g.ndata["n_is_timing_endpt"].bool().unsqueeze(1).expand_as(valid)

endpoint_error = error[valid & endpoint]
non_endpoint_error = error[valid & ~endpoint]

print("Endpoint MAE:", endpoint_error.mean().item())
print("Non-endpoint MAE:", non_endpoint_error.mean().item())

print("Endpoint valid samples:", endpoint_error.numel())
print("Non-endpoint valid samples:", non_endpoint_error.numel())
print("\n===== Predicted Slack Risk Analysis =====")

risk_low = predicted_slack_valid < 0
risk_medium = (predicted_slack_valid >= 0) & (predicted_slack_valid < 1)
risk_high = predicted_slack_valid >= 1

print("Predicted violation MAE:", error[valid & (predicted_slack < 0)].mean().item())
print("Near-critical MAE:", error[valid & (predicted_slack >= 0) & (predicted_slack < 1)].mean().item())
print("Comfortable MAE:", error[valid & (predicted_slack >= 1)].mean().item())
print("Predicted violation samples:", risk_low.sum().item())
print("Near-critical samples:", risk_medium.sum().item())
print("Comfortable samples:", risk_high.sum().item())
