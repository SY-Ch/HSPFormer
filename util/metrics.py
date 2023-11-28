import torch

import torchmetrics

import torch
import torchmetrics

class Metrics(torchmetrics.Metric):
    def __init__(self, num_classes: int, ignore_label: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        
        # Define the state
        self.add_state("hist", default=torch.zeros(num_classes, num_classes), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        
        # Update the state
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        ious[ious.isnan()] = 0.
        miou = ious.mean().item()

        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        f1[f1.isnan()] = 0.
        mf1 = f1.mean().item()

        acc = self.hist.diag() / self.hist.sum(1)
        acc[acc.isnan()] = 0.
        macc = acc.mean().item()

        return {
            "IOUs": ious.cpu().numpy().round(4).tolist(),
            "mIOU": round(miou, 4),
            "F1": f1.cpu().numpy().round(2).tolist(),
            "mF1": round(mf1, 4),
            "ACC": acc.cpu().numpy().round(4).tolist(),
            "mACC": round(macc, 4)
        }

class DepthMetrics(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, min_depth=1e-3):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.min_depth = min_depth
        
        self.add_state("error_squared_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("abs_rel_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("log_10_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("threshold_counts", default=torch.tensor([0, 0, 0]), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"
        
        # Create a mask for valid depth values
        valid_mask = (target > self.min_depth) & (preds > self.min_depth)

        diff = (preds - target)[valid_mask]
        self.error_squared_sum += torch.sum(diff * diff)
        self.abs_rel_error_sum += torch.sum(torch.abs(diff) / target[valid_mask])
        self.log_10_sum += torch.sum(torch.abs(torch.log10(preds[valid_mask]) - torch.log10(target[valid_mask])))
        self.count += torch.sum(valid_mask).item()
        
        ratio = torch.maximum(preds[valid_mask] / target[valid_mask], target[valid_mask] / preds[valid_mask])
        self.threshold_counts[0] += torch.sum(ratio < 1.25)
        self.threshold_counts[1] += torch.sum(ratio < 1.25**2)
        self.threshold_counts[2] += torch.sum(ratio < 1.25**3)

    def compute(self):
        rmse = torch.sqrt(self.error_squared_sum / self.count).item()
        abs_rel_error = (self.abs_rel_error_sum / self.count).item()
        log_10_error = (self.log_10_sum / self.count).item()
        threshold_accuracy_1 = (self.threshold_counts[0].float() / self.count).item()
        threshold_accuracy_2 = (self.threshold_counts[1].float() / self.count).item()
        threshold_accuracy_3 = (self.threshold_counts[2].float() / self.count).item()

        return {
            "RMSE": rmse,
            "Absolute Relative Error": abs_rel_error,
            "Log10 Error": log_10_error,
            "δ<1.25": threshold_accuracy_1,
            "δ<1.25^2": threshold_accuracy_2,
            "δ<1.25^3": threshold_accuracy_3
        }
