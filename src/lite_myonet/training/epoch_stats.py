"""Collect detached epoch statistics without a device transfer per batch."""

import numpy as np
import torch


class EpochStatistics:
    def __init__(self, keep_probabilities=False):
        self.keep_probabilities = keep_probabilities
        self.loss_rows = []
        self.batch_sizes = []
        self.classifications = []
        self.probabilities = []

    def add(self, losses, targets, predictions, top3_hits, probabilities):
        self.loss_rows.append(torch.stack(tuple(value.detach() for value in losses)))
        self.batch_sizes.append(int(targets.shape[0]))
        self.classifications.append(torch.stack(
            (targets.detach(), predictions.detach(), top3_hits.to(dtype=torch.long)), dim=1
        ))
        if self.keep_probabilities:
            self.probabilities.append(probabilities.detach().float())

    def finish(self, num_classes):
        # Keep the original Python-double accumulation order for logged losses.
        rows = torch.stack(self.loss_rows).cpu().tolist()
        totals = [0.0] * len(rows[0])
        for row, batch_size in zip(rows, self.batch_sizes):
            for index, value in enumerate(row):
                totals[index] += value * batch_size
        classification = torch.cat(self.classifications).cpu().numpy()
        probabilities = (
            torch.cat(self.probabilities).cpu().numpy()
            if self.probabilities else np.empty((0, num_classes), dtype=np.float32)
        )
        return (totals, classification[:, 0].copy(), classification[:, 1].copy(),
                int(classification[:, 2].sum()), probabilities)
