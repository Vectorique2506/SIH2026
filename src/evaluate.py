"""
Evaluate trained model against SNR / STOI / PESQ targets.

Owner: shared — run this once a checkpoint exists.

Report metrics SPLIT BY BUCKET (stationary vs impulsive) — do not average
them together. See README targets: SNR > 15 dB, STOI > 0.85, PESQ > 2.5.
"""

from pystoi import stoi
from pesq import pesq


def evaluate_bucket(model, dataset, bucket: str):
    """Run model over all examples in `bucket`, return mean SNR/STOI/PESQ."""
    # TODO:
    # - filter dataset to this bucket
    # - run model on each noisy example
    # - compute SNR improvement, stoi(clean, estimate, fs), pesq(fs, clean, estimate, 'wb')
    # - average and return
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: load checkpoint, run evaluate_bucket for "stationary" and "impulsive"
    # separately, print a table matching the one in README.md
    pass
