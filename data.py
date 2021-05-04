import numpy as np
import pandas as pd

from deft_hep.helper import convert_hwu_to_numpy

# K-factor from LO to NNLO at 13TeV
LO_XSEC = 4.574e2 # 4.584 +- 0.003 from https://arxiv.org/abs/1405.0301
NNLO_XSEC = 831.76 # from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
k_factor = NNLO_XSEC / LO_XSEC

def get_CMS_data(path: str) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(
        path,
        comment="#",
        header=0,
        names=[
            "mid",
            "min",
            "max",
            "value",
            "stat_up",
            "stat_down",
            "sys_up",
            "sys_down",
        ],
    )
    edges = np.append(df["min"].to_numpy(), df["max"][len(df["max"]) - 1])

    return edges, df["value"].to_numpy()


def get_CMS_cov(path: str, num_of_bins: int = 7):
    covar = [[0 for _ in range(num_of_bins)] for _ in range(num_of_bins)]

    with open(path) as f:
        for _ in range(10):
            next(f)

        while line := f.readline():
            x, y, val = iter(line.strip().split(","))
            covar[int(float(x)) - 1][int(float(y)) - 1] = float(val)
    return covar


def get_MC_signal(file_list, num_of_bins: int = 7):
    hist_list = [convert_hwu_to_numpy(f, num_of_bins) for f in file_list]
    for i, (edges, values) in enumerate(hist_list):
        bin_widths = np.fromiter((edges[i+1] - edges[i] for i in range(len(edges) - 1)), float, len(edges)-1)
        scaled_values = values / bin_widths * k_factor
        hist_list[i] = (edges, scaled_values)
    return hist_list


# def get_MC_covariance_matrix(weights):
# variances = np.sqrt(weights) ** 2
# covariance = numpy.zeros(len(weights), len(weights))
# for i, v in enumerate(variances):
# covariance[i][i] = v
# return covariance
