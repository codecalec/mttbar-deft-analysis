import sys
from pathlib import Path
import glob

import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("science")

from data import get_CMS_data, get_CMS_cov, get_MC_signal


MC_PATH = "/home/agv/Documents/Honours/Project/data_generation/ttbar_ctg"
DATA_PATH = Path("/home/agv/Documents/Honours/Project/data/1811.06625/")


def generate_json(mc_path, data_path, ctg_list, filename):
    import json

    output_dict = {
        "config": {
            "run_name": "CMS-TOP-ctg",
            "data": {"observable": "$m_{t\\bar{t}}$"},
            "model": {
                "input": "numpy",
                "inclusive_k_factor": 1,
                "c_i_benchmark": 2,
                "max_coeff_power": 1,
                "cross_terms": False,
            },
            "fit": {"n_burnin": 500, "n_total": 3000, "n_walkers": 32},
        }
    }

    test_output_dict = {
        "config": {
            "run_name": "CMS-TOP-ctg",
            "data": {"observable": "$m_{t\\bar{t}}$"},
            "model": {
                "input": "numpy",
                "inclusive_k_factor": 1,
                "c_i_benchmark": 2,
                "max_coeff_power": 1,
                "cross_terms": False,
            },
            "fit": {"n_burnin": 500, "n_total": 3000, "n_walkers": 32},
        }
    }

    # Data Section
    edges, values = get_CMS_data(data_path / "mttbar.csv")
    output_dict["config"]["data"]["bins"] = edges.tolist()
    output_dict["config"]["data"]["central_values"] = values.tolist()
    test_output_dict["config"]["data"]["bins"] = edges.tolist()
    test_output_dict["config"]["data"]["central_values"] = values.tolist()

    covariance = get_CMS_cov(data_path / "mttbar_cov.csv")
    output_dict["config"]["data"]["covariance_matrix"] = covariance
    test_output_dict["config"]["data"]["covariance_matrix"] = covariance

    # Model Section
    hist_list = get_MC_signal(mc_path, 7)
    samples = [[1, ctg] for ctg in ctg_list]
    output_dict["config"]["model"]["samples"] = samples[:3]
    test_output_dict["config"]["model"]["samples"] = samples[3:]

    predictions = [None] * len(hist_list)
    for i, (edges_mc, values_mc) in enumerate(hist_list):
        assert (edges == edges_mc).all()
        predictions[i] = values_mc.tolist()
    output_dict["config"]["model"]["predictions"] = predictions[:3]
    test_output_dict["config"]["model"]["predictions"] = predictions[3:]

    output_dict["config"]["model"]["prior_limits"] = {"c_{tG}": [-2.0, 2.0]}

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    with open(f"test_{filename}", "w") as test_f:
        json.dump(test_output_dict, test_f, indent=4)


def run_analysis(filename):

    import deft_hep as deft

    config = deft.ConfigReader(filename)

    pb = deft.PredictionBuilder(1, config.samples, config.predictions)

    fitter = deft.MCMCFitter(config, pb)
    sampler = fitter.sampler

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )

    deft.SummaryPlotter(config, pb, sampler)


def run_validation(config_filename, test_filename):
    import deft_hep as deft

    config = deft.ConfigReader(config_filename)
    pb = deft.PredictionBuilder(1, config.samples, config.predictions)
    mv = deft.ModelValidator(pb)

    config_test = deft.ConfigReader(test_filename)
    samples, predictions = mv.validate(config_test)
    print(samples, predictions)

    mv.comparison_plot(config_test, predictions)


if __name__ == "__main__":

    ctg_list = [-2, 0, 2, -1, 1]
    file_list = glob.glob(MC_PATH + "/run_*_LO/*.HwU")

    config_filename = "ttbar_ctg.json"

    if len(sys.argv) <= 1:
        run_analysis(config_filename)
    elif len(sys.argv) <= 2:
        if sys.argv[1] == "config":
            generate_json(file_list, DATA_PATH, ctg_list, config_filename)
        elif sys.argv[1] == "validate":
            run_validation(config_filename, config_filename)
