"""
This file is for running simulations to study how accuracy and EDP 
changes with dim/arrayCol
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
from typing import Tuple
import re
import plotly.express as px
import plotly.graph_objects as go

scriptFolder = Path(__file__).parent
templateConfigPath = scriptFolder.joinpath("acc_vs_variation.yml")
destConfigPath = scriptFolder.parent.joinpath("./cam_config.yml")
simOutputPath = scriptFolder.joinpath("sim_run.log")
pyScriptPath = scriptFolder.parent.joinpath("./main.py")
resultDir = scriptFolder.joinpath("results")
if not resultDir.exists():
    resultDir.mkdir(parents=True)
plotOutputPath = scriptFolder.joinpath("./plot.html")

variationList = list(np.arange(0, 1.1, 0.1))


jobList = {
    "dim128col128": {
        "accuResultPath": resultDir.joinpath("dim128Col128Accu.csv"),
        "edpResultPath": resultDir.joinpath("dim128Col128edp.csv"),
        "dim": 128,
        "col": 128,
    },
    "dim128Col64": {
        "accuResultPath": resultDir.joinpath("dim128Col64Accu.csv"),
        "edpResultPath": resultDir.joinpath("dim128Col64edp.csv"),
        "dim": 128,
        "col": 64,
    },
    "dim128Col32": {
        "accuResultPath": resultDir.joinpath("dim128Col32Accu.csv"),
        "edpResultPath": resultDir.joinpath("dim128Col32edp.csv"),
        "dim": 128,
        "col": 32,
    },
    "dim64col64": {
        "accuResultPath": resultDir.joinpath("dim64Col64Accu.csv"),
        "edpResultPath": resultDir.joinpath("dim64Col64edp.csv"),
        "dim": 64,
        "col": 64,
    },
    "dim32Col32": {
        "accuResultPath": resultDir.joinpath("dim32Col32Accu.csv"),
        "edpResultPath": resultDir.joinpath("dim32Col32edp.csv"),
        "dim": 32,
        "col": 32,
    },
}


def getAccuEDP(logPath: Path) -> Tuple[float, float]:
    """
    return: accuracy, EDP
    """
    accuracy = None
    edp = None
    with open(logPath, mode="r") as f:
        for lineID, line in enumerate(f):
            if re.search("Query Latency", line):
                tokens = line.strip().split(" ")
                latency = float(tokens[-2])
                energy = float(tokens[-1])
                if edp == None:
                    edp = latency * energy
                else:
                    assert (
                        edp == latency * energy
                    ), "EDP for different run is different!"
            elif re.search("CAM acc = ", line):
                accu_this = float(re.search("[0-9]+.[0-9]+", line).group())
                if accuracy == None:
                    accuracy = accu_this
                else:
                    assert (
                        accu_this == accuracy
                    ), "accuracy for different run is different!"

    assert accuracy != None and edp != None, "accuracy or edp not extracted!"
    return accuracy, edp


def plot(jobList: dict):
    traces = []
    for jobName in jobList.keys():
        accuracyResult = jobList[jobName]["accuResult"]
        edpResult = jobList[jobName]["edpResult"]
        accuracies = []
        for var in variationList:
            accuracies.append(accuracyResult.at[0, var])

        traces.append(
            go.Scatter(
                x=variationList,
                y=accuracies,
                mode="lines",
                name=jobName,
            )
        )

    # Create the figure and add the traces
    fig = go.Figure(data=traces)

    # Set layout options
    fig.update_layout(
        title="Accuracy vs Standard Deviation of Variation",
        xaxis=dict(title="Standard Deviation of Variation"),
        yaxis=dict(title="Accuracy"),
    )

    fig.write_html(plotOutputPath)
    print("save plot")
    fig.show()

def run_exp(
    accuResult: pd.DataFrame,
    edpResult: pd.DataFrame,
    dim: int,
    col: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for var in variationList:
        print("*" * 30)
        print(f"var = {var}")
        # change
        with open(templateConfigPath, mode="r") as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        config["array"]["col"] = col
        assert dim % col == 0, "dim must be a multiplier of col!"
        config["arch"]["SubarraysPerArray"] = dim // col
        config["cell"]["writeNoise"]["hasWriteNoise"] = True
        config["cell"]["writeNoise"]["variation"]["stdDev"] = float(var)

        with open(destConfigPath, "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        assert pyScriptPath.exists(), "The script to be run does not exist!"
        assert (
            os.system(f"python {pyScriptPath} --dim {dim} | tee {simOutputPath}")
            == 0
        ), "run script failed."

        accu, edp = getAccuEDP(simOutputPath)

        accuResult.at[0, var] = accu
        edpResult.at[0,var] = edp
    return accuResult, edpResult


def main():
    for jobName in jobList.keys():
        print("**************************************************")
        print(f"               job: {jobName}")
        print("**************************************************")
        jobList[jobName]["accuResult"] = pd.DataFrame(
            np.zeros((1, len(variationList)), dtype=float),
            columns=variationList,
        )
        jobList[jobName]["edpResult"] = pd.DataFrame(
            np.zeros((1, len(variationList)), dtype=float),
            columns=variationList,
        )

        jobList[jobName]["accuResult"], jobList[jobName]["edpResult"] = run_exp(
            jobList[jobName]["accuResult"],
            jobList[jobName]["edpResult"],
            jobList[jobName]["dim"],
            jobList[jobName]["col"],
        )

        jobList[jobName]["accuResult"].to_csv(jobList[jobName]["accuResultPath"])
        jobList[jobName]["edpResult"].to_csv(jobList[jobName]["edpResultPath"])
        print("saved stat")

    plot(jobList)


def plot_jobs():
    for jobName in jobList:
        jobList[jobName]["accuResult"] = pd.read_csv(
            jobList[jobName]["accuResultPath"], index_col=0
        )
        jobList[jobName]["accuResult"].columns = [
            int(i) for i in jobList[jobName]["accuResult"].columns
        ]
        jobList[jobName]["edpResult"] = pd.read_csv(
            jobList[jobName]["edpResultPath"], index_col=0
        )
        jobList[jobName]["edpResult"].columns = [
            int(i) for i in jobList[jobName]["edpResult"].columns
        ]

    plot(jobList)


if __name__ == "__main__":
    main()
    # plot_jobs()
