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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

scriptFolder = Path(__file__).parent
templateConfigPath = scriptFolder.joinpath("acc_EDP_vs_dim_arrayCol.yml")
destConfigPath = scriptFolder.parent.joinpath("./cam_config.yml")
simOutputPath = scriptFolder.joinpath("sim_run.log")
pyScriptPath = scriptFolder.parent.joinpath("./main.py")
resultDir = scriptFolder.joinpath("results")
if not resultDir.exists():
    resultDir.mkdir(parents=True)
plotlyOutputPath = scriptFolder.joinpath("./plot.html")
matplotlibOutputPath = scriptFolder.joinpath("./plot.png")

# dimList = [32, 64, 128, 256, 512]
dimList = [64, 128, 256, 512]
arrayColList = [32, 64, 128, 256]

# dimList = [32, 64]
# arrayColList = [32, 64]

# n_step = 10
n_step = 100

jobList = {
    "2bit": {
        "accuResultPath": resultDir.joinpath("2bitIdealAccu.csv"),
        "edpResultPath": resultDir.joinpath("2bitIdealedp.csv"),
        "hasVar": False,
        "bit": 2,
        "varStdDev": 0,
    },
    # "2bit_var": {
    #     "accuResultPath": resultDir.joinpath("2bitVarAccu.csv"),
    #     "edpResultPath": resultDir.joinpath("2bitVaredp.csv"),
    #     "hasVar": True,
    #     "bit": 2,
    #     "varStdDev": 0.75,
    # },
    # "3bit": {
    #     "accuResultPath": resultDir.joinpath("3bitIdealAccu.csv"),
    #     "edpResultPath": resultDir.joinpath("3bitIdealedp.csv"),
    #     "hasVar": False,
    #     "bit": 3,
    #     "varStdDev": 0,
    # },
    # "3bit_var": {
    #     "accuResultPath": resultDir.joinpath("3bitVarAccu.csv"),
    #     "edpResultPath": resultDir.joinpath("3bitVaredp.csv"),
    #     "hasVar": True,
    #     "bit": 3,
    #     "varStdDev": 1.5,
    # },
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


def matplotlib_plot(jobList: dict):
    dim2colorDict = {64: "b", 128: "g", 256: "m", 512: "c"}
    # dim2colorDict = {32: "r", 64: "b", 128: "g", 256: "m", 512: "c"}
    col2markerDict = {32: "p", 64: "*", 128: "o", 256: "+"}
    for jobName in jobList.keys():
        accuracyResult = jobList[jobName]["accuResult"]
        edpResult = jobList[jobName]["edpResult"]
        accuracies = []
        EDPs = []
        labels = []
        for dim in dimList:
            for arrayCol in arrayColList:
                if dim < arrayCol:
                    continue
                accuracies.append(accuracyResult.at[dim, arrayCol])
                EDPs.append(edpResult.at[dim, arrayCol] * 1e9)
                labels.append(f"dim={dim},col={arrayCol}")
                plt.scatter(
                    edpResult.at[dim, arrayCol] * 1e9,
                    accuracyResult.at[dim, arrayCol],
                    s=80,
                    c=dim2colorDict[dim],
                    marker=col2markerDict[arrayCol],
                )
                # plt.text(
                #     edpResult.at[dim, arrayCol],
                #     accuracyResult.at[dim, arrayCol],
                #     f"dim={dim},col={arrayCol}",
                #     ha="center",
                #     va="bottom",
                # )

    # Add labels and title
    plt.xlabel("EDP (aJ·s)")
    plt.ylabel("Accuracy")

    legendHandles = []
    legendLabels = []
    for dim in dim2colorDict:
        legendHandles.append(
            Line2D([0], [0], color=dim2colorDict[dim], lw=5)
        )
        legendLabels.append(f"dim={dim}")
    for col in col2markerDict:
        legendHandles.append(
            Line2D(
                [0],
                [0],
                color="k",
                marker=col2markerDict[col],
                lw=0,
            )
        )
        legendLabels.append(f"col={col}")
    plt.legend(handles=legendHandles, labels=legendLabels)

    plt.savefig(matplotlibOutputPath, dpi=300)
    print("Saved plot")


def plotly_plot(jobList: dict):
    traces = []
    for jobName in jobList.keys():
        accuracyResult = jobList[jobName]["accuResult"]
        edpResult = jobList[jobName]["edpResult"]
        accuracies = []
        EDPs = []
        labels = []
        for dim in dimList:
            for arrayCol in arrayColList:
                if dim < arrayCol:
                    continue
                accuracies.append(accuracyResult.at[dim, arrayCol])
                EDPs.append(edpResult.at[dim, arrayCol])
                labels.append(f"dim={dim},col={arrayCol}")

        traces.append(
            go.Scatter(
                x=EDPs,
                y=accuracies,
                mode="markers",
                name=jobName,
                text=labels,
                textposition="middle center",
            )
        )

    # Create the figure and add the traces
    fig = go.Figure(data=traces)

    # Set layout options
    fig.update_layout(
        title="ACC-EDP vs dim/arrayCol",
        xaxis=dict(title="EDP"),
        yaxis=dict(title="Accu"),
    )

    fig.write_html(plotlyOutputPath)
    print("save plot")


def run_exp(
    accuResult: pd.DataFrame,
    edpResult: pd.DataFrame,
    bit: int,
    hasVar: bool,
    varStdDev: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for dim in dimList:
        for arrayCol in arrayColList:
            if dim < arrayCol:
                continue
            print("*" * 30)
            print(f"dim = {dim}, arrayCol = {arrayCol}, hasVar = {hasVar}")
            # change
            with open(templateConfigPath, mode="r") as fin:
                config = yaml.load(fin, Loader=yaml.FullLoader)

            config["array"]["col"] = arrayCol
            assert dim % arrayCol == 0, "dim must be a multiplier of arrayCol!"
            config["arch"]["SubarraysPerArray"] = dim // arrayCol
            config["cell"]["writeNoise"]["hasWriteNoise"] = hasVar
            config["cell"]["writeNoise"]["variation"]["stdDev"] = varStdDev
            config["array"]["bit"] = bit
            config["query"]["bit"] = bit

            with open(destConfigPath, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

            assert pyScriptPath.exists(), "The script to be run does not exist!"
            assert (
                os.system(
                    f"python {pyScriptPath} --dim {dim} --n_step {n_step} | tee {simOutputPath}"
                )
                == 0
            ), "run script failed."

            accu, edp = getAccuEDP(simOutputPath)

            accuResult.at[dim, arrayCol] = accu
            edpResult.at[dim, arrayCol] = edp
    return accuResult, edpResult


def main():
    for jobName in jobList.keys():
        print("**************************************************")
        print(f"               job: {jobName}")
        print("**************************************************")
        jobList[jobName]["accuResult"] = pd.DataFrame(
            np.zeros((len(dimList), len(arrayColList)), dtype=float),
            columns=arrayColList,
            index=dimList,
        )
        jobList[jobName]["edpResult"] = pd.DataFrame(
            np.zeros((len(dimList), len(arrayColList)), dtype=float),
            columns=arrayColList,
            index=dimList,
        )

        jobList[jobName]["accuResult"], jobList[jobName]["edpResult"] = run_exp(
            jobList[jobName]["accuResult"],
            jobList[jobName]["edpResult"],
            jobList[jobName]["bit"],
            jobList[jobName]["hasVar"],
            jobList[jobName]["varStdDev"],
        )

        jobList[jobName]["accuResult"].to_csv(jobList[jobName]["accuResultPath"])
        jobList[jobName]["edpResult"].to_csv(jobList[jobName]["edpResultPath"])
        print("saved stat")

    # plotly_plot(jobList)
    matplotlib_plot(jobList)


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

    # plotly_plot(jobList)
    matplotlib_plot(jobList)



if __name__ == "__main__":
    # main()
    plot_jobs()
