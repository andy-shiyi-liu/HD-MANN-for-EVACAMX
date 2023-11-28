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
import shutil
import matplotlib
# matplotlib.font_manager._rebuild()
# shutil.rmtree(matplotlib.get_cachedir())
from matplotlib import rcParams
rcParams['font.family']='sans-serif'
rcParams['font.sans-serif']=['Arial']
rcParams['font.size'] = 22
import matplotlib.pyplot as plt


scriptFolder = Path(__file__).parent
templateConfigPath = scriptFolder.joinpath("latency breakdown.yml")
destConfigPath = scriptFolder.parent.joinpath("./cam_config.yml")
simOutputPath = scriptFolder.joinpath("sim_run.log")
pyScriptPath = scriptFolder.parent.joinpath("./main.py")
resultDir = scriptFolder.joinpath("results")
if not resultDir.exists():
    resultDir.mkdir(parents=True)
plotlyOutputPath = scriptFolder.joinpath("./plot.html")
matplotlibOutputPath = scriptFolder.joinpath("./latency breakdown.png")
emailScriptPath = scriptFolder.joinpath("./sendEmail.py")

n_step = 1
sendEmail = False

latencyItems = {  # 're' for regular expression
    "Array": {
        "word re": "array latency",
        "value re": "[0-9]+[.]*[0-9]*[e]*[-+]*[0-9]*",
    },
    "Peripheral": {
        "word re": "peripheral latency",
        "value re": "[0-9]+[.]*[0-9]*[e]*[-+]*[0-9]*",
    },
    "Interconnect": {
        "word re": "interconnect latency",
        "value re": "[0-9]+[.]*[0-9]*[e]*[-+]*[0-9]*",
    },
}

jobList = {
    "128dim\n128col": {
        "resultPath": resultDir.joinpath("dim128Col128Latency_acc_vs_variation.csv"),
        "dim": 128,
        "col": 128,
    },
    "128dim\n64col": {
        "resultPath": resultDir.joinpath("dim128Col64Latency_acc_vs_variation.csv"),
        "dim": 128,
        "col": 64,
    },
    "128dim\n32col": {
        "resultPath": resultDir.joinpath("dim128Col32Latency_acc_vs_variation.csv"),
        "dim": 128,
        "col": 32,
    },
    "64dim\n64col": {
        "resultPath": resultDir.joinpath("dim64Col64Latency_acc_vs_variation.csv"),
        "dim": 64,
        "col": 64,
    },
    # "Dim=32\nCol=32": {
    #     "resultPath": resultDir.joinpath("dim32Col32Latency_acc_vs_variation.csv"),
    #     "dim": 32,
    #     "col": 32,
    # },
}


def getLatency(logPath: Path, result: pd.DataFrame) -> pd.DataFrame:
    """
    return: accuracy, EDP
    """
    with open(logPath, mode="r") as f:
        for lineID, line in enumerate(f):
            for item in latencyItems.keys():
                if re.search(latencyItems[item]["word re"], line):
                    value = float(
                        re.search(latencyItems[item]["value re"], line).group()
                    )
                    result.at[0, item] = value

    for key in result.columns:
        assert result.at[0, key] != 0, f"{key} not extracted or equals to 0!"
    return result


def plotly_plot(jobList: dict):
    raise DeprecationWarning("This function is untested!")
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

    fig.write_html(plotlyOutputPath)
    print("save plot")
    fig.show()


def run_exp(
    result: pd.DataFrame,
    dim: int,
    col: int,
) -> pd.DataFrame:
    print("*" * 30)
    with open(templateConfigPath, mode="r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    config["array"]["col"] = col
    assert dim % col == 0, "dim must be a multiplier of col!"
    config["arch"]["SubarraysPerArray"] = dim // col

    with open(destConfigPath, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    assert pyScriptPath.exists(), "The script to be run does not exist!"
    assert (
        os.system(
            f"python {pyScriptPath} --dim {dim} --n_step {n_step} --skip_software_inference | tee {simOutputPath}"
        )
        == 0
    ), "run script failed."

    result = getLatency(simOutputPath, result)

    return result


def main():
    for jobName in jobList.keys():
        print("**************************************************")
        print(f"               job: {jobName}")
        print("**************************************************")
        jobList[jobName]["results"] = pd.DataFrame(
            np.zeros((1, len(latencyItems.keys())), dtype=float),
            columns=latencyItems.keys(),
        )

        jobList[jobName]["results"] = run_exp(
            jobList[jobName]["results"],
            jobList[jobName]["dim"],
            jobList[jobName]["col"],
        )

        jobList[jobName]["results"].to_csv(jobList[jobName]["resultPath"])
        print("saved stat")

    matplotlib_plot(jobList)
    if sendEmail:
        os.system(f'python {emailScriptPath} -m "Finished script: acc_vs_variation"')


def matplotlib_plot(jobList: dict):
    for latencyItem in latencyItems.keys():
        latencyItems[latencyItem]["results"] = pd.DataFrame(
            np.zeros((1, len(jobList.keys()))), columns=list(jobList.keys())
        )
        for jobName in jobList.keys():
            # print(energyItems[energyItem]["results"])
            # print(jobList[jobName]["results"])
            latencyItems[latencyItem]["results"].at[0, jobName] = jobList[jobName][
                "results"
            ].at[0, latencyItem]

    barWidth = 0.7
    barOffset = 0
    index = np.arange(0, 3 * len(jobList.keys()), 3)
    for item in latencyItems:
        plt.bar(
            index + barOffset,
            latencyItems[item]["results"].to_numpy().flatten(),
            width=barWidth,
            label=item,
        )
        barOffset += barWidth

    # Adding labels and title
    plt.ylabel("Latency (ps)")
    plt.xticks(
        index + (barOffset - barWidth) / 2, jobList.keys()
    )  # Set x-axis ticks in the middle of the grouped bars
    plt.legend(fontsize=18)  # Display legend
    plt.tight_layout(pad=0.5)  # You can adjust the 'pad' parameter

    # Show the plot
    plt.savefig(matplotlibOutputPath, dpi=300)
    print("saved fig")


def plot_jobs():
    for jobName in jobList:
        jobList[jobName]["results"] = pd.read_csv(
            jobList[jobName]["resultPath"], index_col=0
        )

    # plotly_plot(jobList)
    matplotlib_plot(jobList)


if __name__ == "__main__":
    main()
    # plot_jobs()
