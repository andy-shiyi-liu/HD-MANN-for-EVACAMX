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
templateConfigPath = scriptFolder.joinpath("acc_vs_sensingLimit_var.yml")
destConfigPath = scriptFolder.parent.joinpath("./cam_config.yml")
simOutputPath = scriptFolder.joinpath("sim_run.log")
pyScriptPath = scriptFolder.parent.joinpath("./main.py")
resultDir = scriptFolder.joinpath("results")
if not resultDir.exists():
    resultDir.mkdir(parents=True)
plotOutputPath = scriptFolder.joinpath("./plot.html")
emailScriptPath = scriptFolder.joinpath("./sendEmail.py")

varList = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
sensingLimitList = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# varList = [0, 0.4, 0.8]
# sensingLimitList = [0, 2, 4]

jobList = {
    "128dim128col": {
        "accuResultPath": resultDir.joinpath("128dim128colAccu_acc_vs_sensingLimit_var.csv"),
        "edpResultPath": resultDir.joinpath("128dim128coledp_acc_vs_sensingLimit_var.csv"),
        "dim": 128,
        "col": 128,
    },
    # "128dim32col": {
    #     "accuResultPath": resultDir.joinpath("128dim32colAccu_acc_vs_sensingLimit_var.csv"),
    #     "edpResultPath": resultDir.joinpath("128dim32coledp_acc_vs_sensingLimit_var.csv"),
    #     "dim": 128,
    #     "col": 32,
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


def plot(jobList: dict):
    traces = []
    for jobName in jobList.keys():
        accuracyResult = jobList[jobName]["accuResult"]
        edpResult = jobList[jobName]["edpResult"]
        varGrid, sensingLimitGrid = np.meshgrid(varList, sensingLimitList)

        traces.append(
            go.Surface(x=varGrid, y=sensingLimitGrid, z=accuracyResult, text=jobName)
        )
    
    # print(varGrid)
    # print(sensingLimitGrid)
    # print(accuracyResult)

    fig = go.Figure(data=traces)
    fig.update_layout(
    scene=dict(
        zaxis=dict(title='Accuracy'),
        xaxis=dict(title='Variation'),
        yaxis=dict(title='Sensing Limit')
    )
)

    fig.to_html("./plot.html")
    fig.show()


def run_exp(
    accuResult: pd.DataFrame, edpResult: pd.DataFrame, dim: int, arrayCol: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for var in varList:
        for sensingLimit in sensingLimitList:
            print("*" * 30)
            print(f"var = {var}, sensingLimit = {sensingLimit}")
            # change
            with open(templateConfigPath, mode="r") as fin:
                config = yaml.load(fin, Loader=yaml.FullLoader)

            config["array"]["col"] = arrayCol
            assert dim % arrayCol == 0, "dim must be a multiplier of arrayCol!"
            config["arch"]["SubarraysPerArray"] = dim // arrayCol
            config["cell"]["writeNoise"]["variation"]["stdDev"] = var
            config["array"]["sensingLimit"] = sensingLimit

            with open(destConfigPath, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

            assert pyScriptPath.exists(), "The script to be run does not exist!"
            assert (
                os.system(f"python {pyScriptPath} --dim {dim} | tee {simOutputPath}")
                == 0
            ), "run script failed."

            accu, edp = getAccuEDP(simOutputPath)

            accuResult.at[var, sensingLimit] = accu
            edpResult.at[var, sensingLimit] = edp
    return accuResult, edpResult


def main():
    for jobName in jobList.keys():
        print("**************************************************")
        print(f"               job: {jobName}")
        print("**************************************************")
        jobList[jobName]["accuResult"] = pd.DataFrame(
            np.zeros((len(varList), len(sensingLimitList)), dtype=float),
            columns=sensingLimitList,
            index=varList,
        )
        jobList[jobName]["edpResult"] = pd.DataFrame(
            np.zeros((len(varList), len(sensingLimitList)), dtype=float),
            columns=sensingLimitList,
            index=varList,
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
    os.system(f'python {emailScriptPath} -m "Finished script: acc_vs_sensingLimit_var"')



if __name__ == "__main__":
    main()

