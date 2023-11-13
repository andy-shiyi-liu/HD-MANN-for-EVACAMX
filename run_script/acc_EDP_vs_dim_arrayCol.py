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

scriptFolder = Path(__file__).parent
templateConfigPath = scriptFolder.joinpath("cam_config.yml")
destConfigPath = scriptFolder.parent.joinpath("./cam_config.yml")
simOutputPath = scriptFolder.joinpath("sim_run.log")
pyScriptPath = scriptFolder.parent.joinpath("./main.py")
accuResultPath = scriptFolder.joinpath("./accuracyStat.csv")
EDPResultPath = scriptFolder.joinpath("./EDPStat.csv")
plotOutputPath = scriptFolder.joinpath("./plot.html")

dimList = [32, 64, 128, 256, 512]
arrayColList = [32, 64, 128, 256]


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


def plot(accuracyResult: pd.DataFrame, edpResult: pd.DataFrame):
    accuracys = []
    EDPs = []
    labels = []
    for dim in dimList:
        for arrayCol in arrayColList:
            if dim < arrayCol:
                continue
            accuracys.append(accuracyResult.at[dim, arrayCol])
            EDPs.append(edpResult.at[dim, arrayCol])
            labels.append(f"dim={dim},col={arrayCol}")

    fig = px.scatter(
        x=EDPs,
        y=accuracys,
        text=labels,  # Assign labels to text attribute
        title="Acc-EDP vs dim/arrayCol",
        labels={"x": "EDP", "y": "accu", "text": "Data Point"},
    )

    fig.write_html(plotOutputPath)
    print('save plot')


def main():
    accuracyResult = pd.DataFrame(
        np.zeros((len(dimList), len(arrayColList)), dtype=float),
        columns=arrayColList,
        index=dimList,
    )
    EDPResult = pd.DataFrame(
        np.zeros((len(dimList), len(arrayColList)), dtype=float),
        columns=arrayColList,
        index=dimList,
    )

    for dim in dimList:
        for arrayCol in arrayColList:
            if dim < arrayCol:
                continue
            print("*" * 30)
            print(f"dim = {dim}, arrayCol = {arrayCol}")
            # change
            with open(templateConfigPath, mode="r") as fin:
                config = yaml.load(fin, Loader=yaml.FullLoader)

            config["array"]["col"] = arrayCol
            config['arch']['SubarraysPerArray'] = dim // arrayCol

            with open(destConfigPath, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

            assert pyScriptPath.exists(), "The script to be run does not exist!"
            assert (
                os.system(
                    f'python {pyScriptPath} --dim {dim} | tee {simOutputPath}'
                )
                == 0
            ), "run script failed."

            accu, edp = getAccuEDP(simOutputPath)

            accuracyResult.at[dim, arrayCol] = accu
            EDPResult.at[dim, arrayCol] = edp

    accuracyResult.to_csv(accuResultPath)
    EDPResult.to_csv(EDPResultPath)
    print('saved stat')

    plot(accuracyResult, EDPResult)
    


if __name__ == "__main__":
    main()
