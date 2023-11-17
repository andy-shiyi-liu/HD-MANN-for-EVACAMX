import torch
import torch.nn as nn
import glob
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter
from more_itertools import unzip
import shutil
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import csv
import yaml
import json
import argparse


from cnn import CNNController
from data_generator import DataGenerator
from utils import *

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)

from EvaCAMX.EvaCAMX import EvaCAMX


train_model = False
dim = 256
n_step = 10
skip_software_inference = False

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--dim", type=int, help="Embedding Dimension")
parser.add_argument("--n_step", type=int, help="Running Times for Inference")
parser.add_argument(
    "--skip_software_inference", action="store_true", help="Skip software inference"
)

args = parser.parse_args()
if args.dim:
    dim = args.dim
if args.n_step:
    n_step = args.n_step
if args.skip_software_inference:
    skip_software_inference = True


def load_config():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    # with open(script_dir + "/cam_config.json", "r") as f:
    # config = json.load(f)
    with open(script_dir + "/cam_config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def simCAM(CAM_Data, CAM_Query):
    cam_config = load_config()
    cam = EvaCAMX(cam_config)
    cam.write(CAM_Data)
    CAM_pred_ids, _, _ = cam.query(CAM_Query)  # Accuracy Evaluation
    print("CAM Simulation Done")
    return CAM_pred_ids


def train(
    model,
    data_generator,
    optimizer,
    criterion,
    device,
    euc_MAX,
    euc_MIN,
    softab_num,
    D=512,
    n_step=50000,
    save=True,
    sharpen="softmax",
):
    exp_name = f"{W}way{S}shot{D}dim"
    writer = SummaryWriter(log_dir=scriptFolder.joinpath("./log"), comment=exp_name)

    loss_train = []
    val_accs = []
    steps = []
    best_acc = 0
    # for step in tqdm(range(n_step), disable=True):
    for step in tqdm(range(n_step)):
        # prep data
        (
            support_label,
            support_set,
            query_label,
            query_set,
        ) = data_generator.sample_batch("train", 32)
        support_label, support_set = prep_data(support_label, support_set, device)
        query_label, query_set = prep_data(query_label, query_set, device)
        # support set loading
        support_keys = None
        model.eval()
        with torch.no_grad():
            support_keys = model(support_set)  # output: d-dim real vector

        # query evaluation
        model.train()
        query_keys = model(query_set)
        euc_dis = -get_Euclidean_dist(query_keys, support_keys)
        # euc_dis = (euc_dis - torch.min(euc_dis)) / (torch.max(euc_dis) - torch.min(euc_dis))
        v_min = torch.min(euc_dis)
        v_max = torch.max(euc_dis)
        # euc_dis = euc_dis + (-v_min) #+ 0.0001
        euc_dis = (euc_dis - v_min) / (v_max - v_min) * (euc_MAX - (euc_MIN)) + (
            euc_MIN
        )
        # print("euc_dis:",euc_dis)
        # sharpened = sharpening_softabs(cosine_sim, 10)
        if sharpen == "softmax":
            # print("Using softmax sharpening function")
            sharpened = sharpening_softmax(euc_dis)
            # print(euc_dis.size())
            # sharpened = torch.sub(1,sharpened)
            # sharpened = euc_dis
            # print("sharpened:",sharpened)
            # print(np.shape(sharpened))
        elif sharpen == "softabs":
            # print('Using softabs sharpening function')
            sharpened = sharpening_softabs(euc_dis, softab_num)
            # sharpened = sharpening_softabs(euc_dis, 10)
            # sharpened = euc_dis
            # print(np.shape(sharpened))

        normalized = normalize(sharpened)
        # normalized = sharpened
        # print("normalized:",normalized)
        # print("support_label", support_label)
        pred = weighted_sum(normalized, support_label)
        # pred = torch.sub(1,pred)
        # print("pred:",pred)
        # print("query_label:",query_label)
        optimizer.zero_grad()
        loss = criterion(pred, query_label)
        # print(loss)
        loss.backward()
        if step % 500 == 0:
            print(f"train loss = {loss}")
            writer.add_scalar("Loss/Train", loss, step)
            acc = inference_software(
                model, data_gen, device, key_mem_transform=None, n_step=250, type="val"
            )
            # acc = inference(
            #     model,
            #     data_gen,
            #     device,
            #     key_mem_transform=None,
            #     sum_argmax=False,
            #     type="test",
            # )
            print(f"val acc = {acc}")
            writer.add_scalar("Acc/Val", acc, step)
            loss_train.append(loss.detach().cpu().numpy())
            val_accs.append(acc)
            steps.append(step)

            # save model
            if save:
                if not scriptFolder.joinpath("./results/5way5shot").exists():
                    scriptFolder.joinpath("./results/5way5shot").mkdir(parents=True)
                torch.save(
                    model.state_dict(),
                    scriptFolder.joinpath(
                        f"./results/5way5shot/{job_id_number}_{exp_name}_checkpoint.pth.tar"
                    ),
                )
                if acc > best_acc:
                    best_acc = acc
                    shutil.copy(
                        scriptFolder.joinpath(
                            f"./results/5way5shot/{job_id_number}_{exp_name}_checkpoint.pth.tar"
                        ),
                        scriptFolder.joinpath(
                            f"./results/5way5shot/{job_id_number}_{exp_name}_best.pth.tar"
                        ),
                    )

        # backprop
        optimizer.step()
    return model, steps, loss_train, val_accs


def inference_software(
    model: CNNController,
    data_generator: DataGenerator,
    device,
    key_mem_transform=binarize,
    sum_argmax=True,
    type="val",
):
    model.eval()
    accumulated_acc = 0
    if key_mem_transform in (bipolarize, binarize):
        raise DeprecationWarning("This part of code is deprecated and not tested!")
        for i in tqdm(range(n_step), disable=True):
            (
                support_label,
                support_set,
                query_label,
                query_set,
            ) = data_generator.sample_batch(type, 32)
            support_label, support_set = prep_data(support_label, support_set, device)
            query_label, query_set = prep_data(query_label, query_set, device)
            support_label = support_label.cpu().numpy()
            query_label = query_label.cpu().numpy()
            with torch.no_grad():
                support_keys = key_mem_transform(
                    model(support_set).cpu().detach().numpy()
                )
                query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
                dot_sim = get_dot_prod_similarity(query_keys, support_keys)
                sharpened = np.abs(dot_sim)
                if sum_argmax:
                    pred = np.dot(sharpened, support_label)
                    pred_argmax = np.argmax(pred, axis=1)
                else:
                    support_label_argmax = np.argmax(support_label, axis=1)
                    pred_argmax = support_label_argmax[sharpened.argmax(axis=1)]
                query_label_argmax = np.argmax(query_label, axis=1)
                # print(np.sum(pred_argmax == query_label_argmax))
                accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(
                    pred_argmax
                )
        return accumulated_acc / n_step
    else:
        # for i in tqdm(range(n_step), disable=True):
        for i in tqdm(range(n_step)):
            (
                support_label,
                support_set,
                query_label,
                query_set,
            ) = data_generator.sample_batch(type, 32)
            support_label, support_set = prep_data(support_label, support_set, device)
            query_label, query_set = prep_data(query_label, query_set, device)
            with torch.no_grad():
                support_keys = model(support_set)
                query_keys = model(query_set)
                euc_dis = -get_Euclidean_dist(query_keys, support_keys)
                sharpened = sharpening_softmax(euc_dis)
                normalized = normalize(sharpened)
                if sum_argmax:
                    pred = weighted_sum(normalized, support_label).cpu().numpy()
                    pred_argmax = np.argmax(pred, axis=1)
                else:
                    support_label_argmax = np.argmax(
                        support_label.cpu().numpy(), axis=1
                    )
                    pred_argmax = support_label_argmax[
                        normalized.cpu().numpy().argmax(axis=1)
                    ]
                query_label_argmax = np.argmax(query_label.cpu().numpy(), axis=1)
                accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(
                    pred_argmax
                )
        return accumulated_acc / n_step


def inference_CAM(
    model: CNNController,
    data_generator: DataGenerator,
    device,
    key_mem_transform=binarize,
    sum_argmax=True,
    type="val",
):
    model.eval()
    accumulated_acc = 0
    if key_mem_transform in (bipolarize, binarize):
        raise DeprecationWarning("This part of code is deprecated and not tested!")
        for i in tqdm(range(n_step), disable=True):
            (
                support_label,
                support_set,
                query_label,
                query_set,
            ) = data_generator.sample_batch(type, 32)
            support_label, support_set = prep_data(support_label, support_set, device)
            query_label, query_set = prep_data(query_label, query_set, device)
            support_label = support_label.cpu().numpy()
            query_label = query_label.cpu().numpy()
            with torch.no_grad():
                support_keys = key_mem_transform(
                    model(support_set).cpu().detach().numpy()
                )
                query_keys = key_mem_transform(model(query_set).cpu().detach().numpy())
                dot_sim = get_dot_prod_similarity(query_keys, support_keys)
                sharpened = np.abs(dot_sim)
                if sum_argmax:
                    pred = np.dot(sharpened, support_label)
                    pred_argmax = np.argmax(pred, axis=1)
                else:
                    support_label_argmax = np.argmax(support_label, axis=1)
                    pred_argmax = support_label_argmax[sharpened.argmax(axis=1)]
                query_label_argmax = np.argmax(query_label, axis=1)
                # print(np.sum(pred_argmax == query_label_argmax))
                accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(
                    pred_argmax
                )
        return accumulated_acc / n_step
    else:
        for i in tqdm(range(n_step), disable=True):
            (
                support_label,
                support_set,
                query_label,
                query_set,
            ) = data_generator.sample_batch(type, 32)

            # exit(0)
            support_label, support_set = prep_data(support_label, support_set, device)
            query_label, query_set = prep_data(query_label, query_set, device)

            with torch.no_grad():
                support_keys = model(support_set)
                query_keys = model(query_set)

                pred_row = simCAM(support_keys.cpu().numpy(), query_keys.cpu().numpy())

                popIndices = list(np.where(pred_row >= W * S)[0])
                pred_row = np.array(pred_row)
                pred_row = np.delete(pred_row, popIndices)
                pred_row = pred_row.flatten()

                support_label_argmax = np.argmax(support_label.cpu(), axis=1)
                pred_argmax = support_label_argmax.numpy().flatten()[pred_row]
                query_label_argmax = np.argmax(query_label.cpu().numpy(), axis=1)

                query_label_argmax = np.array(query_label_argmax)
                query_label_argmax = np.delete(query_label_argmax, popIndices)
                query_label_argmax = query_label_argmax.flatten()

                if len(pred_argmax) != 0:
                    accumulated_acc += np.sum(pred_argmax == query_label_argmax) / len(
                        pred_argmax
                    )
        return accumulated_acc / n_step


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    scriptFolder = Path(__file__).resolve()
    scriptFolder = Path(scriptFolder.parent)

    # Get the value of SGE_TASK_ID environment variable
    # job_id_number = int(os.environ.get('SGE_TASK_ID'))-1
    job_id_number = 0

    # num_dim = np.array([32,512])
    # num_ways = np.array([5, 20, 100])
    # num_shots = np.array([1, 5, 5])
    # euc_MAX = np.array([1, 0.8, 0.5, 0.3, 0.1])
    # euc_MIN = np.array([-1, -0.8, -0.5, -0.3, -0.1, 0])
    # softab_num = np.array([15, 13, 11, 10, 9, 7, 5])

    num_dim = np.array([5])
    num_ways = np.array([5])
    num_shots = np.array([5])
    euc_MAX = np.array([1, 0.8, 0.5, 0.3, 0.1])
    euc_MIN = np.array([-1, -0.8, -0.5, -0.3, -0.1, 0])
    softab_num = np.array([15, 13, 11, 10, 9, 7, 5])

    # max_number_of_possible_params = (
    #     num_dim.size * num_ways.size * euc_MAX.size * euc_MIN.size * softab_num.size
    # )
    # print(max_number_of_possible_params)

    # for job_id_number in range(210):
    divider = 1
    num_dim_num = int((job_id_number / divider) % num_dim.size)
    divider = num_dim.size
    num_ways_num = int((job_id_number / divider) % num_ways.size)
    divider = divider * num_ways.size
    euc_MAX_num = int((job_id_number / divider) % euc_MAX.size)
    divider = divider * euc_MAX.size
    euc_MIN_num = int((job_id_number / divider) % euc_MIN.size)
    divider = divider * euc_MIN.size
    softab_num_num = int((job_id_number / divider) % softab_num.size)
    divider = divider * softab_num.size

    W = num_ways[num_ways_num]  # way
    S = num_shots[num_ways_num]  # shots
    D = dim
    exp_name = f"{W}way{S}shot{D}dim"
    print(exp_name)

    data_gen = DataGenerator(W, S)
    model = CNNController(D).float().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    if scriptFolder.joinpath(f"./results/{D}dim/{exp_name}_best.pth.tar").exists():
        model.load_state_dict(
            torch.load(
                scriptFolder.joinpath(f"./results/{D}dim/{exp_name}_best.pth.tar")
            )
        )

    if train_model:
        model, steps, loss, acc = train(
            model,
            data_gen,
            optimizer,
            criterion,
            device,
            euc_MAX[euc_MAX_num],
            euc_MIN[euc_MIN_num],
            softab_num[softab_num_num],
            D,
            30000,
            sharpen="softabs",
        )

    # model, steps, loss, acc = train(
    #     model,
    #     data_gen,
    #     optimizer,
    #     criterion,
    #     device,
    #     euc_MAX[euc_MAX_num],
    #     euc_MIN[euc_MIN_num],
    #     softab_num[softab_num_num],
    #     D,
    #     500,
    #     sharpen="softabs",
    # )

    # steps = np.asarray(steps)
    # loss = np.asarray(loss)
    # acc = np.asarray(acc)

    # evaluation
    # assert scriptFolder.joinpath(
    #     f"./results/5way5shot/{job_id_number}_{exp_name}_best.pth.tar"
    # ).exists(), "No existing model parameters. Consider training the model first by setting global variable `train_model` to true."
    # model.load_state_dict(
    #     torch.load(
    #         scriptFolder.joinpath(
    #             f"./results/5way5shot/{job_id_number}_{exp_name}_best.pth.tar"
    #         )
    #     )
    # )

    if not skip_software_inference:
        software_acc = inference_software(
            model,
            data_gen,
            device,
            key_mem_transform=None,
            sum_argmax=False,
            type="test",
        )
        print(f"Software acc = {software_acc}")

    CAM_acc = inference_CAM(
        model,
        data_gen,
        device,
        key_mem_transform=None,
        sum_argmax=False,
        type="test",
    )
    print(f"CAM acc = {CAM_acc}")

    # np.savez(f'{exp_name}.npz', steps, loss, acc)

    # hyper_parameters_list = []
    # hyper_parameters_list.append(job_id_number)
    # hyper_parameters_list.append(num_dim[num_dim_num])
    # hyper_parameters_list.append(num_ways[num_ways_num])
    # hyper_parameters_list.append(num_shots[num_ways_num])
    # hyper_parameters_list.append(euc_MAX[euc_MAX_num])
    # hyper_parameters_list.append(euc_MIN[euc_MIN_num])
    # hyper_parameters_list.append(softab_num[softab_num_num])
    # hyper_parameters_list.append(acc)
    # print(hyper_parameters_list)

    # if not scriptFolder.joinpath("./parameter_tuning_results/5way5shot/").exists():
    #     scriptFolder.joinpath("./parameter_tuning_results/5way5shot/").mkdir(
    #         parents=True
    #     )
    # with open(
    #     scriptFolder.joinpath(
    #         "./parameter_tuning_results/5way5shot/"
    #         + str(job_id_number)
    #         + "_my_list.csv"
    #     ),
    #     mode="w",
    #     newline="",
    # ) as file:
    #     # Create a writer object
    #     writer = csv.writer(file)
    #     # Write the list to the CSV file
    #     writer.writerow(hyper_parameters_list)
