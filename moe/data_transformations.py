import os.path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import (create_dir, save_scales, save_train_test_indices, load_train_test_indices,
                   TRAIN_TEST_INDICES_FILENAME)
from enum import Enum

SCRATCH_PATH = "path_to_save_models/"
DIR_INFO = "{EXPERIMENT_DIR_NAME}/info/"  # dir for storing scales and indices of samples
DIR_MODELS = "{EXPERIMENT_DIR_NAME}/models/"


class ZDCType(Enum):
    PROTON = "proton"
    NEUTRON = "neutron"


def transform_data_for_training(data_cond, data, data_posi, EXPERIMENT_DIR_NAME, zdc_type: ZDCType = ZDCType.PROTON,
                                SAVE_EXPERIMENT_DATA = True, load_data_file_from_checkpoint: bool = False):
    dir_info = DIR_INFO.format(EXPERIMENT_DIR_NAME=EXPERIMENT_DIR_NAME)
    dir_models = DIR_MODELS.format(EXPERIMENT_DIR_NAME=EXPERIMENT_DIR_NAME)

    # GROUP CONDITIONAL DATA
    data_cond["cond"] = data_cond["Energy"].astype(str) + "|" + data_cond["Vx"].astype(str) + "|" + data_cond[
        "Vy"].astype(str) + "|" + data_cond["Vz"].astype(str) + "|" + data_cond["Px"].astype(str) + "|" + data_cond[
                            "Py"].astype(str) + "|" + data_cond["Pz"].astype(str) + "|" + data_cond["mass"].astype(
        str) + "|" + data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    data = np.log(data + 1).astype(np.float32)
    indices = np.arange(len(data))
    data_2 = data[ids]
    data_cond = data_cond.drop(columns="cond")

    # Diversity regularization
    if zdc_type == ZDCType.PROTON:
        expert_number = data_cond.expert_number  # experts assignments for every sample (e.g. 0, 1, 2 for 3 experts)

        scaler = MinMaxScaler()
        std = data_cond["std_proton"].values.reshape(-1, 1)
        std = np.float32(std)
        std = scaler.fit_transform(std)
        print("std max", std.max(), "min", std.min())

        # Intensity regularization
        intensity = data_cond["proton_photon_sum"].values.reshape(-1, 1)
        intensity = np.float32(intensity)
        print("intensity max", intensity.max(), "min", intensity.min())

        data_cond = data_cond.drop(columns=["std_proton", "proton_photon_sum",
                                            'group_number_proton', 'expert_number'])
    elif zdc_type == ZDCType.NEUTRON:
        scaler = MinMaxScaler()
        std = data_cond["std"].values.reshape(-1, 1)
        std = np.float32(std)
        std = scaler.fit_transform(std)
        print("std max", std.max(), "min", std.min())

        # Intensity regularization
        intensity = data_cond["neutron_photon_sum"].values.reshape(-1, 1)
        intensity = np.float32(intensity)
        print("intensity max", intensity.max(), "min", intensity.min())

        data_cond = data_cond.drop(columns=["std", "neutron_photon_sum",
                                            'group_number'])
    else:
        raise ValueError("Unsupported ZDC type!")

    # Auxiliary regressor
    scaler_poz = StandardScaler()
    data_xy = np.float32(data_posi.copy()[["max_x", "max_y"]])
    # data_xy = scaler_poz.fit_transform(data_xy)
    print('Load positions:', data_xy.shape, "cond max", data_xy.max(), "min", data_xy.min())

    data_cond_names = data_cond.columns
    scaler_cond = StandardScaler()
    data_cond = scaler_cond.fit_transform(data_cond.astype(np.float32))

    ### Return data for training based on the saved indices ###
    if load_data_file_from_checkpoint:
        train_indices, test_indices = load_train_test_indices(dir_info)
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, \
        expert_number_train, expert_number_test = data[train_indices], data[test_indices], \
                                                  data_2[train_indices], data_2[test_indices], \
                                                  data_cond[train_indices], data_cond[test_indices], \
                                                  std[train_indices], std[test_indices],\
                                                  intensity[train_indices], intensity[test_indices], \
                                                  data_xy[train_indices], data_xy[test_indices], \
                                                  expert_number[train_indices], expert_number[test_indices]

        return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, \
        expert_number_train, expert_number_test, scaler_poz, data_cond_names, dir_models

    if zdc_type == ZDCType.PROTON:
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, \
        expert_number_train, expert_number_test, train_indices, test_indices = train_test_split(
            data, data_2, data_cond, std, intensity, data_xy, expert_number.values, indices,
            test_size=0.2, shuffle=True)

        print("Data shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    elif zdc_type == ZDCType.NEUTRON:
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test,\
        train_indices, test_indices = train_test_split(
            data, data_2, data_cond, std, intensity, data_xy, indices,
            test_size=0.2, shuffle=True)

        print("Data shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    else:
        raise ValueError("Unsupported ZDC type!")

    # Save scales
    if SAVE_EXPERIMENT_DATA:
        create_dir(dir_info, SAVE_EXPERIMENT_DATA)
        save_scales(f"{zdc_type.value}", scaler_cond.mean_, scaler_cond.scale_, dir_info)
        create_dir(dir_models, SAVE_EXPERIMENT_DATA)

        # Save only the indices
        save_train_test_indices(dir_info, train_indices=train_indices, test_indices=test_indices)
    else:
        dir_models = None

    if zdc_type == ZDCType.NEUTRON:
        return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
               intensity_train, intensity_test, positions_train, positions_test, scaler_poz, data_cond_names, dir_models
    elif zdc_type == ZDCType.PROTON:
        return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
               intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
               expert_number_test, scaler_poz, data_cond_names, dir_models
