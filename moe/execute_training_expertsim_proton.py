import time
import os
import wandb

import pandas as pd
import numpy as np
from datetime import datetime

from itertools import combinations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from models_pytorch_sg import apply_expert_mask_vectorized

from utils_unified import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss,
                   regressor_loss, calculate_expert_utilization_entropy,
                   StratifiedBatchSampler, plot_cond_pca_tsne, plot_expert_heatmap)
from data_transformations import transform_data_for_training, ZDCType, SCRATCH_PATH
from training_setup_unified import setup_experts_unified, setup_router, load_checkpoint_weights
from training_utils import save_models_and_architectures


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)           # Check which CUDA version PyTorch was built with
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA Device 0: {torch.cuda.get_device_name(0)}")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

SAVE_EXPERIMENT_DATA = True
PLOT_IMAGES = False
epoch_to_load = None

EPOCHS = 200
N_EXPERTS = 3
BATCH_SIZE = 256

# SETTINGS & PARAMETERS
WS_MEAN_SAVE_THRESHOLD = 3.25  # Save the model parameters If achieves this threshold in evaluation

# Standard parameters for generator's regularization, constant for every expert
DI_STRENGTH = 0.1
IN_STRENGTH = 1e-3
IN_STRENGTH_LOWER_VAL = 0.001
AUX_STRENGTH = 1e-3
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
LR_R = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_DIM = 10
N_COND = 9


GEN_STRENGTH = 1 #e-1  # Strength on the generator loss in the router loss calculation
DIFF_STRENGTH = 1e-5  # Differentation on the generator loss in the router loss calculation
UTIL_STRENGTH = 1e-2  # Strength on the expert utilization entropy in the router loss calculation
STOP_ROUTER_TRAINING_EPOCH = EPOCHS

DATA_IMAGES_PATH = "data_proton_photonsum_proton_1_2312.pkl"
DATA_COND_PATH = "data_cond_photonsum_proton_1_2312.pkl"
DATA_POSITIONS_PATH = "data_coord_proton_photonsum_proton_1_2312.pkl"

INPUT_IMAGE_SHAPE = (56, 30)

NAME = f"<wandb_run_name>"

data = pd.read_pickle(DATA_IMAGES_PATH)
data_cond = pd.read_pickle(DATA_COND_PATH)
data_posi = pd.read_pickle(DATA_POSITIONS_PATH)

photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
wandb_run_name = f"{NAME}_{DATE_STR}"
EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}"
EXPERIMENT_DIR_NAME = os.path.join(SCRATCH_PATH, EXPERIMENT_DIR_NAME)

### TRANSFORM DATA FOR TRAINING ###
x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
expert_number_test, scaler_poz, data_cond_names, filepath_models = transform_data_for_training(
    data_cond, data,
    data_posi,
    EXPERIMENT_DIR_NAME,
    ZDCType.PROTON,
    SAVE_EXPERIMENT_DATA,
    load_data_file_from_checkpoint=False)

# CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
org = np.exp(x_test) - 1
ch_org = np.array(org).reshape(-1, *INPUT_IMAGE_SHAPE)
del org
ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

# Loss and optimizer
binary_cross_entropy_criterion = nn.BCELoss()
aux_reg_criterion = regressor_loss

generator, generator_optimizer, discriminator, discriminator_optimizer, aux_reg, aux_reg_optimizer, \
num_params_gen, num_params_disc, num_params_aux = \
    setup_experts_unified(N_EXPERTS, N_COND, NOISE_DIM, LR_G,
                          LR_D, LR_A, DI_STRENGTH,
                          IN_STRENGTH, device)
router_network, router_optimizer, num_params_router = setup_router(N_COND, N_EXPERTS, LR_R, device)


def generator_train_step(generator, discriminator, a_reg, cond, g_optimizer, a_optimizer, criterion,
                         true_positions, std, intensity, n_experts, gumbel_softmax, BATCH_SIZE):
    # Train Generator
    g_optimizer.zero_grad()

    noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    noise_2 = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)

    # generate fake images
    fake_images = generator(noise, cond)
    fake_images_2 = generator(noise_2, cond)

    fake_images = apply_expert_mask_vectorized(fake_images, gumbel_softmax)
    fake_images_2 = apply_expert_mask_vectorized(fake_images_2, gumbel_softmax)

    # validate two images
    fake_output, fake_latent = discriminator(fake_images, cond)
    fake_output_2, fake_latent_2 = discriminator(fake_images_2, cond)
    # print("fake_output", fake_output.shape)

    gen_loss = criterion(fake_output, torch.ones_like(fake_output))


    div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
                                      noise, noise_2,
                                      std, generator.di_strength)
    div_loss = div_loss.sum()
    intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
                                                                                               intensity,
                                                                                               generator.in_strength)\

    intensity_loss = intensity_loss.sum()
    gen_loss = gen_loss + div_loss + intensity_loss

    # Train auxiliary regressor
    a_optimizer.zero_grad()
    generated_positions = a_reg(fake_images)

    aux_reg_loss = aux_reg_criterion(generated_positions, true_positions, router_logits, n_experts)
    aux_reg_loss = aux_reg_loss.sum()

    gen_loss += aux_reg_loss*AUX_STRENGTH

    gen_loss.backward(retain_graph=True)
    g_optimizer.step()
    a_optimizer.step()
    aux_reg_loss = torch.tensor(0.0, device=device)
    return gen_loss.data, div_loss.data, intensity_loss.data, aux_reg_loss.data,\
           std_intensity, mean_intensity, mean_intenisties


def discriminator_train_step(disc, generator, d_optimizer, criterion, real_images, cond, router_logits, n_experts, BATCH_SIZE) -> np.float32:
    """Returns Python float of disc_loss value"""
    # Train discriminator
    d_optimizer.zero_grad()

    # calculate loss for real images
    real_output, _ = disc(real_images, cond)
    real_labels = torch.ones_like(real_output)
    loss_real_disc = criterion(real_output, real_labels)

    # calculate loss for generated images
    noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    fake_images = generator(noise, cond)  # [B, n_experts, H, W] or [B, n_experts, 1, H, W]

    masked_fake = apply_expert_mask_vectorized(fake_images, router_logits)
    fake_output, _ = disc(masked_fake.detach(), cond)
    loss_fake_disc = criterion(fake_output, torch.zeros_like(fake_output))

    # Accumulate and compute discriminator loss
    disc_loss = loss_real_disc + loss_fake_disc
    disc_loss.backward()
    d_optimizer.step()
    return disc_loss.data


def train_step(batch, epoch):
    # print("train step start")
    real_images, real_images_2, cond, std, intensity, true_positions = batch

    # Vectorized device transfer
    real_images = real_images.unsqueeze(1).to(device)  # [B, 1, H, W]
    cond = cond.to(device, non_blocking=True)  # [B, 9]
    std = std.unsqueeze(1).to(device, non_blocking=True)
    intensity = intensity.unsqueeze(1).to(device, non_blocking=True)
    true_positions = true_positions.unsqueeze(1).to(device, non_blocking=True)
    BATCH_SIZE = real_images.shape[0]


    # Train router network
    router_optimizer.zero_grad()
    gumbel_softmax, router_logits = router_network(cond)  # Get predicted experts assignments for samples. Outputs are the probabilities of each expert for each sample. Shape: (batch, N_EXPERTS)

    predicted_expert = gumbel_softmax.argmax(dim=1)  # (BATCH_SIZE, 1)


    # calculate the class counts for each expert
    class_counts = torch.zeros(N_EXPERTS, dtype=torch.float).to(device)
    for class_label in range(N_EXPERTS):
        class_counts[class_label] = (predicted_expert == class_label).sum().item()
    class_counts_adjusted = class_counts / predicted_expert.size(0)


    # Train each discriminator independently
    disc_loss = discriminator_train_step(discriminator, generator,
                                         discriminator_optimizer,
                                         binary_cross_entropy_criterion, real_images,
                                         cond, gumbel_softmax, N_EXPERTS, BATCH_SIZE)

    selected_cond = cond
    selected_true_positions = true_positions
    selected_intensity = intensity
    selected_std = std
    selected_generator = generator
    selected_generator_optimizer = generator_optimizer
    selected_discriminator = discriminator
    selected_aux_reg = aux_reg
    selected_aux_reg_optimizer = aux_reg_optimizer

    gen_loss, div_loss, intensity_loss, \
    aux_reg_loss, std_intensity, \
    mean_intensity, mean_intensities = generator_train_step(selected_generator,
                                                            selected_discriminator,
                                                            selected_aux_reg,
                                                            selected_cond,
                                                            selected_generator_optimizer,
                                                            selected_aux_reg_optimizer,
                                                            binary_cross_entropy_criterion,
                                                            selected_true_positions,
                                                            selected_std,
                                                            selected_intensity,
                                                            N_EXPERTS,
                                                            gumbel_softmax,
                                                            BATCH_SIZE)


    # Save statistics
    mean_intensities_experts = mean_intensity
    std_intensities_experts = std_intensity

    gan_loss_scaled = (gen_loss+disc_loss) * GEN_STRENGTH  #  Added that on 06.01.25
    expert_entropy_loss = calculate_expert_utilization_entropy(gumbel_softmax,
                                                               UTIL_STRENGTH) if UTIL_STRENGTH != 0 else torch.tensor(
        0.0,
        requires_grad=False,
        device=gumbel_softmax.device)

    router_loss = gan_loss_scaled - expert_entropy_loss - differentiation_loss

    # Train Router Network
    router_loss.backward()
    router_optimizer.step()

    return gen_loss.item(), \
           disc_loss.item(), router_loss.item(), div_loss.cpu().item(), \
           intensity_loss.cpu().item(), aux_reg_loss.cpu().item(), class_counts.cpu().detach(), \
           std_intensities_experts, mean_intensities_experts, expert_distribution_loss.item(), \
           differentiation_loss.item(), expert_entropy_loss.item(), gan_loss_scaled.item()


def train(train_loader, epochs, y_test):
    if epoch_to_load is None:
        start_epoch = 0
    else:
        start_epoch = epoch_to_load + 1
    for epoch in range(start_epoch, epochs):
        start = time.time()
        gen_losses_epoch = []
        disc_losses_epoch = []
        router_loss_epoch = []
        div_loss_epoch = []
        intensity_loss_epoch = []
        aux_reg_loss_epoch = []
        expert_distribution_loss_epoch = []
        differentiation_loss_epoch = []
        expert_entropy_loss_epoch = []
        gan_loss_epoch = []

        n_chosen_experts = [[] for _ in range(N_EXPERTS)]
        mean_intensities_experts = [[] for _ in range(N_EXPERTS)]
        std_intensities_experts = [0] * N_EXPERTS

        # Iterate through both data loaders
        for batch in train_loader:
            start_batch = time.time()
            gen_losses, disc_losses, router_loss, div_loss, intensity_loss, \
            aux_reg_loss, n_chosen_experts_batch, std_intensities_experts_batch, \
            mean_intensities_experts_batch, expert_distribution_loss, differentiation_loss, \
            expert_entropy_loss, gan_loss = train_step(batch, epoch)

            gen_losses_epoch.append(gen_losses)
            disc_losses_epoch.append(disc_losses)
            router_loss_epoch.append(router_loss)
            div_loss_epoch.append(div_loss)
            intensity_loss_epoch.append(intensity_loss)
            aux_reg_loss_epoch.append(aux_reg_loss)
            expert_distribution_loss_epoch.append(expert_distribution_loss)
            differentiation_loss_epoch.append(differentiation_loss)
            expert_entropy_loss_epoch.append(expert_entropy_loss)
            gan_loss_epoch.append(gan_loss)

            for i in range(N_EXPERTS):
                n_chosen_experts[i].append(n_chosen_experts_batch[i])
                mean_intensities_experts[i].append(mean_intensities_experts_batch[i].cpu().numpy())
                std_intensities_experts[i] = np.mean(std_intensities_experts_batch[i].cpu().numpy())

        epoch_time = time.time() - start
        print("Epoch: ", epoch, "Time: ", epoch_time, "s")

        # =====================================================
        #                   TEST GENERATION
        # =====================================================
        y_test_temp = torch.tensor(y_test, device=device)


        # Calculate WS distance across all distribution
        ws_mean, ws_std = calculate_joint_ws_across_experts(
            1,
            y_test_temp, router_network, generator,
            ch_org,
            NOISE_DIM, device)


        # Log to WandB tool
        log_data = {
            'ws_mean': ws_mean,
            'ws_std': ws_std,
            'div_loss': np.mean(div_loss_epoch),
            'intensity_loss': np.mean(intensity_loss_epoch),
            'router_loss': np.mean(router_loss_epoch),
            'expert_distribution_loss': np.mean(expert_distribution_loss_epoch),
            'differentiation_loss': np.mean(differentiation_loss_epoch),
            'expert_entropy_loss': np.mean(expert_entropy_loss_epoch),
            'gan_loss': np.mean(gan_loss_epoch),
            'aux_reg_loss': np.mean(aux_reg_loss_epoch),
            'epoch_time': epoch_time,
            'epoch': epoch
        }

        log_data[f"gen_loss"] = np.mean(gen_losses_epoch)
        log_data[f"disc_loss"] = np.mean(disc_losses_epoch)

        if SAVE_EXPERIMENT_DATA:
            wandb.log(log_data)

        print(f'Time for epoch {epoch} is {epoch_time:.2f} sec')


config_wandb = {
    "Model": NAME,
    "n_experts": N_EXPERTS,
    "epochs": EPOCHS,
    "Date": DATE_STR,
    "Proton_min": photon_sum_proton_min,
    "Proton_max": photon_sum_proton_max,
    "generator_architecture": generator.name,
    "discriminator_architecture": discriminator.name,
    'stop_router_training_epoch': STOP_ROUTER_TRAINING_EPOCH,
    "diversity_strength": DI_STRENGTH,
    "intensity_strength": IN_STRENGTH,
    "intensity_strength_after_router_stops": IN_STRENGTH_LOWER_VAL,
    "auxiliary_strength": AUX_STRENGTH,
    "Generator_strength": GEN_STRENGTH,
    "Utilization_strength": UTIL_STRENGTH,
    "differentiation_strength": DIFF_STRENGTH,
    "Learning rate_generator": LR_G,
    "Learning rate_discriminator": LR_D,
    "Learning rate_router": LR_R,
    "Learning rate_aux_reg": LR_A,
    "Experiment_dir_name": EXPERIMENT_DIR_NAME,
    "Batch_size": BATCH_SIZE,
    "Noise_dim": NOISE_DIM,
    "router_arch": router_network.name,
    "intensity_loss_type": "mae"
}


# Separate datasets for each expert
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                              torch.tensor(y_train), torch.tensor(std_train),
                              torch.tensor(intensity_train), torch.tensor(positions_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                             torch.tensor(y_test), torch.tensor(std_test),
                             torch.tensor(intensity_test), torch.tensor(positions_test))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if SAVE_EXPERIMENT_DATA:
    wandb.login("<your-wandb-api-key>")
    wandb.finish()
    run = wandb.init(
        project="<your-project-name>",
        entity="<your-profile-name>",
        name=wandb_run_name,
        config=config_wandb,
    )
    run.log_code()

train(train_loader, EPOCHS, y_test)
