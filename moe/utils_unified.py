import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import center_of_mass
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats


TRAIN_TEST_INDICES_FILENAME = "train_test_indices.npz"


def get_channel_masks(input_array: np.ndarray):
    """
    Returns masks of 5 for input array.

    input_array: Array of shape(N, M)
    """

    # Create a copy of the input array to use as the mask
    mask = np.ones_like(input_array)
    n, m = input_array.shape

    # Define the pattern of checks
    pattern = np.array([[0, 1], [1, 0]])

    # Fill the input array with the pattern
    for i in range(n):
        for j in range(m):
            mask[i, j] = pattern[i % 2, j % 2]

    mask5 = np.ones_like(input_array) - mask

    # Divide the mask into four equal rectangles
    rows, cols = mask.shape
    mid_row, mid_col = rows // 2, cols // 2

    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()
    mask4 = mask.copy()

    mask4[mid_row:, :] = 0
    mask4[:, :mid_col] = 0

    mask2[:, :mid_col] = 0
    mask2[:mid_row, :] = 0

    mask3[mid_row:, :] = 0
    mask3[:, mid_col:] = 0

    mask1[:, mid_col:] = 0
    mask1[:mid_row, :] = 0

    return mask1, mask2, mask3, mask4, mask5


def sum_channels_parallel(data: np.ndarray):
    """
    Calculates the sum of 5 channels of input images. Each Input image is divided into 5 sections.

    data: Array of shape(x, N, M)
        Array of x images of the same size.
    """
    mask1, mask2, mask3, mask4, mask5 = get_channel_masks(data[0])

    ch1 = (data * mask1).sum(axis=1).sum(axis=1)
    ch2 = (data * mask2).sum(axis=1).sum(axis=1)
    ch3 = (data * mask3).sum(axis=1).sum(axis=1)
    ch4 = (data * mask4).sum(axis=1).sum(axis=1)
    ch5 = (data * mask5).sum(axis=1).sum(axis=1)

    return zip(ch1, ch2, ch3, ch4, ch5)


def get_max_value_image_coordinates(img):
    return np.unravel_index(np.argmax(img), img.shape)


def calculate_image_features(images):
    n_samples = images.shape[0]
    max_values_x = []
    max_values_y = []
    centers_x = []
    centers_y = []
    non_zero_pixels = []

    height, width = images.shape[1], images.shape[2]  # Image dimensions

    for img in images:
        # Max value across x and y directions
        max_values_x.append(np.max(np.sum(img, axis=0)))  # Sum along rows (columns remain)
        max_values_y.append(np.max(np.sum(img, axis=1)))  # Sum along columns (rows remain)

        # Handle all-zero images
        if np.sum(img > 0) == 0:  # Check if the image is all zeros
            centers_x.append(width / 2)  # Default to center of the image
            centers_y.append(height / 2)
        else:
            center = center_of_mass(img > 0)  # Use binary mask for center of mass
            centers_x.append(center[1])  # x-center
            centers_y.append(center[0])  # y-center

        # Number of pixels > 0
        non_zero_pixels.append(np.sum(img > 0))

    # Convert to numpy arrays for easier manipulation
    return np.array([max_values_x, max_values_y, centers_x, centers_y, non_zero_pixels])


def create_dir(path, SAVE_EXPERIMENT_DATA):
    if SAVE_EXPERIMENT_DATA:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)


def save_scales(model_name, scaler_means, scaler_scales, filepath):
    out_fnm = f"{model_name}_scales.txt"
    res = "#means"
    for mean_ in scaler_means:
        res += "\n" + str(mean_)
    res += "\n\n#scales"
    for scale_ in scaler_scales:
        res += "\n" + str(scale_)

    with open(filepath+out_fnm, mode="w") as f:
        f.write(res)


def save_train_test_indices(filepath_dir: str, train_indices: np.array, test_indices: np.array):
    filepath = os.path.join(filepath_dir, TRAIN_TEST_INDICES_FILENAME)
    np.savez(filepath, train_indices=train_indices, test_indices=test_indices)
    print("Data train-test-split indices saved to", filepath)


def load_train_test_indices(checkpoint_data_filepath_dir: str):
    try:
        checkpoint_data_load_file = os.path.join(checkpoint_data_filepath_dir, TRAIN_TEST_INDICES_FILENAME)
        data_indices = np.load(checkpoint_data_load_file)
        print("Data train-test-split indices loaded!")
        return data_indices["train_indices"], data_indices["test_indices"]
    except FileNotFoundError:
        print("No data train-test-split indices found!")
        raise FileNotFoundError
#
# def calculate_joint_ws_across_experts(n_calc, x_tests: List, y_tests: List, generator,
#                                       ch_org, ch_org_expert, noise_dim, device, batch_size=64, n_experts=3,
#                                       shape_images=(56, 30)):
#     """
#     Calculates the Wasserstein (WS) distance across the whole distribution
#     for a unified generator with multi-expert outputs.
#     """
#     # Validate input data lengths
#     if len(x_tests) != len(y_tests) or len(x_tests) != n_experts:
#         raise ValueError("Length of data is not the same")
#
#     # Initialize WS distance arrays
#     ws = np.zeros((n_calc, 5))  # Overall WS distances
#     ws_exp = np.zeros((n_calc, n_experts, 5))  # WS distances for each expert
#
#     for j in range(n_calc):  # Perform multiple calculations of the WS distance
#         ch_gen_all = []  # For gathering the whole generated distribution of pixels
#         ch_gen_expert = [[] for _ in range(n_experts)]  # For gathering expert-specific distributions
#
#         # Process each expert data separately
#         for expert_idx in range(n_experts):
#             y_test_temp = torch.tensor(y_tests[expert_idx], device=device)
#             num_samples = x_tests[expert_idx].shape[0]
#
#             if num_samples == 0:
#                 continue
#
#             try:
#                 # Get predictions from the unified generator for this expert's data
#                 results_all = get_predictions_from_unified_generator(
#                     batch_size, num_samples, noise_dim,
#                     device, y_test_temp, generator,
#                     expert_idx, n_experts,
#                     shape_images=shape_images
#                 )
#                 # print(
#                 #     f"For expert {expert_idx}. Samples generated: {results_all.shape}, real_samples: {num_samples}")
#             except Exception as e:
#                 print("Error for expert:", expert_idx)
#                 print("y_tests length:", len(y_tests[expert_idx]))
#                 print("num_samples:", num_samples)
#                 print("Exception:", e)
#                 continue
#
#             # Sum channels and store results
#             ch_gen_smaller = pd.DataFrame(sum_channels_parallel(results_all)).values
#             ch_gen_expert[expert_idx] = ch_gen_smaller.copy()  # Expert-specific data
#             ch_gen_all.extend(ch_gen_smaller.copy())  # Overall data
#
#         ch_gen_all = np.array(ch_gen_all)  # Convert to numpy array
#         # print("Shape of all generated:", ch_gen_all.shape)
#
#         # Calculate WS distances
#         for i in range(5):
#             # Overall WS distance
#             ws[j][i] = wasserstein_distance(ch_org[:, i], ch_gen_all[:, i])
#
#             # Per-expert WS distances
#             # for exp_idx in range(n_experts):
#             #     if len(ch_gen_expert[exp_idx]) == 0 or ch_org_expert[exp_idx].shape[0] == 0:
#             #         continue
#             #     ch_gen_exp_array = np.array(ch_gen_expert[exp_idx])
#             #     ws_exp[j][exp_idx][i] = wasserstein_distance(
#             #         ch_org_expert[exp_idx][:, i], ch_gen_exp_array[:, i]
#             #     )
#
#     # Calculate the mean WS distances across runs
#     print("WS SHAPE", ws.shape)
#     print("WS", ws)
#     ws_runs = ws.mean(axis=1)  # Mean across channels for each run
#     ws_mean, ws_std = ws_runs.mean(), ws_runs.std()
#
#     ws_exp_runs = ws_exp.mean(axis=2)  # Mean across channels per expert per run
#     ws_mean_exp = ws_exp_runs.mean(axis=0)  # Mean across runs for each expert
#     ws_std_exp = ws_exp_runs.std(axis=0)  # Standard deviation across runs for each expert
#     print("ws mean", f'{ws_mean:.2f}', end=" ")
#
#     return ws_mean, ws_std, ws_mean_exp, ws_std_exp

from models_pytorch_sg import apply_expert_mask_vectorized

def calculate_joint_ws_across_experts(n_calc, y_test_tensor, router_network, generator,
                                      ch_org, noise_dim, device, batch_size=64,
                                      shape_images=(56, 30)):
    """
    Calculates the Wasserstein (WS) distance across the whole distribution
    for a unified generator with multi-expert outputs.
    """

    # Initialize WS distance arrays
    ws = np.zeros((n_calc, 5))  # Overall WS distances

    for j in range(n_calc):  # Perform multiple calculations of the WS distance
        all_fakes = []
        for i in range(0, y_test_tensor.size(0), batch_size):
            cond_batch = y_test_tensor[i: i + batch_size]  # [b, cond_dim]
            noise_batch = torch.randn(cond_batch.size(0), noise_dim, device=device)
            with torch.no_grad():
                gates, logits = router_network(cond_batch)
                imgs = generator(noise_batch, cond_batch)  # [b, E,1,H,W]
                masked = apply_expert_mask_vectorized(imgs, gates)  # [b, H, W] or [b,1,H,W]
            # print("masked shape", masked.shape)
            all_fakes.append(masked.cpu())

        all_fakes = torch.cat(all_fakes, dim=0).squeeze(1)  # [N, H, W] or [N,1,H,W]

        # print('all fakes shape', all_fakes.shape)
        ch_gen_all = pd.DataFrame(sum_channels_parallel(all_fakes.cpu().numpy())).values
        ch_gen_all = np.array(ch_gen_all)  # Convert to numpy array
        # # print("Shape of all generated:", ch_gen_all.shape)
        #
        # Calculate WS distances
        for i in range(5):
            # Overall WS distance
            ws[j][i] = wasserstein_distance(ch_org[:, i], ch_gen_all[:, i])

            # Per-expert WS distances
            # for exp_idx in range(n_experts):
            #     if len(ch_gen_expert[exp_idx]) == 0 or ch_org_expert[exp_idx].shape[0] == 0:
            #         continue
            #     ch_gen_exp_array = np.array(ch_gen_expert[exp_idx])
            #     ws_exp[j][exp_idx][i] = wasserstein_distance(
            #         ch_org_expert[exp_idx][:, i], ch_gen_exp_array[:, i]
            #     )

    # Calculate the mean WS distances across runs
    print("WS SHAPE", ws.shape)
    print("WS", ws)
    ws_runs = ws.mean(axis=1)  # Mean across channels for each run
    ws_mean, ws_std = ws_runs.mean(), ws_runs.std()

    print("ws mean", f'{ws_mean:.2f}', end=" ")

    return ws_mean, ws_std


def get_predictions_from_unified_generator(batch_size, num_samples, noise_dim,
                                           device, y_test, generator, expert_idx, n_experts,
                                           shape_images=(56, 30), input_noise=None):
    """
    Get predictions from a unified generator with multiple expert outputs.

    Args:
        batch_size: Number of samples per batch
        num_samples: Total number of samples to generate
        noise_dim: Dimension of the input noise vector
        device: Device to run the model on
        y_test: Conditioning inputs
        generator: The unified generator model
        expert_idx: The index of the expert whose outputs we want to extract
        n_experts: Total number of experts in the generator
        shape_images: Shape of the output images
        input_noise: Optional pregenerated noise

    Returns:
        numpy array of generated samples for the specified expert
    """
    num_batches = (num_samples + batch_size - 1) // batch_size
    results_all = np.zeros((num_samples, *shape_images))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_size_actual = end_idx - start_idx

        if input_noise is not None:
            noise = input_noise[start_idx:end_idx, :]
        else:
            noise = torch.randn(batch_size_actual, noise_dim, device=device)

        noise_cond = y_test[start_idx:end_idx]

        # Generate results using the unified generator
        with torch.no_grad():
            generator.eval()
            # Generate outputs of shape [batch_size, n_experts, H, W]
            unified_results = generator(noise, noise_cond)

            # Extract the specific expert's output
            # If the shape is [B, n_experts, H, W]
            results = unified_results[:, expert_idx].cpu().numpy()

        # Post-process the results
        results = np.exp(results) - 1
        results_all[start_idx:end_idx] = results.reshape(-1, *shape_images)

    return results_all


def get_predictions_from_experts_results(num_samples, noise_dim,
                                         device, y_test, router_network, experts, shape_images=(56, 30)):
    y_test_tensor = torch.tensor(y_test, device=device)
    results_all = np.zeros((num_samples, *shape_images))

    with torch.no_grad():
        router_network.eval()
        predicted_expert_one_hot = router_network(y_test_tensor).cpu().numpy()
        predicted_expert = np.argmax(predicted_expert_one_hot, axis=1)

    indx_0 = np.where(predicted_expert == 0)[0].tolist()
    indx_1 = np.where(predicted_expert == 1)[0].tolist()
    indx_2 = np.where(predicted_expert == 2)[0].tolist()
    data_cond_0 = y_test_tensor[indx_0]
    data_cond_1 = y_test_tensor[indx_1]
    data_cond_2 = y_test_tensor[indx_2]

    noise_0 = torch.randn(len(data_cond_0), noise_dim, device=device)
    with torch.no_grad():
        experts[0].eval()
        results_0 = experts[0](noise_0, data_cond_0).cpu().numpy()

    noise_1 = torch.randn(len(data_cond_1), noise_dim, device=device)
    with torch.no_grad():
        experts[1].eval()
        results_1 = experts[1](noise_1, data_cond_1).cpu().numpy()

    noise_2 = torch.randn(len(data_cond_2), noise_dim, device=device)
    with torch.no_grad():
        experts[2].eval()
        results_2 = experts[2](noise_2, data_cond_2).cpu().numpy()

    results_0 = np.exp(results_0) - 1 # 40
    results_1 = np.exp(results_1) - 1 # 40
    results_2 = np.exp(results_2) - 1 # 40

    results_all[indx_0] = results_0.reshape(-1, *shape_images)
    results_all[indx_1] = results_1.reshape(-1, *shape_images)
    results_all[indx_2] = results_2.reshape(-1, *shape_images)
    return results_all


# Define the loss function
def regressor_loss(fake_coords, real_coords, router_logits, n_experts):
    B = real_coords.shape[0]

    # Reshape fake coordinates from [B, 2*n_experts] to [B, n_experts, 2]
    # This separates each expert's predictions
    fake_coords_reshaped = fake_coords.view(B, n_experts, 2)

    # Expand real coordinates to match expert dimension
    # [B, 2] -> [B, n_experts, 2]
    real_coords_expanded = real_coords.unsqueeze(1).expand(-1, n_experts, -1)

    # Get router mask using Gumbel-softmax (one-hot selection)
    router_mask = F.gumbel_softmax(router_logits, tau=1.0, hard=True)  # [B, n_experts]

    # Add dimension for broadcasting with coordinates
    # [B, n_experts] -> [B, n_experts, 1]
    router_mask_expanded = router_mask.unsqueeze(-1)

    # Apply mask to both fake and real coordinates
    # This zeros out non-selected experts
    masked_fake_coords = fake_coords_reshaped * router_mask_expanded
    masked_real_coords = real_coords_expanded * router_mask_expanded

    # Calculate squared error per coordinate per expert
    squared_error = (masked_fake_coords - masked_real_coords) ** 2  # [B, n_experts, 2]

    # Sum errors across coordinate dimensions (x,y)
    # [B, n_experts, 2] -> [B, n_experts]
    coord_error_sum = squared_error.sum(dim=2)

    # Count how many samples were assigned to each expert
    expert_sample_counts = router_mask.sum(dim=0)  # [n_experts]

    # Avoid division by zero
    expert_sample_counts = torch.clamp(expert_sample_counts, min=1.0)

    # Calculate mean error per expert
    # Sum across batch dimension and divide by number of samples per expert
    # [B, n_experts] -> [n_experts]
    mse_per_expert = coord_error_sum.sum(dim=0) / expert_sample_counts
    mse_per_expert = mse_per_expert.view(n_experts, 1)
    # Return as [n_experts, 1] tensor
    return mse_per_expert


def calculate_ws_ch_proton_model(n_calc, x_test, y_test, generator,
                                 ch_org, noise_dim, device, batch_size=64):
    ws = np.zeros(5)

    # Ensure y_test is a PyTorch tensor
    y_test = torch.tensor(y_test, device=device)

    num_samples = x_test.shape[0]

    for j in range(n_calc):  # Perform few calculations of the ws distance
        # appends the generated image to the specific indices of the num_batches
        results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                             device, y_test, generator)
        # now results_all contains all images in batch
        try:
            ch_gen = pd.DataFrame(sum_channels_parallel(results_all)).values
            for i in range(5):
                if len(ch_org[:, i]) > 0 and len(ch_gen[:, i]) > 0:
                    ws[i] += wasserstein_distance(ch_org[:, i], ch_gen[:, i])
        except ValueError as e:
            print('Error occurred while generating')
            print(e)
    ws = ws / n_calc  # Average over the number of calculations
    ws_mean = ws.mean()
    print("ws mean", f'{ws_mean:.2f}', end=" ")
    for n, score in enumerate(ws):
        print("ch" + str(n + 1), f'{score:.2f}', end=" ")
    return ws_mean


def evaluate_router(router_network, y_test, expert_number_test, accuracy_metric, precision_metric, recall_metric, f1_metric):
    router_network.eval()
    with torch.no_grad():
        predicted_expert = router_network(y_test)
        _, predicted_labels = torch.max(predicted_expert, 1)

        accuracy = accuracy_metric(predicted_labels, expert_number_test).cpu().item()
        precision = precision_metric(predicted_labels, expert_number_test).cpu().item()
        recall = recall_metric(predicted_labels, expert_number_test).cpu().item()
        f1 = f1_metric(predicted_labels, expert_number_test).cpu().item()

    return accuracy, precision, recall, f1


def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, DI_STRENGTH):
    # Calculate the absolute differences and their means along the batch dimension
    abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=1)
    abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=1)

    # Compute the division term
    div = abs_diff_latent / (abs_diff_noise + 1e-5)

    # Calculate the div_loss
    div_loss = std * DI_STRENGTH / (div + 1e-5)

    # Calculate the final div_loss
    div_loss = torch.mean(std) * torch.mean(div_loss)

    return div_loss

# def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, n_experts, router_logits, DI_STRENGTH):
#     # 1. Repeat both latents n_experts number of times
#     # Calculate the difference between noises and expand it n_expert of times
#     # 2 calculate the mas from the router_logits and apply on the latents
#     # calcualte abs_diff latent shape should be [B, n_experts, 384]
#
#     # repeat the std n_expert of times and calcualte div loss.
#     # Div loss should have shape [n_experts, 1]
#
#     B = fake_latent.shape[0]
#     feature_dim = fake_latent.shape[1]
#
#     # Reshape latents to include expert dimension if needed
#     if len(fake_latent.shape) == 2:  # If shape is [B, feature_dim]
#         # Expand latents to have expert dimension
#         fake_latent = fake_latent.unsqueeze(1).expand(-1, n_experts, -1)  # [B, n_experts, feature_dim]
#         fake_latent_2 = fake_latent_2.unsqueeze(1).expand(-1, n_experts, -1)  # [B, n_experts, feature_dim]
#
#     # print('latent shape', fake_latent.shape)
#     # Calculate absolute differences between latents per expert
#     abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=2)  # [B, n_experts]
#     # print("abs_diff_latent", abs_diff_latent.shape)
#     # Calculate noise differences and expand to match expert dimension
#
#     # print("noise", noise.shape)
#     abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=1, keepdim=True)  # [B, 1]
#     # print("abs_diff_noise", abs_diff_noise.shape)
#     abs_diff_noise = abs_diff_noise.expand(-1, n_experts)  # [B, n_experts]
#
#     # Compute diversity ratio for each expert: latent_diff / noise_diff
#     # Higher values indicate better diversity
#     diversity_ratio = abs_diff_latent / (abs_diff_noise + 1e-5)  # [B, n_experts]
#
#     # print("diversity_ratio", diversity_ratio.shape)
#     # Apply router masking to focus on selected experts
#     router_mask = F.gumbel_softmax(router_logits, tau=1.0, hard=True)  # [B, n_experts]
#
#     # Ensure std has correct shape for broadcasting
#     if isinstance(std, torch.Tensor):
#         if std.dim() == 0:  # scalar
#             std = std.expand(B, n_experts)
#         elif std.shape == (B, 1):
#             std = std.expand(-1, n_experts)  # [B, n_experts]
#
#     else:  # if std is a Python scalar
#         std = torch.ones(B, n_experts, device=router_mask.device) * std
#
#     # Apply mask to focus on expert-specific diversity
#     masked_diversity = diversity_ratio * router_mask  # [B, n_experts]
#     # print("masked_diversity", masked_diversity.shape)
#
#     # Calculate final loss - negative because higher diversity is better
#     # We want to maximize diversity, so we negate the loss
#     div_loss = -DI_STRENGTH * (masked_diversity * std).sum(dim=0)
#     return div_loss

#
# def intensity_regularization(gen_im_proton, intensity_proton, router_logits, IN_STRENGTH):
#     """
#     Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
#     and the mean and standard deviation of the intensity across the batch.
#
#     Args:
#         gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
#         intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
#         IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.
#
#     Returns:
#         torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
#                       sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
#         torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
#         torch.Tensor: The standard deviation of the scaled intensity values across the batch.
#         torch.Tensor: The mean of the scaled intensity values across the batch.
#     """
#
#     # Sum the intensities in the generated images
#     # gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1 #<- this fixed previous bad optimization
#     gen_im_proton_rescaled = torch.exp(gen_im_proton) - 1
#     # print("gen expert", gen_im_proton_rescaled.shape)
#
#     # Calculate sum of intensities for each expert separately
#     B, n_experts, H, W = gen_im_proton_rescaled.shape
#     sum_intensities_per_expert = torch.sum(gen_im_proton_rescaled, dim=[2, 3])  # [B, n_experts]
#     # print("sum int per expert", sum_intensities_per_expert.shape)
#
#     router_mask = F.gumbel_softmax(router_logits, tau=1.0, hard=True)  # [B, n_experts]
#     # print("router mask", router_mask.shape)
#
#
#     # [B, 1] -> [B, n_experts]
#     repeated_intensity = intensity_proton.expand(-1, n_experts)
#     # print("repeated_intensity", repeated_intensity.shape)
#
#     # Apply router mask to get the expected intensity for each expert
#     # This determines how much of the true intensity each expert is responsible for
#     masked_real_intensities = repeated_intensity * router_mask  # [B, n_experts]
#     # print("masked_real_intensities", masked_real_intensities.shape)
#
#     # Calculate absolute difference for each expert separately
#     abs_diff = torch.abs(sum_intensities_per_expert - masked_real_intensities)  # [B, n_experts]
#     # print("mabs_diff", abs_diff.shape)
#
#     # Average the absolute differences across batch dimension for each expert
#     mae_per_expert = torch.mean(abs_diff, dim=0, keepdim=True).t() * IN_STRENGTH  # [n_experts, 1]
#     # print("mabs_diff", mae_per_expert.shape)
#
#     # Calculate statistics per expert
#     std_intensity = torch.std(sum_intensities_per_expert, dim=0)  # [n_experts]
#     mean_intensity = torch.mean(sum_intensities_per_expert, dim=0)  # [n_experts]
#
#     # print("mean_intensity", mean_intensity.shape)
#     return mae_per_expert, sum_intensities_per_expert, std_intensity.clone().detach(), mean_intensity.clone().detach()


def intensity_regularization(gen_im_proton, intensity_proton, IN_STRENGTH):
    """
    Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
    and the mean and standard deviation of the intensity across the batch.

    Args:
        gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
        intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
        IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.

    Returns:
        torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
                      sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
        torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
        torch.Tensor: The standard deviation of the scaled intensity values across the batch.
        torch.Tensor: The mean of the scaled intensity values across the batch.
    """

    # Sum the intensities in the generated images
    # gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1 #<- this fixed previous bad optimization
    gen_im_proton_rescaled = torch.exp(gen_im_proton) - 1
    # print("Gen shape from model", gen_im_proton_rescaled.shape)
    # Gen shape from model torch.Size([138, 1, 56, 30])
    # After sum: torch.Size([138, 1, 1, 1])
    sum_all_axes_p_rescaled = torch.sum(gen_im_proton_rescaled, dim=[2, 3], keepdim=False)  # Sum along all but batch dimension

    # REMOVE THIS RESHAPE BECAUSE IT FLATTENS THE DATA FROM ALL EXPERTS
    # sum_all_axes_p_rescaled = sum_all_axes_p_rescaled.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

    # Compute mean and std as PyTorch tensors
    std_intensity_scaled = sum_all_axes_p_rescaled.std()
    mean_intensity_scaled = sum_all_axes_p_rescaled.mean()

    # # Ensure intensity_proton is correctly shaped and on the same device
    intensity_proton = intensity_proton.view(-1, 1).to(gen_im_proton.device)  # Ensure it is of shape [batch_size, 1]
    assert intensity_proton.shape == sum_all_axes_p_rescaled.shape

    # apply the MASK AS WELL FOR EXPERT COMPUTATIONS TO BOTH THE GENERATED AND REAL DATA
    # OR MAYBE CALCULATE THIS N_EXPERT times each for separate expert. TRY TO MAKE THIS PARALLEL

    # print("sum_all_axes_p_rescaled", sum_all_axes_p_rescaled.shape)
    # print(sum_all_axes_p_rescaled.shape, intensity_proton.shape)
    # Calculate MAE loss
    mae_value_p = F.l1_loss(sum_all_axes_p_rescaled, intensity_proton)*IN_STRENGTH

    return mae_value_p, sum_all_axes_p_rescaled, std_intensity_scaled.detach(),\
           mean_intensity_scaled.detach()


def generate_and_save_images(model, epoch, noise, noise_cond, x_test,
                             photon_sum_proton_min, photon_sum_proton_max,
                             device, random_generator, shape_images=(56, 30)):
    if noise_cond is None:
        return None
    SUPTITLE_TXT = f"\nModel: SDI-GAN data from {random_generator}" \
                   f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
                   f"\nEPOCH: {epoch}"

    # Set the model to evaluation mode
    model.eval()

    # Ensure y_test is a PyTorch tensor
    noise_cond = torch.tensor(noise_cond, device=device)

    # Generate predictions
    with torch.no_grad():
        noise = noise.to(device)
        noise_cond = noise_cond.to(device)
        predictions = model(noise, noise_cond).cpu().numpy().reshape(-1, *shape_images)  # Move to CPU for numpy operations

    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')
    for i in range(0, 12):
        if i < 6:
            x = x_test[i].reshape(*shape_images)
        else:
            x = predictions[i - 6]
        im = axs[i // 6, i % 6].imshow(x, cmap='gnuplot')
        axs[i // 6, i % 6].axis('off')
        fig.colorbar(im, ax=axs[i // 6, i % 6])

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    plt.close(fig)
    return fig


def calculate_expert_distribution_loss(gating_probs, features, lambda_reg=0.1):
    """
    Calculate the regularization loss for the router network to encourage balanced task distribution among experts.

    Args:
        gating_probs (torch.Tensor): The gating probabilities for each sample and expert with shape (batch_size, num_experts).
        features (torch.Tensor): The feature representations of the inputs with shape (batch_size, feature_dim).
        lambda_reg (float): The regularization strength.

    Returns:
        torch.Tensor: The calculated regularization loss.
    """
    # reshape the features from shape (batch_size) to (batch_size, 1)
    # Compute the pairwise Euclidean distance between the features
    pairwise_distances = torch.cdist(features, features, p=2)  # Shape: (batch_size, batch_size)

    # Compute the gating similarities (dot product of gating probabilities for each pair of samples)
    gating_similarities = torch.matmul(gating_probs, gating_probs.T)  # Shape: (batch_size, batch_size)

    reg_loss = torch.sum(gating_similarities * pairwise_distances) / gating_similarities.size(0)

    # Apply the regularization strength
    reg_loss = lambda_reg * reg_loss

    return reg_loss


def calculate_entropy(p):
    """
    Calculate entropy of a probability distribution p.
    """
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


def calculate_expert_utilization_entropy(gating_probs, ENT_STRENGTH=0.1):
    """
    Calculate the expert utilization entropy H_u.
    To promote fair utilization of experts, maximize this term.
    Parameters:
    gating_probs (torch.Tensor): A tensor of shape (N, M) where N is the number of samples
                                 and M is the number of experts. Each entry is the gating
                                 probability of an expert for a given sample.

    Returns:
    torch.Tensor: The entropy of the average gating probabilities.
    """
    avg_gating_probs = torch.mean(gating_probs, dim=0)  # Average over samples. The sum of that is equal roughly to 1
    entropy = calculate_entropy(avg_gating_probs)
    return entropy * ENT_STRENGTH


class StratifiedBatchSampler:
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


def plot_cond_pca_tsne(y_test_temp, indices_experts, epoch):
    """
    Plot PCA and t-SNE 2D projections with data points colored by expert indices.

    Parameters:
    y_test_temp (np.ndarray): The data to project.
    indices_experts (list of np.ndarray): List of indices for each expert.
    N_EXPERTS (int): Number of experts.
    """
    # Create labels for each data point based on indices_experts
    SUPTITLE_TXT = f"\nEPOCH: {epoch}"

    labels = np.zeros(y_test_temp.shape[0], dtype=int)
    for expert_idx, indices in enumerate(indices_experts):
        labels[indices] = expert_idx

    # PCA projection
    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(y_test_temp)

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    y_tsne = tsne.fit_transform(y_test_temp)

    # Plot PCA and t-SNE projections
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')

    # Plot PCA
    scatter = axes[0].scatter(y_pca[:, 0], y_pca[:, 1], c=labels, cmap='viridis', s=10)
    axes[0].set_title('PCA Projection')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    legend1 = axes[0].legend(*scatter.legend_elements(), title="Experts")
    axes[0].add_artist(legend1)

    # Plot t-SNE
    scatter = axes[1].scatter(y_tsne[:, 0], y_tsne[:, 1], c=labels, cmap='viridis', s=10)
    axes[1].set_title('t-SNE Projection')
    axes[1].set_xlabel('Dim 1')
    axes[1].set_ylabel('Dim 2')
    legend2 = axes[1].legend(*scatter.legend_elements(), title="Experts")
    axes[1].add_artist(legend2)

    return fig


def plot_expert_heatmap(y_test, indices_experts, epoch, data_cond_names, num_bins=50, save_path=None):
    """
    Create a heatmap showing the distribution of samples across bins (OX axis) and experts (OY axis) for all 9 variables.

    Parameters:
    y_test (np.ndarray or torch.Tensor): The data containing continuous variables (shape: [n_samples, num_variables]).
    indices_experts (list of np.ndarray): List of indices for each expert.
    num_bins (int): Number of bins to divide the continuous OX axis.

    Returns:
    matplotlib.figure.Figure: The generated figure with 9 subplots.
    """
    # Convert y_test to NumPy array if it is a PyTorch tensor
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

    # Ensure y_test has 9 variables
    num_variables = y_test_np.shape[1]
    if num_variables != len(data_cond_names):
        raise ValueError(f"y_test must have exactly {len(data_cond_names)} variables")

    # Create a figure with 9 subplots (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Sample Distribution Across Experts and Continuous Bins. Epoch {epoch}", fontsize=16)
    print(data_cond_names)
    # Loop through each variable and create a heatmap
    for var_idx, var_name in enumerate(data_cond_names):
        ax = axes[var_idx // 3, var_idx % 3]

        # Define bins for the continuous OX axis
        continuous_variable = y_test_np[:, var_idx]  # Current variable
        bins = np.linspace(continuous_variable.min(), continuous_variable.max(), num_bins + 1)
        bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(num_bins)]

        # Initialize a DataFrame to store counts
        heatmap_data = pd.DataFrame(0, index=[f"E{i+1}" for i in range(len(indices_experts))], columns=bin_labels)

        # Count samples for each expert and bin
        for expert_idx, indices in enumerate(indices_experts):
            binned_data = pd.cut(continuous_variable[indices], bins=bins, labels=bin_labels, include_lowest=True)
            counts = binned_data.value_counts().reindex(bin_labels, fill_value=0)
            heatmap_data.loc[f"E{expert_idx+1}"] = counts.values

        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{var_name}")
        ax.set_xlabel("Bins")
        ax.set_ylabel("Experts")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    save_path = os.path.join(save_path, "cond_data_expert_specialization_2.png")
    # Save the plot if a save path is provided
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_expert_specialization(y_test, indices_experts, epoch, data_cond_names, save_path=None):
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

    num_points_kde = 100

    num_variables = y_test_np.shape[1]
    if num_variables != len(data_cond_names):
        raise ValueError(f"y_test must have exactly {len(data_cond_names)} variables")

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Expert Specialization on Input Data- Epoch {epoch}", fontsize=16)

    colors = ['steelblue', 'darkorange', 'forestgreen']  # Define colors
    expert_labels = [f"Expert {i+1}" for i in range(len(indices_experts))]

    for var_idx, var_name in enumerate(data_cond_names):
        ax = axes[var_idx // 3, var_idx % 3]

        if var_idx == len(data_cond_names) - 1:  # Categorical variable (stacked bar plot)
            categorical_variable = y_test_np[:, var_idx]
            unique_values = np.unique(categorical_variable)

            # Prepare data for the grouped bar plot
            data = []
            for expert_idx, indices in enumerate(indices_experts):
                expert_data = categorical_variable[indices]
                counts = pd.Series(expert_data).value_counts()
                # Ensure all unique values are represented, filling missing values with 0
                expert_counts = [counts.get(val, 0) for val in unique_values]
                data.append(expert_counts)

            # Set the positions and width for the bars
            x = np.arange(len(unique_values))  # the label locations
            width = 0.2  # the width of the bars

            # Plot the grouped bar plot with specified colors
            for i, expert_data in enumerate(data):
                ax.bar(x + (i - 1) * width, expert_data, width, label=expert_labels[i], color=colors[i])

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_title(f"{var_name} (Categorical)")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count (Log Scale)")  # Update y-axis label
            ax.set_yscale('log')  # Set y-axis to log scale
            ax.set_xticks(x)
            ax.set_xticklabels(unique_values)
            ax.legend(loc='upper right', fontsize='x-small')
        else:  # Continuous variables (KDE plots)
            continuous_variable = y_test_np[:, var_idx]
            x_range = np.linspace(continuous_variable.min(), continuous_variable.max(), num_points_kde)

            for expert_idx, indices in enumerate(indices_experts):
                expert_data = continuous_variable[indices]
                kde = stats.gaussian_kde(expert_data)
                y = kde(x_range)
                ax.plot(x_range, y, label=f"Expert {expert_idx + 1}")

            ax.set_title(f"{var_name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(loc='upper right', fontsize='x-small')

    save_path = os.path.join(save_path, "expert_spec_cond_data.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig


def calculate_adaptive_load_balancing_loss(routing_scores, alb_strength=1e-2, eps=1e-6):
    """
    Computes an adaptive load balancing loss that penalizes experts with low routing scores.

    Args:
        routing_scores (torch.Tensor): A 1D tensor of shape (N_experts,) representing the sum
                                       of the routing probabilities (i.e. the "routing score") for each expert over the batch.
        alb_strength (float): A scalar that controls the strength of the load balancing loss.
        eps (float): A small constant to avoid division by zero.

    Returns:
        torch.Tensor: A scalar tensor representing the adaptive load balancing loss.
    """
    # Compute a penalty for each expert: low routing scores result in a high penalty.
    penalties = torch.exp(1.0 / (routing_scores + eps))

    # Average the penalties over all experts.
    loss = penalties.mean()
    return loss*alb_strength
