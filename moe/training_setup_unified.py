from models_pytorch_sg import GeneratorUnified, Discriminator, RouterNetwork, AuxRegUnified, count_model_parameters

import torch
import torch.optim as optim
import os


def setup_experts_unified(N_EXPERTS, N_COND, NOISE_DIM, LR_G, LR_D, LR_A, DI_STRENGTH, IN_STRENGTH, device=torch.device("cuda:0")):
    # Define experts
    generator = GeneratorUnified(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH, N_EXPERTS).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=LR_G)

    num_params_gen = count_model_parameters(generator)
    print(f"Generator model has {num_params_gen} trainable parameters.")

    # Define discriminators
    discriminator = Discriminator(N_COND).to(device)
    num_params_disc = count_model_parameters(discriminator)
    print(f"Discriminator model has {num_params_disc} trainable parameters.")
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR_D)

    aux_reg = AuxRegUnified(N_EXPERTS).to(device)
    num_params_aux = count_model_parameters(aux_reg)
    print(f"Aux Reg model has {num_params_aux} trainable parameters.")
    aux_reg_optimizer = optim.Adam(aux_reg.parameters(), lr=LR_A)

    return generator, generator_optimizer, discriminator, discriminator_optimizer, aux_reg, aux_reg_optimizer,\
           num_params_gen, num_params_disc, num_params_aux

#
# def setup_experts_neutron(N_EXPERTS, N_COND, NOISE_DIM, LR_G, LR_D, LR_A, DI_STRENGTH, IN_STRENGTH,
#                           device=torch.device("cuda:0")):
#     # Define experts
#     generators = []
#     for generator_idx in range(N_EXPERTS):
#         generator = GeneratorNeutron(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH).to(device)
#         # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/gen_{generator_idx}_80.h5"
#         # print(f'Loading weights for {generator_idx} GEN')
#         # generator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
#         generator = generator.to(device)  # or whichever CUDA device you're using
#         generators.append(generator)
#
#     num_params = count_model_parameters(generator)
#     print(f"Generator model has {num_params} trainable parameters.")
#     generator_optimizers = [optim.Adam(gen.parameters(), lr=LR_G) for gen in generators]
#
#     # Define discriminators
#     discriminators = []
#     for generator_idx in range(N_EXPERTS):
#         discriminator = DiscriminatorNeutron(N_COND).to(device)
#         # # load previous weights
#         # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/disc_{generator_idx}_80.h5"
#         # print(f'weights loaded for {generator_idx} DISC')
#         # discriminator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
#         discriminator = discriminator.to("cuda:0")
#         discriminators.append(discriminator)
#     num_params = count_model_parameters(discriminator)
#     print(f"Discriminator model has {num_params} trainable parameters.")
#     discriminator_optimizers = [optim.Adam(disc.parameters(), lr=LR_D) for disc in discriminators]
#
#     # Define aux reg
#     aux_regs = []
#     for generator_idx in range(N_EXPERTS):
#         aux_reg = AuxRegNeutron().to(device)
#         aux_regs.append(aux_reg)
#     num_params = count_model_parameters(aux_reg)
#     print(f"Aux Reg model has {num_params} trainable parameters.")
#     aux_reg_optimizers = [optim.Adam(aux_reg.parameters(), lr=LR_A) for aux_reg in aux_regs]
#
#     # Replace with wrapped version
#     # From training_setup.py
#     base_generator = Generator(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH).to(device)
#     expert_wrapper = ParallelExpertWrapper([base_generator] * N_EXPERTS).to(device)
#
#     return expert_wrapper, generators, generator_optimizers, discriminators, discriminator_optimizers, aux_regs, aux_reg_optimizers
#
#
# def setup_router_attention(cond_dim, n_experts, num_heads, hidden_dim, lr_r, device=torch.device("cuda:0")):
#     router_network = AttentionRouterNetwork(cond_dim=cond_dim, num_experts=n_experts, num_heads=num_heads,
#                                             hidden_dim=hidden_dim)
#     # load previous weights
#     # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/router_network_epoch_80.pth"
#     # print(f'weights loaded for ROUTER')
#     # router_network.load_state_dict(torch.load(expert_weights, map_location='cpu'))
#     router_network = router_network.to(device)
#     router_optimizer = optim.Adam(router_network.parameters(), lr=lr_r)
#     num_params = count_model_parameters(router_network)
#     print(f"Router model has {num_params} trainable parameters.")
#     # Define the learning rate scheduler
#     # router_scheduler = lr_scheduler.ReduceLROnPlateau(router_optimizer, mode='min', patience=3, factor=0.1, verbose=True)
#     return router_network, router_optimizer


def setup_router(N_COND, N_EXPERTS, LR_R=1e-3, device=torch.device("cuda:0")):
    router_network = RouterNetwork(N_COND, N_EXPERTS)
    # load previous weights
    # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/router_network_epoch_80.pth"
    # print(f'weights loaded for ROUTER')
    # router_network.load_state_dict(torch.load(expert_weights, map_location='cpu'))
    router_network = router_network.to(device)
    router_optimizer = optim.Adam(router_network.parameters(), lr=LR_R)
    num_params = count_model_parameters(router_network)
    print(f"Router model has {num_params} trainable parameters.")
    # Define the learning rate scheduler
    # router_scheduler = lr_scheduler.ReduceLROnPlateau(router_optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    return router_network, router_optimizer, num_params


def load_checkpoint_weights(checkpoint_dir,
                            epoch,
                            generators,
                            generator_optimizers,
                            discriminators,
                            discriminator_optimizers,
                            aux_regs,
                            aux_reg_optimizers,
                            router_network,
                            router_optimizer,
                            device="cuda"):
    """
    Load weights for generators, discriminators, auxiliary regularizers, router network,
    and their respective optimizers from a specific checkpoint directory and epoch.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        epoch (int): Epoch for which the weights should be loaded.
        generators (list): List of generator models.
        generator_optimizers (list): List of generator optimizers.
        discriminators (list): List of discriminator models.
        discriminator_optimizers (list): List of discriminator optimizers.
        aux_regs (list): List of auxiliary regularizers.
        aux_reg_optimizers (list): List of auxiliary regularizer optimizers.
        router_network (nn.Module): Router network model.
        router_optimizer (torch.optim.Optimizer): Router network optimizer.
        device (str): Device to load the weights onto.
    """
    # --------- Load Generators (full models) and update optimizers ---------
    for i, gen_opt in enumerate(generator_optimizers):
        gen_file = os.path.join(checkpoint_dir, f"gen_{i}_{epoch}.pth")
        gen_opt_file = os.path.join(checkpoint_dir, f"gen_optim_{i}_{epoch}.pth")

        if os.path.exists(gen_file):
            print(f"Loading generator {i} model from {gen_file}")
            try:
                # Load the entire generator object (not just a state_dict)
                loaded_generator = torch.load(gen_file, map_location=device)
                generators[i] = loaded_generator  # replace with loaded model

                # Since the optimizer was referencing the old model’s parameters,
                # update every parameter group to use the new ones.
                for group in gen_opt.param_groups:
                    group["params"] = list(loaded_generator.parameters())
            except Exception as e:
                print(f"Error loading generator {i} model from {gen_file}: {e}")
        else:
            print(f"Generator {i} model not found for epoch {epoch}")

        if os.path.exists(gen_opt_file):
            print(f"Loading generator {i} optimizer state from {gen_opt_file}")
            try:
                # Open the file in binary mode to avoid zip archive issues.
                with open(gen_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                gen_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading generator {i} optimizer state from {gen_opt_file}: {e}")
        else:
            print(f"Generator {i} optimizer state not found for epoch {epoch}")

    # --------- Load Discriminators (full models) and update optimizers ---------
    for i, disc_opt in enumerate(discriminator_optimizers):
        disc_file = os.path.join(checkpoint_dir, f"disc_{i}_{epoch}.pth")
        disc_opt_file = os.path.join(checkpoint_dir, f"disc_optim_{i}_{epoch}.pth")

        if os.path.exists(disc_file):
            print(f"Loading discriminator {i} model from {disc_file}")
            try:
                loaded_disc = torch.load(disc_file, map_location=device)
                discriminators[i] = loaded_disc
                for group in disc_opt.param_groups:
                    group["params"] = list(loaded_disc.parameters())
            except Exception as e:
                print(f"Error loading discriminator {i} model from {disc_file}: {e}")
        else:
            print(f"Discriminator {i} model not found for epoch {epoch}")

        if os.path.exists(disc_opt_file):
            print(f"Loading discriminator {i} optimizer state from {disc_opt_file}")
            try:
                with open(disc_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                disc_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading discriminator {i} optimizer state from {disc_opt_file}: {e}")
        else:
            print(f"Discriminator {i} optimizer state not found for epoch {epoch}")

    # --------- Load Auxiliary Regularizers (full models) and update optimizers ---------
    for i, aux_opt in enumerate(aux_reg_optimizers):
        aux_file = os.path.join(checkpoint_dir, f"aux_reg_{i}_{epoch}.pth")
        aux_opt_file = os.path.join(checkpoint_dir, f"aux_reg_optim_{i}_{epoch}.pth")

        if os.path.exists(aux_file):
            print(f"Loading auxiliary regularizer {i} model from {aux_file}")
            try:
                loaded_aux = torch.load(aux_file, map_location=device)
                aux_regs[i] = loaded_aux
                for group in aux_opt.param_groups:
                    group["params"] = list(loaded_aux.parameters())
            except Exception as e:
                print(f"Error loading auxiliary regularizer {i} model from {aux_file}: {e}")
        else:
            print(f"Auxiliary regularizer {i} model not found for epoch {epoch}")

        if os.path.exists(aux_opt_file):
            print(f"Loading auxiliary regularizer {i} optimizer state from {aux_opt_file}")
            try:
                with open(aux_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                aux_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading auxiliary regularizer {i} optimizer state from {aux_opt_file}: {e}")
        else:
            print(f"Auxiliary regularizer {i} optimizer state not found for epoch {epoch}")

    # --------- Load Router Network (full model) and update its optimizer ---------
    router_file = os.path.join(checkpoint_dir, f"router_network_{epoch}.pth")
    router_opt_file = os.path.join(checkpoint_dir, f"router_network_optim_{epoch}.pth")

    if os.path.exists(router_file):
        print(f"Loading router network model from {router_file}")
        try:
            loaded_router = torch.load(router_file, map_location=device)
            # Update in-place so that references to router_network remain valid.
            router_network.__dict__.update(loaded_router.__dict__)
        except Exception as e:
            print(f"Error loading router network model from {router_file}: {e}")
    else:
        print(f"Router network model not found for epoch {epoch}")

    if os.path.exists(router_opt_file):
        print(f"Loading router optimizer state from {router_opt_file}")
        try:
            with open(router_opt_file, "rb") as f:
                optimizer_state = torch.load(f, map_location=device)
            # Update the optimizer’s parameter groups to refer to the (updated) router_network.
            for group in router_optimizer.param_groups:
                group["params"] = list(router_network.parameters())
            router_optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"Error loading router optimizer state from {router_opt_file}: {e}")
    else:
        print(f"Router optimizer state not found for epoch {epoch}")