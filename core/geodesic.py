import torch
import core.config.configuration as cnfg
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

import gc
gc.collect()
torch.cuda.empty_cache()


def compute_etta(model, zi, zi_minus, zi_plus, dt):
    # Compute the finite difference
    g_zi_minus = model.decoder(zi_minus).view(-1)
    g_zi = model.decoder(zi).view(-1)
    g_zi_plus = model.decoder(zi_plus).view(-1)

    finite_diff = (g_zi_plus - 2 * g_zi + g_zi_minus) / dt
    finite_diff = finite_diff.view(1, cnfg.channel_in , cnfg.image_width, cnfg.image_width)  # Reshape it to match the encoder's output shape

    # a wrapper function for the encoder, so it can handle just the required output
    def partial_encoder(input_data):
        return model.encoder(input_data)[1]

    # Compute Jacobian-vector product
    vjp_outputs = torch.autograd.functional.jvp(partial_encoder, g_zi.view(1, cnfg.channel_in, cnfg.image_width, cnfg.image_width), finite_diff)

    Jv = vjp_outputs[1].view_as(zi)

    # Compute etta_i
    etta_i = -Jv

    # Free up memory
    del g_zi_minus, g_zi, g_zi_plus, finite_diff, Jv, vjp_outputs
    torch.cuda.empty_cache()
    gc.collect()

    return etta_i


def compute_etta_d(model, zi, zi_minus, zi_plus, dt):
    # Compute etta using the decoder 
    # Compute the finite difference
    g_zi_minus = model.decoder(zi_minus).view(-1)
    g_zi = model.decoder(zi).view(-1)
    g_zi_plus = model.decoder(zi_plus).view(-1)

    finite_diff = (g_zi_plus - 2 * g_zi + g_zi_minus) / dt
    finite_diff = finite_diff.view(1, cnfg.channel_in, cnfg.image_width, cnfg.image_width)  # Reshape it to match the encoder's output shape

    # Compute Jacobian-vector product
    vjp_outputs = torch.autograd.functional.vjp(model.decoder, zi, finite_diff)
    # Get the result from the vjp outputs
    Jv = vjp_outputs[1].view_as(zi)

    # Compute etta_i
    etta_i = -Jv

    # Free up memory
    del g_zi_minus, g_zi, g_zi_plus, finite_diff, Jv, vjp_outputs
    torch.cuda.empty_cache()

    return etta_i

def backtracking_line_search(model, z_collection, i, direction, start_alpha, dt, beta, c=cnfg.c):
    alpha = start_alpha
    current_energy = sum_of_etta_norms(model, z_collection, dt)
    gradient_norm_square = direction.norm().pow(2)

    
    tmp_z = z_collection[i] - alpha * direction
    new_z_collection = [z.clone() for z in z_collection]
    new_z_collection[i] = tmp_z
    #iterations_count = 0
    
    while sum_of_etta_norms(model, new_z_collection, dt) > current_energy - c * alpha * gradient_norm_square:
        alpha *= beta
        tmp_z = z_collection[i] - alpha * direction
        new_z_collection[i] = tmp_z
        #iterations_count+=1
        
    return alpha

def sum_of_etta_norms(model, z_collection, dt):
    norms = []
    for j in range(1, len(z_collection) - 1):
        etta_j = compute_etta(model, z_collection[j], z_collection[j-1], z_collection[j+1], dt)
        norms.append(etta_j.norm().pow(2).item())
        del etta_j
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    return sum(norms)

def sum_energy_norms(model, z_collection, dt):
    norms = []
    for j in range(1, len(z_collection) - 1):
        etta_j = compute_etta_d(model, z_collection[j], z_collection[j-1], z_collection[j+1], dt)
        norms.append(etta_j.norm().pow(2).item())
        #del etta_j
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    return sum(norms)
    



def geodesic_path_algorithm(model, z_collection, alpha, T, beta, epsilon, max_iterations):
    model.eval()
    dt = 1.0 / T
    initial_sum_norms = float('inf')
    iterations = 0

    while sum_energy_norms(model, z_collection, dt) > epsilon:
        print(f"It{iterations}:total_VEnergy", sum_energy_norms(model, z_collection, dt))
        
        etta_norms = []
        if sum_energy_norms(model, z_collection, dt) > initial_sum_norms:
                initial_sum_norms = sum_energy_norms(model, z_collection, dt)
        else:
            pass
    
        if iterations == max_iterations:
            break

        iterations +=1
        
        for i in range(1, T-1):
            etta_i = compute_etta_d(model, z_collection[i], z_collection[i-1], z_collection[i+1], dt)
            # Line search can be time consuming, instead use a small default alpha
            #alpha_i = backtracking_line_search(model, z_collection, i, etta_i, alpha, dt, beta)
            alpha_i= alpha
            z_collection[i] -= alpha_i * etta_i
            etta_norms.append(etta_i.norm().pow(2).item())
            
            del etta_i
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
    return z_collection
