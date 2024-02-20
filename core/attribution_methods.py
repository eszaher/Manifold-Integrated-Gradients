import torch
import core.config.configuration as cnfg


def integrated_gradients(model, baseline, image, target_class, steps=cnfg.num_interpolants):
    """
    Compute the integrated gradients for a given model and image.

    Parameters:
    - model: the PyTorch model (should be in eval mode)
    - baseline: the baseline image (usually a tensor of zeros with the same shape as the image)
    - image: the input image tensor
    - target_class: the index of the target class for which gradients should be calculated
    - steps: the number of steps in the Riemann sum approximation of the integral

    Returns:
    - Integrated gradients with respect to the input image
    """
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Generate scaled versions of the image
    scaled_images = [baseline + (float(i) / steps) * (image - baseline) for i in range(0, steps)]

    # Convert to tensor
    scaled_images = torch.cat([img.unsqueeze(0) for img in scaled_images], dim=0)
    
    #print("SSSS",scaled_images.shape)
    scaled_images = scaled_images.view(steps, cnfg.channel_in, cnfg.image_width, cnfg.image_width)
    
    # Require gradient
    #scaled_images.requires_grad_(True)
    scaled_images = scaled_images.requires_grad_(True)
    
    # Get model predictions for the scaled images
    model_outputs = model(scaled_images)
    m = torch.nn.Softmax(dim=1)
    model_outputs = m(model_outputs)

    # Extract the scores of the target class
    scores = model_outputs[:, target_class]
    #print(scores[-1])
    
    # Compute gradients
    #scores.backward(torch.ones_like(scores))
    
    # Retrieve gradients; gradients are now a tensor of the same shape as scaled_images
    gradients = torch.autograd.grad(outputs=scores, inputs=scaled_images,
                                    grad_outputs=torch.ones(scores.size()).to(image.device),
                                    create_graph=True)[0]

    
    # Average the gradients across all steps
    avg_gradients = torch.mean(gradients, dim=0)
    
    # Compute the integrated gradients
    integrated_gradients = (image - baseline) * avg_gradients
    
    return integrated_gradients

def integrated_gradients_geo(model, geodesic_imgs, target_class, steps=cnfg.num_interpolants):
    """
    Compute the integrated gradients for a given model and image.

    Parameters:
    - model: the PyTorch model (should be in eval mode)
    - geodesic_imgs: images on the geodesic path
    - target_class: the index of the target class for which gradients should be calculated
    - steps: the number of steps in the Riemann sum approximation of the integral

    Returns:
    - Integrated gradients with respect to the input image
    """
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Generate scaled versions of the image
    scaled_images = geodesic_imgs #[baseline + (float(i) / steps) * (image - baseline) for i in range(0, steps)]

    # Convert geodesic images to tensors
    scaled_images = torch.cat([img.unsqueeze(0) for img in scaled_images], dim=0)
    
    scaled_images = scaled_images.view(steps, cnfg.channel_in, cnfg.image_width, cnfg.image_width)
    
    # Require gradient
    #scaled_images.requires_grad_(True)
    scaled_images = scaled_images.requires_grad_(True)
    
    # Get model predictions for the scaled images
    model_outputs = model(scaled_images)
    m = torch.nn.Softmax(dim=1)
    model_outputs = m(model_outputs)

    # Extract the scores of the target class
    scores = model_outputs[:, target_class]
    
    # Retrieve gradients; gradients are now a tensor of the same shape as scaled_images
    gradients = torch.autograd.grad(outputs=scores, inputs=scaled_images,
                                    grad_outputs=torch.ones(scores.size()).to(scaled_images[0].device),
                                    create_graph=True)[0]

    
    # Average the gradients across all steps
    avg_gradients = torch.mean(gradients, dim=0)

    # Compute IG along geodesics
    integrated_gradients = (scaled_images[-1] - scaled_images[0]) * avg_gradients
    
    return integrated_gradients
