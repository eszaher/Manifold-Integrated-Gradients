import torch.nn.functional as F
import core.config.configuration as cnfg
import core.utils as utils
import torch

def target_attack_ig(model, baseline, target_class, x, x_target):
    """
    Targeted atrributional attack on Integrated Gradients 
    """
    
    x_adv = x.clone().detach().requires_grad_()

    org_expl, org_acc, org_idx = utils.get_expl(model, x, baseline, target_class)
    target_expl, _, _ = utils.get_expl(model, x_target, baseline, target_class)
    target_expl = target_expl.detach()
    target_expl =  utils.normalize(target_expl)
    
    optimizer_adv = torch.optim.Adam([x_adv], lr=cnfg.lr_attack)
    
    for i in range(cnfg.num_iterations_att):

        optimizer_adv.zero_grad()
        # calculate loss
        adv_expl, adv_acc, class_idx = utils.get_expl(model, x_adv, baseline, target_class)
        adv_expl = utils.normalize(adv_expl)
        loss_expl = F.mse_loss(adv_expl, target_expl)
        loss_output = F.mse_loss(adv_acc, org_acc.detach())
        #input_loss = F.mse_loss(x.detach(), x_adv.detach())
        total_loss = cnfg.expl_loss_prefactor*loss_expl + cnfg.class_loss_prefactor*loss_output 

        # update adversarial example
        total_loss.backward()
        optimizer_adv.step()
        if class_idx != org_idx:
            print("class index changed")
            break

        # clamp adversarial example
        x_adv.data = utils.clamp(x_adv.data)

        print("{}: Tot_Lss: {}, Expl_Lss: {}, Out_Lss: {} ".format(i, total_loss.item(),loss_expl.item(),loss_output.item()))

    adv_expl, adv_acc, class_idx = utils.get_expl(model, x_adv, baseline, target_class)
    
    return x_adv, adv_expl, org_expl, target_expl
    
