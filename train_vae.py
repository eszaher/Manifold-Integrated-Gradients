import torch 
import core.config.configuration as cnfg
import core.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url
from core.vae import VAE
from tqdm.notebook import trange, tqdm
from dataset_preparation import vae_dataset


def train():

    # Prepare data 
    train_loader, test_images = vae_dataset.prepare_data_vae()

    # Prepare Model
    device = cnfg.device
    feature_extractor = utils.VGG19().to(device)
    vae_net = VAE(channel_in=cnfg.channel_in, ch=cnfg.channels, blocks=cnfg.blocks, latent_channels=cnfg.latent_channels).to(device)
    optimizer = optim.Adam(vae_net.parameters(), lr=cnfg.lr_vae, betas=cnfg.betas)

    loss_log = []
    for epoch in trange(cnfg.start_epoch, cnfg.nepochs_vae, leave=False):
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        train_perceptual_loss = 0
        
        vae_net.train()
        for i, images in enumerate(tqdm(train_loader, leave=False)):
            images = images.to(device)

            recon_img, mu, logvar = vae_net(images)
            #VAE loss
            kl_loss_ = utils.kl_loss(mu, logvar)
            mse_loss = F.mse_loss(recon_img, images)

            #Perceptual loss
            feat_in = torch.cat((recon_img, images), 0)
            feature_loss = feature_extractor(feat_in)

            loss = kl_loss_ + mse_loss + feature_loss
            
            train_loss += loss.item()
            train_recon_loss += mse_loss.item()
            train_kld_loss += kl_loss_.item()
            train_perceptual_loss += feature_loss.item()
            
            loss_log.append(loss.item())
            vae_net.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = train_loss / len(train_loader)  # Calculate average loss for the epoch
        avg_recon_loss = train_recon_loss / len(train_loader) 
        avg_kld_loss = train_kld_loss / len(train_loader) 
        avg_perceptual_loss = train_perceptual_loss / len(train_loader)

        vae_net.eval()
        with torch.no_grad():
            recon_img, _, _ = vae_net(test_images.to(device))
            img_cat = torch.cat((recon_img.cpu(), test_images), 2)

            vutils.save_image(img_cat,
                            "%s/%s/%s_%d.png" % (cnfg.save_dir, "Results" , cnfg.model_name, cnfg.image_width),
                            normalize=True)

            #Save a checkpoint
            torch.save({
                        'epoch'                         : epoch,
                        'loss_log'                      : loss_log,
                        'model_state_dict'              : vae_net.state_dict(),
                        'optimizer_state_dict'          : optimizer.state_dict()

                        }, cnfg.save_dir + "/Models/"+ cnfg.model_name + ".pt")
        print(f'Epoch {epoch}/{cnfg.nepochs_vae} - Avg Total Loss: {avg_loss} - Avg Recon Loss: {avg_recon_loss}\
        - Avg KLD Loss: {avg_kld_loss} - Avg Percp Loss: {avg_perceptual_loss}')

    return vae_net

def main():
    _ = train()

if __name__ == "__main__":
    train()