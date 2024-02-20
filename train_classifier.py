import torch 
import core.config.configuration as cnfg
import core.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from core.vae import VAE
from tqdm.notebook import trange, tqdm
from dataset_preparation import classifier_dataset
from core import classifier as clf



def train_clf():
    # Prepare data 
    train_loader, test_loader = classifier_dataset.prepare_data_classifier()

    # Prepare Model
    device = cnfg.device
    if cnfg.backbone_type == "resnet":
        classifier = clf.ResClassifier().to(device)
    elif cnfg.backbone_type == "vgg16":
        classifier = clf.VGG16Classifier().to(device)
    else:
        classifier = clf.InceptionClassifier().to(device)

    vae_net = VAE(channel_in=cnfg.channel_in, ch=cnfg.channels, blocks=cnfg.blocks, latent_channels=cnfg.latent_channels).to(device)
    vae_checkpoint = torch.load(cnfg.save_dir + "/Models/" + cnfg.model_name + ".pt", map_location="cpu")
    print("VAE checkpoint loaded")
    vae_net.load_state_dict(vae_checkpoint['model_state_dict'])

    optimizer = optim.Adam(classifier.parameters(), lr=cnfg.lr_classifier, weight_decay=cnfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    vae_net.eval()

    def test_model():
        classifier.eval()
        val_loss = 0
        val_corrects = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                reconstructed_data, _, _ = vae_net(data)
                
                output = classifier(reconstructed_data)
                val_loss = criterion(output, target)  
                
                _, predictions = torch.max(output.data, 1)
            
                val_corrects += torch.sum(predictions.squeeze().int() == target.squeeze().int()).item()
                
                val_loss += val_loss.item() * data.size(0)
        
        # Print epoch statistics
        epoch_loss = val_loss / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_loss, val_corrects, len(test_loader.dataset),
            100.0 * float(val_corrects) / len(test_loader.dataset)))
        
        torch.save({
                        'model_state_dict'              : classifier.state_dict(),
                        'optimizer_state_dict'          : optimizer.state_dict()
                        }, cnfg.save_dir + "/Models/"+ cnfg.model_name + "_" + "classifier_"+ cnfg.backbone_type + ".pt")

        return epoch_loss

    # Existing training loop starts here...
    for epoch in range(cnfg.n_epochs_classifier):
        training_corrects = 0
        classifier.train()
        training_loss = 0
        total_train = 0 
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
            data, target = data.to(device), target.to(device)
            #target = target.unsqueeze(1).float()
            reconstructed_data, _, _ = vae_net(data)

            optimizer.zero_grad()
            
            output = classifier(reconstructed_data)
            loss = criterion(output, target)

            loss.backward()
            
            optimizer.step()
            
            training_loss += loss.item() * data.size(0)
            _, predictions = torch.max(output.data, 1)
            
            training_corrects += torch.sum(predictions.squeeze().int() == target.squeeze().int()).item()

        
        # Print epoch statistics
        epoch_loss = training_loss / len(train_loader.dataset)
        print('Epoch: {} Average training loss: {:.4f}, Training Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, epoch_loss, training_corrects, len(train_loader.dataset),
            100.0 * float(training_corrects) / len(train_loader.dataset)))
        # Test the model at the end of each epoch
        test_model()

    # Finetuning
    for param in classifier.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(classifier.parameters(), lr=cnfg.lr_finetuning, weight_decay=cnfg.weight_decay)

    print("Finetuning")
    for epoch in range(cnfg.n_epochs_finetuning):
        training_corrects = 0
        classifier.train()
        training_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
            data, target = data.to(device), target.to(device)
            #target = target.unsqueeze(1).float()
            reconstructed_data, _, _ = vae_net(data)

            optimizer.zero_grad()
            
            output = classifier(reconstructed_data)
            loss = criterion(output, target)

            loss.backward()
            
            optimizer.step()
            
            training_loss += loss.item() * data.size(0)
            _, predictions = torch.max(output.data, 1)
            
            training_corrects += torch.sum(predictions.squeeze().int() == target.squeeze().int()).item()

        epoch_loss = training_loss / len(train_loader.dataset)
        print('Epoch: {} Average training loss: {:.4f}, Training Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, epoch_loss, training_corrects, len(train_loader.dataset),
            100.0 * float(training_corrects) / len(train_loader.dataset)))

        test_model()

    return classifier, vae_net

def main():
    _, _ = train_clf()

    
if __name__ == "__main__":
    train_clf()