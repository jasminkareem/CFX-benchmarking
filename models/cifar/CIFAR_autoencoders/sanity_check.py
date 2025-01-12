import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

def create_model():

    class Autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoder, self).__init__()
            # Input size: [batch, 3, 32, 32]
            # Output size: [batch, 3, 32, 32]
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
                nn.ReLU(),
    # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
    #             nn.ReLU(),
            )
            self.decoder = nn.Sequential(
    #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
    #             nn.ReLU(),
                nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
                nn.Sigmoid(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded

    return Autoencoder()


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



def visualize_results(images, decoded_imgs, title, filename):
    import matplotlib.pyplot as plt

    images = images.cpu().numpy().transpose((0, 2, 3, 1))
    decoded_imgs = decoded_imgs.cpu().detach().numpy().transpose((0, 2, 3, 1))

    num_images = min(8, len(images))  # Display up to 8 images
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Original images
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        # Reconstructed images
        axes[1, i].imshow(decoded_imgs[i])
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.suptitle(title)
    plt.tight_layout()

    # Save the plot to a PDF
    os.makedirs('./plots', exist_ok=True)
    pdf_path = f"./plots/{filename}"
    plt.savefig(pdf_path, format="pdf")
    print(f"Saved plot to {pdf_path}")

    plt.close(fig)
  

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder for each class")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    print("Validating autoencoder trained on the full dataset...")
    autoencoder_full = create_model()
    autoencoder_full.load_state_dict(torch.load("./weights/autoencoder.pth"))
    autoencoder_full.eval()
    if torch.cuda.is_available():
        autoencoder_full.cuda()

    # Visualize the full dataset's reconstruction
    full_testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    dataiter = iter(full_testloader)
    images, labels = next(dataiter)
    images = get_torch_vars(images)

    # Forward pass to get reconstructed images
    _, decoded_imgs = autoencoder_full(images)

    # Save visualization to PDF
    visualize_results(images, decoded_imgs, "Full Dataset Autoencoder", "full_set_reconstruction.pdf")


    for class_idx, class_name in enumerate(classes):
        print("here")
        # Filter dataset for the current class
        train_indices = [i for i, (_, label) in enumerate(trainset) if label == class_idx]
        test_indices = [i for i, (_, label) in enumerate(testset) if label == class_idx]

        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, train_indices), batch_size=16, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, test_indices), batch_size=16, shuffle=False, num_workers=2)


        print(f"Validating autoencoder for class: {class_name}")
        autoencoder = create_model()
        autoencoder.load_state_dict(torch.load(f"./weights/autoencoder_{class_name}.pth"))
        autoencoder.eval()
        if torch.cuda.is_available():
            autoencoder.cuda()
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images = get_torch_vars(images)
        _, decoded_imgs = autoencoder(images)
        # Visualization
        visualize_results(images, decoded_imgs, f"Class: {class_name}", f"{class_name}_reconstruction.pdf")

    
    

if __name__ == "__main__":
    main()