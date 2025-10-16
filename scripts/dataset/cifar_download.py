import torchvision
import torchvision.transforms as transforms

# Target directory
data_dir = "/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/cifar10_dataset/"

# Define transform (basic: convert to tensor & normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10 training set
trainset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

# Download CIFAR-10 test set
testset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)
