import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import random
from tools.dataloader import load_dataset
from tools.conduct import train
from tools.conduct import test
from tools.visualize import plot_loss
from tools.visualize import plot_accuracy

def train_epochs(model, trainloader, testloader, criterion, optimizer, device, num_epochs):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'-------------------> Epoch {epoch + 1}/{num_epochs} <-------------------')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')
        print()

        # Save the model if the current test accuracy is higher than the best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy' : test_accuracy
            }
            torch.save(checkpoint, 'best_model.pth')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

if __name__ == '__main__':

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load the dataset
    num_classes = 10
    batch_size = 64
    trainset, trainloader, testset, testloader, classes = load_dataset(batch_size)

    # Load the pre-trained model
    model = models.resnet50(pretrained=True)
    # Modify conv1 to suit CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify the final fully connected layer according to the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 60
    

    
    print("Training the model...")
    # Train the model
    model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(
          model, trainloader, testloader, criterion, optimizer, device, num_epochs)
    # Plot the loss and accuracy curves
    plot_loss(train_losses, test_losses)
    plot_accuracy(train_accuracies, test_accuracies)