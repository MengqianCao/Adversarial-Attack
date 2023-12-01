import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tools.dataloader import load_dataset
from tools.conduct import test
from tools.attack import fgsm
from tools.attack import pgd
from tools.visualize import plot_adv_images
from matplotlib import pyplot as plt

def epsilon_compare(epsilon_values, adversarial_accuracies, attack_success_rates):
    if len(epsilon_values) != len(adversarial_accuracies) or len(epsilon_values) != len(attack_success_rates):
        print("Error: Input lists have different lengths.")
        return
    plt.figure(figsize=(10, 6))

    plt.plot(epsilon_values, adversarial_accuracies, 'o-', label='Adversarial Accuracy')
    plt.plot(epsilon_values, attack_success_rates, 'o-', label='Attack Success Rate')

    for i in range(len(epsilon_values)):
        plt.text(epsilon_values[i], adversarial_accuracies[i], f"{adversarial_accuracies[i]:.2f}", ha='center', va='bottom')
        plt.text(epsilon_values[i], attack_success_rates[i], f"{attack_success_rates[i]:.2f}", ha='center', va='bottom')

    plt.xlabel('Epsilon')
    plt.ylabel('Percentage')
    plt.title('Comparison of Adversarial Accuracies and Attack Success Rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ep_list_results')
    plt.show()

def test_adversarial(model, testloader, criterion, device, epsilon, attack_func):
    adversarial_correct = 0
    attack_success = 0
    total = 0

    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        adversarial_images, _ = attack_func(model, criterion, images, labels, device, epsilon)

        adversarial_outputs = model(adversarial_images)

        _, adversarial_predicted = torch.max(adversarial_outputs.data, 1)

        adversarial_correct += (adversarial_predicted == labels).sum().item()
        attack_success += (adversarial_predicted != labels).sum().item()
        total += labels.size(0)

    adversarial_accuracy = 100.0 * adversarial_correct / total
    attack_success_rate = 100.0 * attack_success / total
    print(f'Epsilon = {epsilon}:')
    print(f'Adversarial Accuracy: {adversarial_accuracy:.2f}%')
    print(f'Attack Success Rate: {attack_success_rate:.2f}%')
    print('------------------------------------------------------')
    return adversarial_accuracy, attack_success_rate

if __name__ == '__main__':
    try: func = sys.argv[1]
    except: func = 'fgsm'
    if func =='fgsm':
        attack_func = fgsm
        model_path = 'best_fgsm_model.pth'
    elif func =='pgd':
        attack_func = pgd
        model_path = 'best_pgd_model.pth'
    else:
        exit()

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
    epsilon_values = [0.01, 0.03, 0.07, 0.1, 0.3, 0.5]

    # Load the best model
    best_model = models.resnet50(pretrained=True)
    # Modify conv1 to suit CIFAR-10
    best_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    best_model.fc = nn.Linear(num_features, num_classes)
    # Load checkpoints
    checkpoint = torch.load(model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    test_accuracy = checkpoint['test_accuracy']
    best_model = best_model.to(device)
    print("Best Trained Model Loaded!")
    print(f"Checkpoint at Epoch {epoch+1} with accuracy of {test_accuracy}%")

    # Evaluate adversarial attacks for each epsilon value
    adversarial_accuracies = []
    attack_success_rates = []
    print("Attack function: "+func)
    print('--------------------START--TESTING--------------------')

    for e in epsilon_values:
        adversarial_accuracy, attack_success_rate = test_adversarial(best_model, testloader, criterion, device, e, attack_func)
        adversarial_accuracies.append(adversarial_accuracy)
        attack_success_rates.append(attack_success_rate)
    epsilon_compare(epsilon_values, adversarial_accuracies, attack_success_rates)
    epsilon = epsilon_values[attack_success_rates.index(max(attack_success_rates))]

    num_images = 4
    # Visualize some adversarial examples
    plot_adv_images(testset, best_model, criterion, classes, device, epsilon, num_images,attack_func)

