from matplotlib import pyplot as plt
import random
import torch

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    #plt.show()


def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
   # plt.show()

def plot_image(dataset, model, classes, device):
    idx = random.randint(0, len(dataset))
    label = dataset[idx][1]
    img = dataset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    # Convert the image and show it
    img = img.squeeze().permute(1, 2, 0).cpu()  # Move the image tensor back to the CPU and adjust dimensions
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.savefig('predicted_image.png')
    plt.show()
    print("Predicted label: ", classes[predicted[0].item()])
    print("Actual label: ", classes[label])

def plot_adv_images(dataset, model, criterion, classes, device, epsilon, num_images,attack_func):

    clean_images = []
    clean_labels = []
    for _ in range(num_images):
        index = random.randint(0, len(dataset))
        image, label = dataset[index]
        clean_images.append(image)
        clean_labels.append(label)

    clean_images = torch.stack(clean_images).to(device)
    clean_labels = torch.tensor(clean_labels).to(device)

    adversarial_images, perturbations = attack_func(model, criterion, clean_images, clean_labels, device, epsilon)

    fig, axes = plt.subplots(num_images, 5, figsize=(15, 10))

    for i in range(num_images):
        clean_img = clean_images[i].cpu().permute(1, 2, 0).detach().numpy()
        perturbation = perturbations[i].cpu().permute(1, 2, 0).detach().numpy()
        adversarial_img = adversarial_images[i].cpu().permute(1, 2, 0).detach().numpy()

        axes[i, 0].imshow(clean_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Clean: {classes[clean_labels[i]]}', fontweight='bold', color='green')

        axes[i, 1].axis('off')
        axes[i, 1].text(0.5, 0.5, '+', fontsize=40, ha='center', va='center')
        axes[i, 1].set_title('')

        axes[i, 2].imshow(perturbation)
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Perturbation')

        axes[i, 3].axis('off')
        axes[i, 3].text(0.5, 0.5, '=', fontsize=40, ha='center', va='center')
        axes[i, 3].set_title('')

        axes[i, 4].imshow(adversarial_img)
        axes[i, 4].axis('off')
        axes[i, 4].set_title(f'Adversarial: {classes[model(adversarial_images[i].unsqueeze(0)).argmax().item()]}', fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('Generated_Adversarial_examples.png')
    plt.show()
