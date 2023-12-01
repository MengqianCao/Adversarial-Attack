import torch

def fgsm(model, criterion, images, labels, device, epsilon):
    images.requires_grad_ (True)
    outputs = model(images)
    loss = criterion(outputs, labels).to(device)
    model.zero_grad()
    loss.backward()

    gradient = images.grad.data
    perturbations = epsilon * torch.sign(gradient)
    adversarial_images = images + perturbations
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images, perturbations


def pgd(model, criterion, images, labels, device, epsilon, alpha=2 / 255, iters=40):
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        lost = criterion(outputs, labels).to(device)
        lost.backward()

        adv_images = images + alpha * images.grad.sign()
        perturbations = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + perturbations, min=0, max=1).detach_()

    return images, perturbations