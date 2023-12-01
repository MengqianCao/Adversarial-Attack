import torch

def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    
    # Switch to train mode
    model.train()
    num = 1

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        if num % 100 == 0:
            print('[{:5}/{:5} ({:.0f}%)] Finished\tTrain Loss: {:.6f}'.format(
                    num*inputs.size(0), len(trainloader.dataset),
                    100.0 * num*inputs.size(0) / len(trainloader.dataset), train_loss / (num*inputs.size(0))))
        num +=1

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def adversiral_train(model, trainloader, criterion, optimizer, device, attack_func):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    epsilon = 0.3

    # Switch to train mode
    model.train()
    num = 1

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, _ = attack_func(model,criterion,inputs,labels,device,epsilon)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        if num % 100 == 0:
            print('[{:5}/{:5} ({:.0f}%)] Finished\tTrain Loss: {:.6f}'.format(
                    num*inputs.size(0), len(trainloader.dataset),
                    100.0 * num*inputs.size(0) / len(trainloader.dataset), train_loss / (num*inputs.size(0))))
        num +=1

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy