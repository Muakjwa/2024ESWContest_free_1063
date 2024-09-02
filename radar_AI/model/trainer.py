import torch
import numpy as np
from sklearn.metrics import accuracy_score

def train_model(model, device, trainloader, criterion, optimizer, num_epochs=25, scheduler = None):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(targets.shape), targets)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item()
        if epoch % 10 == 9:
            print(f'[Epoch {epoch + 1}] loss: {running_loss / 10:.3f}')

    print('Finished Training')

def evaluate_model(model, device, dataloader, criterion):
    inference_label = []
    model.to(device)
    model.eval()
    mse = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            inference_label.append(outputs[0].cpu())
            mse += criterion(outputs.view(targets.shape), targets).item()

    print(f'Mean Squared Error on the test dataset: {mse / len(dataloader):.3f}')
    return inference_label


def train_model_clf(model, device, trainloader, criterion, optimizer, num_epochs=25, scheduler = None):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            running_loss += loss.item()
        if epoch % 10 == 9:
            print(f'[Epoch {epoch + 1}] loss: {running_loss / 10:.3f}')

    print('Finished Training')

def evaluate_model_clf(model, device, dataloader, criterion):
    inference_label = []
    ground_truth = []
    model.to(device)
    model.eval()
    mse = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            inference_label.append(int(np.argmax(outputs.cpu(), axis=1).reshape(1)[0]))
            ground_truth.append(int(targets.cpu()))
            loss = criterion(outputs, targets.view(-1))

    accuracy = accuracy_score(ground_truth, inference_label)

    print(f'Sleep Stage Classification Accuracy: {accuracy:.3f}')
    return inference_label