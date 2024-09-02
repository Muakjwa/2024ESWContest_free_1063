import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label_data = self.labels[idx]
        return input_data, label_data
    

def train_test_dataloader(input_data, label, test_ratio = 0.1):
    entire_len = len(label)
    test_len = int(entire_len * test_ratio)

    train_dataset = CustomDataset(torch.tensor(input_data).unsqueeze(1).float()[:entire_len - test_len], label[:entire_len - test_len])
    test_dataset = CustomDataset(torch.tensor(input_data).unsqueeze(1).float()[entire_len - test_len:], label[entire_len - test_len:])

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return trainloader, testloader

def train_dataloader(input_data, label, batch_size = 64, shuffle = True):
    train_dataset = CustomDataset(input_data.clone().detach().float(), label)

    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=shuffle)

    return trainloader

def test_dataloader(input_data, label, batch_size = 1):
    test_dataset = CustomDataset(input_data.clone().detach().float(), label)

    testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    return testloader