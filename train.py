from models import CNNClassifier, save_model, ClassificationLoss
from utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import IPython
from torchsummary import summary
def train():
    model = CNNClassifier()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95, weight_decay=1e-6)
    epochs = 10

    data = load_data('data/train')

    for epoch in range(epochs):
        model.train()
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    args = parser.parse_args()
    train()