import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 32, 3, 1, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, 1, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(7*7*64, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = datasets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = datasets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=128,
                         shuffle=True,
                         drop_last=True)
                         
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("[+] Train Start")
total_epochs = 20
total_batch = len(data_loader)
for epoch in range(total_epochs):
    avg_cost = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        cost = criterion(y_pred, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost
    avg_cost /= total_batch

    print("Epoch: %d, Cost: %f" % (epoch, avg_cost))

print("[+] Test Start")
x = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
y = mnist_test.targets.to(device)
y_pred = model(x)
correct = torch.argmax(y_pred, 1) == y
accuracy = correct.float().mean()
print("[+] Accuracy:", accuracy.item())