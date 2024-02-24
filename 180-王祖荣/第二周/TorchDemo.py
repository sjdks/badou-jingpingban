import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 多个输出神经元
        self.activation = nn.Softmax(dim=1)  # 使用 Softmax 激活函数
        self.loss = nn.CrossEntropyLoss()  # 多分类任务使用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成样本，使用整数表示类别
def build_sample_multi_class(num_classes):
    x = np.random.random(5)
    label = np.random.randint(0, num_classes)
    return x, label


# 生成数据集，多分类任务
def build_dataset_multi_class(total_sample_num, num_classes):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample_multi_class(num_classes)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 训练过程中的评估函数
def evaluate_multi_class(model, num_classes):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset_multi_class(test_sample_num, num_classes)
    with torch.no_grad():
        y_pred = model(x)
        correct = sum(torch.argmax(y_pred, dim=1) == y).item()
    accuracy = correct / test_sample_num
    print(f"Correct predictions: {correct}, Accuracy: {accuracy}")
    return accuracy


# 在 main 函数中调用 evaluate_multi_class 时传递 num_classes 参数
def main_multi_class():
    num_classes = 3  # 设置类别数量
    input_size = 5
    model = TorchModel(input_size, num_classes)
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    learning_rate = 0.001

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset_multi_class(train_sample, num_classes)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("=========\nEpoch %d, Average Loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate_multi_class(model, num_classes)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model_multi_class.pt")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_multi_class()