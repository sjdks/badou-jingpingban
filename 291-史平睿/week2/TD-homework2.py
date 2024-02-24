import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

"""
实现一个自行构建的找规律（基于pytorch框架编写机器学习模型训练任务）
规律：x是一个5维向量，如果第1个数最大，输出0；第二个数最大，输出1；
第三个数最大，输出2；第四个数最大，输出3；第五个数最大，输出4
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    index = np.argmax(x)
    if index == 0:
        return x, 0
    elif index == 1:
        return x, 1
    elif index == 2:
        return x, 2
    elif index == 3:
        return x, 3
    else:
        return x, 4

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    #return torch.FloatTensor(X), torch.FloatTensor(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("x:", x)
    print("y:", y)
    count_dict = Counter(y.tolist())
    print("count_dict:", count_dict)
    print("本次预测集中共有：")
    for key, value in count_dict.items():
        print(f"{key} 类样本 {value} 个")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        print("y_pred:", y_pred)

        for y_p, y_t in zip(y_pred, y):
            print("y_p:", y_p)
            print("y_t:", y_t)
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)  # 创建训练集，正常任务是读取训练集
    for epoch in range(epoch_num):  # 训练过程
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]
            loss = model.forward(x, y)  # 输入真实值和标签，计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 梯度更新
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("============\n第%d轮平均loss:%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")  # 保存模型

    print(log)  # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    #print("result:", result)
    for vec, res in zip(input_vec, result):
        print("输入：%s，预测类别：%d，概率值：%f" % (vec, np.argmax(res), res[np.argmax(res)]))
        ## 存储四舍五入后的子列表
        #rounded_sublist = []
        ## 遍历子列表中的每个元素并四舍五入
        #for i in range(len(res)):
        #    rounded_sublist.append(round(res[i]))  # 四舍五入后的列表
        #print("输入：%s，预测类别：%s，概率值：%s" % (vec, rounded_sublist, res))


if __name__ == "__main__":
    main()
    #test_vec, _ = build_dataset(10)
    ##print("test_vec.tolist():", test_vec.tolist())
    #predict("model.pt", test_vec)
