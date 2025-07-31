import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示问题


# 特征提取类 - 词袋模型
class Bag:
    def __init__(self, my_data, max_item=1000, test_rate=0.3, random_state=2025):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()
        self.len = 0
        self.train, self.test = train_test_split(
            self.data, test_size=test_rate, random_state=random_state
        )
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for term in self.data:
            s = term[2].upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = np.zeros((len(self.test), self.len))
        self.train_matrix = np.zeros((len(self.train), self.len))

    def get_matrix(self):
        for i in range(len(self.train)):
            s = self.train[i][2].upper()
            words = s.split()
            for word in words:
                self.train_matrix[i][self.dict_words[word]] = 1
        for i in range(len(self.test)):
            s = self.test[i][2].upper()
            words = s.split()
            for word in words:
                self.test_matrix[i][self.dict_words[word]] = 1


# 特征提取类 - N-gram模型
class Gram:
    def __init__(self, my_data, dimension=2, max_item=1000, test_rate=0.3, random_state=2025):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()
        self.len = 0
        self.dimension = dimension
        self.train, self.test = train_test_split(
            self.data, test_size=test_rate, random_state=random_state
        )
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for d in range(1, self.dimension + 1):
            for term in self.data:
                s = term[2].upper()
                words = s.split()
                for i in range(len(words) - d + 1):
                    temp = words[i:i + d]
                    temp = "_".join(temp)
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = np.zeros((len(self.test), self.len))
        self.train_matrix = np.zeros((len(self.train), self.len))

    def get_matrix(self):
        for d in range(1, self.dimension + 1):
            for i in range(len(self.train)):
                s = self.train[i][2].upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.train_matrix[i][self.dict_words[temp]] = 1
            for i in range(len(self.test)):
                s = self.test[i][2].upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.test_matrix[i][self.dict_words[temp]] = 1


# Softmax回归模型
class Softmax:
    def __init__(self, sample, typenum, feature):
        self.sample = sample
        self.typenum = typenum
        self.feature = feature
        self.W = np.random.randn(feature, typenum)

    def softmax_calculation(self, x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

    def softmax_all(self, wtx):
        wtx -= np.max(wtx, axis=1, keepdims=True)
        wtx = np.exp(wtx)
        wtx /= np.sum(wtx, axis=1, keepdims=True)
        return wtx

    def change_y(self, y):
        ans = np.array([0] * self.typenum)
        ans[y] = 1
        return ans.reshape(-1, 1)

    def prediction(self, X):
        prob = self.softmax_all(X.dot(self.W))
        return prob.argmax(axis=1)

    def correct_rate(self, train, train_y, test, test_y):
        # 计算准确率
        n_train = len(train)
        pred_train = self.prediction(train)
        train_correct = sum([train_y[i] == pred_train[i] for i in range(n_train)]) / n_train
        
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        
        return train_correct, test_correct

    def loss_calculation(self, X, y):
        # 计算交叉熵损失
        y_pred = self.softmax_all(X.dot(self.W))
        y_onehot = np.zeros((len(y), self.typenum))
        for i in range(len(y)):
            y_onehot[i] = self.change_y(y[i]).flatten()
        
        epsilon = 1e-10
        cross_entropy = -np.sum(y_onehot * np.log(y_pred + epsilon), axis=1)
        return np.mean(cross_entropy)

    def regression(self, X, y, alpha, times, strategy="mini", mini_size=100):
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("样本数量不匹配!")
        
        if strategy == "mini":
            for _ in range(times):
                increment = np.zeros((self.feature, self.typenum))
                for _ in range(mini_size):
                    k = random.randint(0, self.sample - 1)
                    yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                    increment += X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)
                self.W += alpha / mini_size * increment
        
        elif strategy == "shuffle":
            for _ in range(times):
                k = random.randint(0, self.sample - 1)
                yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                increment = X[k].reshape(-1, 1).dot((self.change_y(y[k]) - yhat).T)
                self.W += alpha * increment
        
        elif strategy == "batch":
            for _ in range(times):
                increment = np.zeros((self.feature, self.typenum))
                for j in range(self.sample):
                    yhat = self.softmax_calculation(self.W.T.dot(X[j].reshape(-1, 1)))
                    increment += X[j].reshape(-1, 1).dot((self.change_y(y[j]) - yhat).T)
                self.W += alpha / self.sample * increment
        
        else:
            raise Exception("未知的优化策略")



# 计算指标
def calculate_metrics(bag, gram, total_steps, step_interval, alpha, mini_size):
    steps = np.arange(step_interval, total_steps + step_interval, step_interval)
    results = {
        'steps': steps,
        'bag': {
            'shuffle': {'acc': ([], []), 'loss': ([], [])},
            'batch': {'acc': ([], []), 'loss': ([], [])},
            'mini': {'acc': ([], []), 'loss': ([], [])}
        },
        'gram': {
            'shuffle': {'acc': ([], []), 'loss': ([], [])},
            'batch': {'acc': ([], []), 'loss': ([], [])},
            'mini': {'acc': ([], []), 'loss': ([], [])}
        }
    }

    # 词袋模型计算
    for strategy in ['shuffle', 'batch', 'mini']:
        model = Softmax(len(bag.train), 5, bag.len)
        for step in steps:
            # 每次训练间隔步数
            train_times = step_interval if step == step_interval else step_interval
            
            if strategy == 'shuffle':
                model.regression(bag.train_matrix, bag.train_y, alpha, train_times, "shuffle")
            elif strategy == 'batch':
                model.regression(bag.train_matrix, bag.train_y, alpha, train_times, "batch")
            else:  # mini
                model.regression(bag.train_matrix, bag.train_y, alpha, train_times, "mini", mini_size)
            
            # 记录准确率和损失
            train_acc, test_acc = model.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
            train_loss = model.loss_calculation(bag.train_matrix, bag.train_y)
            test_loss = model.loss_calculation(bag.test_matrix, bag.test_y)
            
            results['bag'][strategy]['acc'][0].append(train_acc)
            results['bag'][strategy]['acc'][1].append(test_acc)
            results['bag'][strategy]['loss'][0].append(train_loss)
            results['bag'][strategy]['loss'][1].append(test_loss)

    # N-gram模型计算
    for strategy in ['shuffle', 'batch', 'mini']:
        model = Softmax(len(gram.train), 5, gram.len)
        for step in steps:
            train_times = step_interval if step == step_interval else step_interval
            
            if strategy == 'shuffle':
                model.regression(gram.train_matrix, gram.train_y, alpha, train_times, "shuffle")
            elif strategy == 'batch':
                model.regression(gram.train_matrix, gram.train_y, alpha, train_times, "batch")
            else:  # mini
                model.regression(gram.train_matrix, gram.train_y, alpha, train_times, "mini", mini_size)
            
            # 记录准确率和损失
            train_acc, test_acc = model.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
            train_loss = model.loss_calculation(gram.train_matrix, gram.train_y)
            test_loss = model.loss_calculation(gram.test_matrix, gram.test_y)
            
            results['gram'][strategy]['acc'][0].append(train_acc)
            results['gram'][strategy]['acc'][1].append(test_acc)
            results['gram'][strategy]['loss'][0].append(train_loss)
            results['gram'][strategy]['loss'][1].append(test_loss)

    return results


# 绘制准确率可视化图表
def plot_accuracy(results):
    steps = results['steps']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("准确率随训练次数变化", fontsize=16, y=0.96)

    # 1. 词袋模型 - 训练集准确率
    axes[0, 0].plot(steps, results['bag']['shuffle']['acc'][0], 'r--', label='随机梯度')
    axes[0, 0].plot(steps, results['bag']['batch']['acc'][0], 'g--', label='批量梯度')
    axes[0, 0].plot(steps, results['bag']['mini']['acc'][0], 'b--', label='小批量梯度')
    axes[0, 0].plot(steps, results['bag']['shuffle']['acc'][0], 'ro-', markersize=4)
    axes[0, 0].plot(steps, results['bag']['batch']['acc'][0], 'g+-', markersize=4)
    axes[0, 0].plot(steps, results['bag']['mini']['acc'][0], 'b^-', markersize=4)
    axes[0, 0].set_title("词袋模型 - 训练集", fontsize=12)
    axes[0, 0].set_xlabel("训练次数", fontsize=10)
    axes[0, 0].set_ylabel("准确率", fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=9)

    # 2. 词袋模型 - 测试集准确率
    axes[0, 1].plot(steps, results['bag']['shuffle']['acc'][1], 'r--', label='随机梯度')
    axes[0, 1].plot(steps, results['bag']['batch']['acc'][1], 'g--', label='批量梯度')
    axes[0, 1].plot(steps, results['bag']['mini']['acc'][1], 'b--', label='小批量梯度')
    axes[0, 1].plot(steps, results['bag']['shuffle']['acc'][1], 'ro-', markersize=4)
    axes[0, 1].plot(steps, results['bag']['batch']['acc'][1], 'g+-', markersize=4)
    axes[0, 1].plot(steps, results['bag']['mini']['acc'][1], 'b^-', markersize=4)
    axes[0, 1].set_title("词袋模型 - 测试集", fontsize=12)
    axes[0, 1].set_xlabel("训练次数", fontsize=10)
    axes[0, 1].set_ylabel("准确率", fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=9)

    # 3. N-gram模型 - 训练集准确率
    axes[1, 0].plot(steps, results['gram']['shuffle']['acc'][0], 'r--', label='随机梯度')
    axes[1, 0].plot(steps, results['gram']['batch']['acc'][0], 'g--', label='批量梯度')
    axes[1, 0].plot(steps, results['gram']['mini']['acc'][0], 'b--', label='小批量梯度')
    axes[1, 0].plot(steps, results['gram']['shuffle']['acc'][0], 'ro-', markersize=4)
    axes[1, 0].plot(steps, results['gram']['batch']['acc'][0], 'g+-', markersize=4)
    axes[1, 0].plot(steps, results['gram']['mini']['acc'][0], 'b^-', markersize=4)
    axes[1, 0].set_title("N-gram模型 - 训练集", fontsize=12)
    axes[1, 0].set_xlabel("训练次数", fontsize=10)
    axes[1, 0].set_ylabel("准确率", fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=9)

    # 4. N-gram模型 - 测试集准确率
    axes[1, 1].plot(steps, results['gram']['shuffle']['acc'][1], 'r--', label='随机梯度')
    axes[1, 1].plot(steps, results['gram']['batch']['acc'][1], 'g--', label='批量梯度')
    axes[1, 1].plot(steps, results['gram']['mini']['acc'][1], 'b--', label='小批量梯度')
    axes[1, 1].plot(steps, results['gram']['shuffle']['acc'][1], 'ro-', markersize=4)
    axes[1, 1].plot(steps, results['gram']['batch']['acc'][1], 'g+-', markersize=4)
    axes[1, 1].plot(steps, results['gram']['mini']['acc'][1], 'b^-', markersize=4)
    axes[1, 1].set_title("N-gram模型 - 测试集", fontsize=12)
    axes[1, 1].set_xlabel("训练次数", fontsize=10)
    axes[1, 1].set_ylabel("准确率", fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(axis='both', labelsize=9)

    # 调整布局，避免文字重叠
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, hspace=0.3, wspace=0.25)
    plt.show()


# 绘制损失可视化图表
def plot_loss(results):
    steps = results['steps']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("损失值随训练次数变化", fontsize=16, y=0.96)

    # 1. 词袋模型 - 训练集损失
    axes[0, 0].plot(steps, results['bag']['shuffle']['loss'][0], 'r--', label='随机梯度')
    axes[0, 0].plot(steps, results['bag']['batch']['loss'][0], 'g--', label='批量梯度')
    axes[0, 0].plot(steps, results['bag']['mini']['loss'][0], 'b--', label='小批量梯度')
    axes[0, 0].plot(steps, results['bag']['shuffle']['loss'][0], 'ro-', markersize=4)
    axes[0, 0].plot(steps, results['bag']['batch']['loss'][0], 'g+-', markersize=4)
    axes[0, 0].plot(steps, results['bag']['mini']['loss'][0], 'b^-', markersize=4)
    axes[0, 0].set_title("词袋模型 - 训练集", fontsize=12)
    axes[0, 0].set_xlabel("训练次数", fontsize=10)
    axes[0, 0].set_ylabel("损失值", fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=9)

    # 2. 词袋模型 - 测试集损失
    axes[0, 1].plot(steps, results['bag']['shuffle']['loss'][1], 'r--', label='随机梯度')
    axes[0, 1].plot(steps, results['bag']['batch']['loss'][1], 'g--', label='批量梯度')
    axes[0, 1].plot(steps, results['bag']['mini']['loss'][1], 'b--', label='小批量梯度')
    axes[0, 1].plot(steps, results['bag']['shuffle']['loss'][1], 'ro-', markersize=4)
    axes[0, 1].plot(steps, results['bag']['batch']['loss'][1], 'g+-', markersize=4)
    axes[0, 1].plot(steps, results['bag']['mini']['loss'][1], 'b^-', markersize=4)
    axes[0, 1].set_title("词袋模型 - 测试集", fontsize=12)
    axes[0, 1].set_xlabel("训练次数", fontsize=10)
    axes[0, 1].set_ylabel("损失值", fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=9)

    # 3. N-gram模型 - 训练集损失
    axes[1, 0].plot(steps, results['gram']['shuffle']['loss'][0], 'r--', label='随机梯度')
    axes[1, 0].plot(steps, results['gram']['batch']['loss'][0], 'g--', label='批量梯度')
    axes[1, 0].plot(steps, results['gram']['mini']['loss'][0], 'b--', label='小批量梯度')
    axes[1, 0].plot(steps, results['gram']['shuffle']['loss'][0], 'ro-', markersize=4)
    axes[1, 0].plot(steps, results['gram']['batch']['loss'][0], 'g+-', markersize=4)
    axes[1, 0].plot(steps, results['gram']['mini']['loss'][0], 'b^-', markersize=4)
    axes[1, 0].set_title("N-gram模型 - 训练集", fontsize=12)
    axes[1, 0].set_xlabel("训练次数", fontsize=10)
    axes[1, 0].set_ylabel("损失值", fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=9)

    # 4. N-gram模型 - 测试集损失
    axes[1, 1].plot(steps, results['gram']['shuffle']['loss'][1], 'r--', label='随机梯度')
    axes[1, 1].plot(steps, results['gram']['batch']['loss'][1], 'g--', label='批量梯度')
    axes[1, 1].plot(steps, results['gram']['mini']['loss'][1], 'b--', label='小批量梯度')
    axes[1, 1].plot(steps, results['gram']['shuffle']['loss'][1], 'ro-', markersize=4)
    axes[1, 1].plot(steps, results['gram']['batch']['loss'][1], 'g+-', markersize=4)
    axes[1, 1].plot(steps, results['gram']['mini']['loss'][1], 'b^-', markersize=4)
    axes[1, 1].set_title("N-gram模型 - 测试集", fontsize=12)
    axes[1, 1].set_xlabel("训练次数", fontsize=10)
    axes[1, 1].set_ylabel("损失值", fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(axis='both', labelsize=9)

    # 调整布局，避免文字重叠
    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, hspace=0.3, wspace=0.25)
    plt.show()


# 学习率对准确率影响的可视化函数
def alpha_gradient_plot(bag, gram, total_times, mini_size):
    # 不同学习率设定，观察欠拟合/过拟合/梯度爆炸
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    # 词袋模型
    # 随机梯度下降
    shuffle_train = []
    shuffle_test = []
    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        # 修正regression参数，与原始方法匹配
        soft.regression(
            X=bag.train_matrix,
            y=bag.train_y,
            alpha=alpha,
            times=total_times,
            strategy="shuffle"
        )
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        shuffle_train.append(r_train)
        shuffle_test.append(r_test)
    
    # 批量梯度下降
    batch_train = []
    batch_test = []
    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        soft.regression(
            X=bag.train_matrix,
            y=bag.train_y,
            alpha=alpha,
            times=int(total_times / bag.max_item),  # 按总次数/样本量调整批量训练步数
            strategy="batch"
        )
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        batch_train.append(r_train)
        batch_test.append(r_test)
    
    # 小批量梯度下降
    mini_train = []
    mini_test = []
    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        soft.regression(
            X=bag.train_matrix,
            y=bag.train_y,
            alpha=alpha,
            times=int(total_times / mini_size),  # 按总次数/批次大小调整小批量训练步数
            strategy="mini",
            mini_size=mini_size
        )
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)
    
    # 绘制词袋模型子图
    plt.subplot(2, 2, 1)
    plt.semilogx(alphas, shuffle_train, 'r--', label='随机梯度')
    plt.semilogx(alphas, batch_train, 'g--', label='批量梯度')
    plt.semilogx(alphas, mini_train, 'b--', label='小批量梯度')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_train, 'b^-')
    plt.legend()
    plt.title("词袋模型 - 训练集")
    plt.xlabel("学习率")
    plt.ylabel("准确率")
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 2)
    plt.semilogx(alphas, shuffle_test, 'r--', label='随机梯度')
    plt.semilogx(alphas, batch_test, 'g--', label='批量梯度')
    plt.semilogx(alphas, mini_test, 'b--', label='小批量梯度')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_test, 'b^-')
    plt.legend()
    plt.title("词袋模型 - 测试集")
    plt.xlabel("学习率")
    plt.ylabel("准确率")
    plt.ylim(0, 1)
    
    # N-gram模型
    # 随机梯度下降
    shuffle_train = []
    shuffle_test = []
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, total_times, "shuffle")
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        shuffle_train.append(r_train)
        shuffle_test.append(r_test)
    
    # 批量梯度下降
    batch_train = []
    batch_test = []
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / gram.max_item), "batch")
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        batch_train.append(r_train)
        batch_test.append(r_test)
    
    # 小批量梯度下降
    mini_train = []
    mini_test = []
    for alpha in alphas:
        soft = Softmax(len(gram.train), 5, gram.len)
        soft.regression(gram.train_matrix, gram.train_y, alpha, int(total_times / mini_size), "mini", mini_size)
        r_train, r_test = soft.correct_rate(gram.train_matrix, gram.train_y, gram.test_matrix, gram.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)
    
    # 绘制N-gram模型子图
    plt.subplot(2, 2, 3)
    plt.semilogx(alphas, shuffle_train, 'r--', label='随机梯度')
    plt.semilogx(alphas, batch_train, 'g--', label='批量梯度')
    plt.semilogx(alphas, mini_train, 'b--', label='小批量梯度')
    plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_train, 'b^-')
    plt.legend()
    plt.title("N-gram模型 - 训练集")
    plt.xlabel("学习率")
    plt.ylabel("准确率")
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 4)
    plt.semilogx(alphas, shuffle_test, 'r--', label='随机梯度')
    plt.semilogx(alphas, batch_test, 'g--', label='批量梯度')
    plt.semilogx(alphas, mini_test, 'b--', label='小批量梯度')
    plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_test, 'b^-')
    plt.legend()
    plt.title("N-gram模型 - 测试集")
    plt.xlabel("学习率")
    plt.ylabel("准确率")
    plt.ylim(0, 1)
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


def main():
    max_item = 1000  # 最大数据量
    total_steps = 10000  # 总训练次数
    step_interval = 1000  # 记录间隔
    alpha = 1  # 固定学习率
    mini_size = 10  # mini-batch大小
    random_seed = 2025  

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 读取数据
    with open('train.tsv') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
    data = temp[1:]  # 跳过表头

    # 特征提取 - 词袋模型
    bag = Bag(data, max_item=max_item, random_state=random_seed)
    bag.get_words()
    bag.get_matrix()

    # 特征提取 - N-gram模型
    gram = Gram(data, dimension=2, max_item=max_item, random_state=random_seed)
    gram.get_words()
    gram.get_matrix()

    # 计算指标
    results = calculate_metrics(bag, gram, total_steps, step_interval, alpha, mini_size)
    
    # 绘制学习率对准确率的影响图表
    alpha_gradient_plot(bag, gram, total_steps, mini_size)
    # 可视化
    plot_accuracy(results)  # 准确率图表
    plot_loss(results)      # 损失图表
    


if __name__ == "__main__":
    main()


