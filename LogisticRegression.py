import numpy as np
import time
import matplotlib.pyplot as plt 

class LogisticRegression:
    
    def __init__(self, k, n, method, alpha = 0.001, max_iter=5000, use_penalty=False, lambda_=0.1):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.use_penalty = use_penalty  # Ridge Logistic
        self.lambda_ = lambda_
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
    #Add Ridge Penalty    
    def gradient(self, X, Y):
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y * np.log(h))  # Cross-Entropy Loss
        if self.use_penalty:    #lambda * sum(theta^2)
            loss += self.lambda_ * np.sum(np.square(self.W))
        
        error = h - Y
        grad = self.softmax_grad(X, error)

        if self.use_penalty:    #lambda * theta
            grad += self.lambda_ * self.W
    
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
    
    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)
    
    def precision(self, y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FP = np.sum((y_true != class_label) & (y_pred == class_label))
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    def recall(self, y_true, y_pred, class_label):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FN = np.sum((y_true == class_label) & (y_pred != class_label))
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def f1_score(self, precision_value, recall_value):
        return 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0

    def macro_precision(self, y_true, y_pred, num_classes):
        precision_scores = [self.precision(y_true, y_pred, c) for c in range(num_classes)]
        return np.mean(precision_scores)

    def weighted_precision(self, y_true, y_pred, num_classes):
        weights = []
        precision_scores = []
        total_samples = len(y_true)
        
        for c in range(num_classes):
            class_count = np.sum(y_true == c)  # จำนวนตัวอย่างของคลาสนั้น
            weight = class_count / total_samples  # น้ำหนักของคลาสนั้น
            weights.append(weight)
            precision_scores.append(self.precision(y_true, y_pred, c))
        
        weighted_prec = sum([w * p for w, p in zip(weights, precision_scores)])
        return weighted_prec
    
    def weighted_recall(self, y_true, y_pred, num_classes):
        weights = []
        recall_scores = []
        total_samples = len(y_true)
        
        for c in range(num_classes):
            class_count = np.sum(y_true == c)  # จำนวนตัวอย่างของคลาสนั้น
            weight = class_count / total_samples  # น้ำหนักของคลาสนั้น
            weights.append(weight)
            recall_scores.append(self.recall(y_true, y_pred, c))
        
        weighted_rec = sum([w * r for w, r in zip(weights, recall_scores)])
        return weighted_rec

    def weighted_f1(self, y_true, y_pred, num_classes):
        prec = self.weighted_precision(y_true, y_pred, num_classes)
        rec = self.weighted_recall(y_true, y_pred, num_classes)
        return self.f1_score(prec, rec)