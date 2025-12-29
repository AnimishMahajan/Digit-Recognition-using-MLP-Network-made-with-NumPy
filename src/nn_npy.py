from tqdm import tqdm
import numpy as np

class network:

    def __init__(self):
        self.W = {}
        self.best_model = {}
        self.alpha = 0.0001
        self.f = ["none"]
        self.max_epch = 10
        self.patience = 5
        self.momentum = 0.8
        self.batch_size = 1
        # self.mse_array = []
        self.arch = []

    def createWeights(self, arch):
        for i in range(0, len(arch)-1):
        
            x = arch[i]

            if i == len(arch)-2:
                y = arch[i+1]
            else:
                y = arch[i+1] - 1
            
            limit = np.sqrt(6 / (x + y))
            self.W[i] = np.random.uniform(low=-limit, high=limit, size=(y, x)).astype(np.float32)

    def model(self, Inputs, Outputs, hidden_layers, activation_func, epochs, learning_rate, batch = 1):
        
        feature_shape = [Inputs.shape[1] + 1]
        output_shape = [Outputs.shape[1]]
        
        self.arch = feature_shape + hidden_layers + output_shape
        
        self.createWeights(self.arch)

        self.f += activation_func
        self.max_epch = epochs
        self.batch_size = batch
        self.alpha = learning_rate
    
    def summary(self):
        print(f"{'--' * 10}\nModel Summary\n")
        print(f"Architecture : {self.arch}")
        print(f"Activation Functions : {self.f}")
        print(f"Epochs : {self.max_epch}")
        print(f"Batch Size : {self.batch_size}")
        print(f"Learning Rate : {self.alpha}")
        print(f"Patience : {self.patience}")
        print(f"Momentum : {self.momentum}")

        print(f"\nParameters:")
        for i in range(len(self.W)):
            print(f"W{i} : {self.W[i].shape[0] * self.W[i].shape[1]} parameters")
        print(f"{'--' * 10}")
    
    def save_model(self, path):
        import os
        model_arch = {
            "arch" : self.arch,
            "f" : self.f,
            "max_epch" : self.max_epch,
            "batch_size" : self.batch_size,
            "alpha" : self.alpha,
            "patience" : self.patience,
            "momentum" : self.momentum
        }

        W_save = {}
        for i in range(len(self.W)):
            W_save[f"{i}"] = self.W[i]

        param_file = os.path.join(path, "params.npz")
        model_arch_file = os.path.join(path, "model_arch.npz")

        np.savez(param_file, **W_save)
        np.savez(model_arch_file, **model_arch)

    def load_model(self, path):
        import os

        param_file = os.path.join(path, "params.npz")
        model_arch_file = os.path.join(path, "model_arch.npz")

        W_save = np.load(param_file)
        model_arch = np.load(model_arch_file)

        for i in range(len(W_save)):
            self.W[i] = W_save[f"{i}"]

        self.arch = model_arch["arch"]
        self.f = model_arch["f"]
        self.max_epch = model_arch["max_epch"]
        self.batch_size = model_arch["batch_size"]
        self.alpha = model_arch["alpha"]
        self.patience = model_arch["patience"]
        self.momentum = model_arch["momentum"]

    def checkBias(self, X, W):
        if X.shape[0] == W.shape[1]:
            return X
            
        else:
            A = np.vstack((np.ones((1, X.shape[1])), X))
            return A

    def act(self, x, act = "linear", train = False):
        if act == "linear":
            output = x
            if train:
                derivative = np.ones_like(x)
        elif act == "sig":
            output = 1 / (1 + np.exp(-x))
            if train:
                derivative = output * (1 - output)
        elif act == "bisig":
            output = (1 - np.exp(-x)) / (1 + np.exp(-x))
            if train:
                derivative = 0.5 * (1 - (output ** 2))
        elif act == "relu":
            output = np.maximum(0, x)
            if train:
                derivative = (output > 0).astype(x.dtype)
        elif act == "softmax":
            c = np.max(x, axis = 0, keepdims = True)
            output = np.exp(x-c) / np.sum(np.exp(x-c), axis = 0, keepdims = True)
            derivative = None

        if train:
            return output, derivative
        return output
    
    def predict(self, x, train = False):

        A = {}
        der = {}
        
        A[0] = self.checkBias(x, self.W[0])

        for i in range(len(self.W)):
            
            Wi = self.W[i]
            Ai = A[i]

            Ai = self.checkBias(Ai, Wi)
            Zi = Wi @ Ai
            
            if train:
                A[i+1], der[i] = self.act(Zi, self.f[i+1], train = True)
            else:
                A[i+1] = self.act(Zi, self.f[i+1])

        if train:
            return A, der
        return A[len(A) - 1]

    def train(self, X, target, progress_bar = True):

        ep_patience = 0
        ep_mse = 9999
        break_condition = False
        
        A = {}
        der = {}
        dE = {}
        V = {}

        shape0 = X.shape[0]

        for ep in range(1, self.max_epch+1):

            count = 0
            epoch_loss_sum = 0

            for i in range(len(self.W)):
                V[i] = 0

            if progress_bar:
                prog = tqdm(zip(X, target), total=len(X), desc=f"Epoch {ep}", unit="sample")
            else:
                prog = zip(X, target)

            batch = 0
            it = 0
            x = None
            t = None

            for x_record, t_record in prog:
                it += 1
                
                x_record = x_record.reshape(-1, 1)
                t_record = t_record.reshape(-1, 1)

                #BATCHING
                if batch == 0:
                    x = x_record
                    t = t_record
                    batch += 1
                    continue

                if batch < self.batch_size:
                    x = np.hstack((x, x_record))
                    t = np.hstack((t, t_record))
                    batch += 1

                    if it == shape0:
                        None
                    else:
                        continue

                # FORWARD PASS
                A, der = self.predict(x, train = True)

                #LOSS CALCULATION
                if "softmax" in self.f:
                    loss = -np.sum(t * np.log(A[len(A) - 1] + 1e-9))
                else:
                    loss = np.mean((A[len(A) - 1] - t) ** 2)
                epoch_loss_sum += loss

                if progress_bar:
                    prog.set_postfix([("loss", f"{loss:.6f}")])
                # self.mse_array.append(loss)

                #BREAK CONDITIONS
                if np.isnan(loss):
                    break_condition = True
                    print("\n\nBreak Condition : NaN Loss\n\n")
                    break

                #BACKWARD PASS
                delta = None
                
                for i in range(len(self.W)-1, -1, -1):

                    der_act = der[i]
                    
                    if i == (len(self.W)-1): 
                        E = A[i+1] - t

                        if "softmax" in self.f:
                            delta = E
                        else:
                            delta = E * der_act

                    else:
                        d_Ai = self.W[i+1].T @ delta
                        d_Ai = d_Ai[1:, :]
                        delta = d_Ai * der_act

                    Ai = self.checkBias(A[i], self.W[i])
                    dE[i] = (delta @ Ai.T) / batch


                #MOMENTUM BASED GD
                for i in range(len(self.W)):
                    V[i] = (self.momentum * V[i]) + dE[i]
                
                #UPDATE WEIGHTS
                for i in range(len(self.W)):
                    self.W[i] = self.W[i] - (self.alpha * V[i])
                
                #RESET BATCHES
                batch = 0
                x = None
                t = None

            avg_epoch_loss = epoch_loss_sum / len(X)

            if count > 5: 
                if avg_epoch_loss > ep_mse:
                    ep_patience += 1
                else:
                    self.best_model = self.W
                    ep_mse = avg_epoch_loss
                    ep_patience = 0

                if ep_patience >= self.patience:
                    self.W = self.best_model
                    print("--- Early Stopping ---")
                    break
            else:
                count += 1

            if "softmax" in self.f:
                print(f"Epoch {ep} : Accuracy = {self.test_classification(X, target)}%")
            else:
                result = self.test_regression(X, target)
                print(f"Epoch {ep} : Avg MSE : {result[0]} | Avg MAE : {result[1]}")

    def test_classification(self, X_test, Y_test):
        A = {}

        correct = 0
        total = 0

        for x, t in zip(X_test, Y_test):
        
            x = x.reshape(-1, 1)
            t = t.reshape(-1, 1)

            # FORWARD PASS
            A = self.predict(x)

            if np.argmax(A, axis = 0) == np.argmax(t, axis = 0):
                correct += 1
            total += 1

        return (correct/total) * 100
    
    def test_regression(self, X_test, Y_test):

        total_mse = 0
        total_mae = 0

        for x, t in zip(X_test, Y_test):
        
            x = x.reshape(-1, 1)
            t = t.reshape(-1, 1)

            # FORWARD PASS
            A = self.predict(x)

            total_mse += 0.5 * np.mean((A - t) ** 2)
            total_mae += np.mean(np.abs(A - t))

        return [total_mse/X_test.shape[0], total_mae/X_test.shape[0]]