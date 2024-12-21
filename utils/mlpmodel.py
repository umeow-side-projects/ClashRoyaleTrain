import pickle
import numpy as np

class MLPModel(object):
    alpha = 1.2
    def __init__(self, input_dim=28*28, output_dim=26, epochs=100, num_neurons=256, learning_rate=3e-7):
        self.num_neurons = num_neurons # 每層的節點數
        self.lambda_num = np.random.randint(1, 9) + np.random.random() # L2 Regularization 的變數 使模型在收斂期間不容易陷入 overfitting 但可能些許影響模型的正確率

        # 隨機生成 第 1~3 層的權重 並乘 sqrt(2 / 來源層節點數) 防止權重太大造成 overfitting
        self.layer_1_weight = np.random.randn(self.num_neurons, input_dim) * np.sqrt(2. / input_dim)
        self.layer_2_weight = np.random.randn(self.num_neurons, self.num_neurons) * np.sqrt(2. / num_neurons)
        self.layer_3_weight = np.random.randn(output_dim, self.num_neurons) * np.sqrt(2. / output_dim)

        # 隨機生成每個節點的偏差值
        self.layer_1_bias = np.zeros((self.num_neurons, 1))
        self.layer_2_bias = np.zeros((self.num_neurons, 1))
        self.layer_3_bias = np.zeros((output_dim, 1))
        
        # 神經網路輸出的誤差
        self.network_error = []
        
        # 第一層的預設資料
        self.layer_1_sum = [] # 上一層 forward 下來未經 activate function 的結果
        self.layer_1_output = [] # sum 經過 activate function 的結果
        self.layer_1_error = [] # 各節點的誤差值
        self.layer_1_weight_delta = [] # 權重修正值
        self.layer_1_bias_delta = [] # 偏差修正值
        
        # 第二層的預設資料
        self.layer_2_sum = []
        self.layer_2_output = []
        self.layer_2_error = []
        self.layer_2_weight_delta = []
        self.layer_2_bias_delta = []
        
        # 第三層的預設資料
        self.layer_3_sum = []
        self.layer_3_output = []
        self.layer_3_error = []
        self.layer_3_weight_delta = []
        self.layer_3_bias_delta = []
        
        # 正確率
        self.accuracy = 0
        
        # 正向傳遞的結果
        self.predictions = []

        # 學習率
        self.learning_rate = learning_rate
        
        # 訓練次屬
        self.epochs = epochs
        
    @staticmethod
    def sigmoid(x):
        """ S型函數 """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """ S型函數微分 """
        return MLPModel.sigmoid(x) * (1 - MLPModel.sigmoid(x))

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x > 0, 1.0, 0.0)

    @staticmethod
    def ELU(x):
        return np.where(x >= 0.0, x, MLPModel.alpha * (np.exp(x) - 1))

    @staticmethod
    def ELU_derivative(x):
        return np.where(x >= 0, 1, MLPModel.alpha * np.exp(x))

    def forward(self, x_list):
        """ 正向傳遞 """
        self.layer_1_sum = np.dot(self.layer_1_weight, x_list) + self.layer_1_bias # 通過矩陣乘法快速傳遞多個結果
        self.layer_1_output = MLPModel.ELU(self.layer_1_sum)

        self.layer_2_sum = np.dot(self.layer_2_weight, self.layer_1_output) + self.layer_2_bias
        self.layer_2_output = MLPModel.ReLU(self.layer_2_sum)
        
        self.layer_3_sum = np.dot(self.layer_3_weight, self.layer_2_output) + self.layer_3_bias
        self.layer_3_output = MLPModel.sigmoid(self.layer_3_sum)

        self.predictions = np.argmax(self.layer_3_output, axis=0)
        return self.predictions

    def backward(self, y_list, x_list):
        """ 反向傳遞 """
        one_hot_labels = np.eye(4)[y_list].T # 從 y_list 取得所有節點的理想值
        self.network_error = self.layer_3_output - one_hot_labels # 神經網路輸出誤差為 輸出層節點輸出-理想值
        self.layer_3_error = self.network_error * MLPModel.sigmoid_derivative(self.layer_3_output) # 神經網路誤差 * 輸出層結果微分 = 輸出層誤差
        self.layer_3_weight_delta = np.dot(self.layer_3_error, self.layer_2_output.T) # 輸出層誤差 * 上一層的輸出 = 誤差修正值
        self.layer_3_bias_delta = np.sum(self.layer_3_error, axis=1, keepdims=True) # 輸出層節點誤差總和 = 偏差修正值
        
        # 反向傳遞依此類推...
        
        self.layer_2_error = np.dot(self.layer_3_weight.T, self.layer_3_error) * MLPModel.ReLU_derivative(self.layer_2_output)
        self.layer_2_weight_delta = np.dot(self.layer_2_error, self.layer_1_output.T)
        self.layer_2_bias_delta = np.sum(self.layer_2_error, axis=1, keepdims=True)

        self.layer_1_error = np.dot(self.layer_2_weight.T, self.layer_2_error) * MLPModel.ELU_derivative(self.layer_1_output)
        self.layer_1_weight_delta = np.dot(self.layer_1_error, x_list.T)
        self.layer_1_bias_delta = np.sum(self.layer_1_error, axis=1, keepdims=True)

    def update_neurons(self):
        """ 更新節點權重 """
        self.layer_1_weight = self.layer_1_weight * (1 - self.learning_rate * self.lambda_num) - (self.learning_rate * self.layer_1_weight_delta)
        self.layer_2_weight = self.layer_2_weight * (1 - self.learning_rate * self.lambda_num) - (self.learning_rate * self.layer_2_weight_delta)
        self.layer_3_weight = self.layer_3_weight * (1 - self.learning_rate * self.lambda_num) - (self.learning_rate * self.layer_3_weight_delta)

        self.layer_1_bias -= self.learning_rate * self.layer_1_bias_delta
        self.layer_2_bias -= self.learning_rate * self.layer_2_bias_delta
        self.layer_3_bias -= self.learning_rate * self.layer_3_bias_delta

    def get_accuracy(self, y_list):
        """ 取得對於 train data 的正確率"""
        correct_predictions = np.sum(self.predictions == y_list)
        total_predictions = self.predictions.shape[0]
        self.accuracy = correct_predictions / total_predictions
        return self.accuracy

    def fit(self, x_list, y_list, x_test=None, y_test=None):
        """ 訓練函式 """
        for epoch in range(self.epochs):
            self.forward(x_list) # 正向傳遞
            self.backward(y_list, x_list) # 反向傳遞
            self.update_neurons() # 修正權重及偏差
            self.get_accuracy(y_list) # 取得這次的準確率
            print(f"Accuracy: {self.accuracy * 100}%, Epoch: {epoch + 1}/{self.epochs}")
            
            if x_test is not None and y_test is not None:
                self.predict(x_test)
                self.get_accuracy(y_test)
                print(f"Test: {self.accuracy * 100}%, Epoch {epoch + 1}/{self.epochs}")

    def predict(self, test_data):
        """ 單純預測結果 不做反向傳遞來訓練 """
        self.forward(test_data)
        predictions = self.predictions
        return predictions

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)