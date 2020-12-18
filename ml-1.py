import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 定义 Sequential类型的神经网络
model = Sequential()
# 定义两层的神经网络，units神经元个数，越多训练时间越长
model.add(Dense(units=8, activation='relu', input_dim=1))
model.add(Dense(units=1, activation='sigmoid'))
# MSE 作为损失函数， SGD 梯度下降作为优化方式
model.compile(loss='mean_squared_error', optimizer='sgd')

x = [1, 2, 3, 10, 20, -2, -10, -100, -5, -20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 训练10次 随机抽取4组数据
model.fit(x, y, epochs=10, batch_size=4)

test_x = [30, 40, -20, -60]
test_y = model.predict(test_x)

for i in range(0, len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], test_y[i]))

