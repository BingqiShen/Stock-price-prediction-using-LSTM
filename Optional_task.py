import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
# 加上这两句，不然下边plt要出现warning
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.models import load_model
import math
import stock_pred
import os
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # 训练测试比
    global train_test_ration
    train_test_ratio = 0.7
    # 导入文件
    file_path = "DatasetForLSTM"
    file_lists = os.listdir(file_path)
    # 以当前日期的前90天作为训练数据
    global reference_num
    reference_num = 90

    # 定义训练模式，train_mode = 1时训练模型，train_mode = 0时导入模型进行预测
    train_mode = 0
    # 定义价格模式，price_mode = 1时导入最高价，price_mode = 0时导入最低价
    price_mode = 1

    if train_mode == 1:
        lstm_model = load_model("Model/BFXT.xlsx_saved_lstm_model.h5")

        for filename in file_lists:
            print(filename)

            df = stock_pred.Read_data(file_path + '/' + filename)
            if price_mode == 1:
                scaler,new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = stock_pred.High_data_preprocess(df,reference_num,train_test_ratio)
            elif price_mode == 0:
                scaler,new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = stock_pred.Low_data_preprocess(df,reference_num,train_test_ratio)

            if train_mode == 1:
                inputs_data = new_dataset[len(new_dataset) - len(valid_data) - reference_num:].values
                inputs_data = inputs_data.reshape(-1, 1)
                inputs_data = scaler.transform(inputs_data)

                # lstm_model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
                history = lstm_model.fit(x_train_data, y_train_data, epochs=50, batch_size=4, verbose=2, validation_split=0.3,
                                         shuffle=True)

                epochs = range(len(history.history['loss']))

                plt.figure()
                plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
                plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
                plt.title('Traing and Validation loss')
                plt.legend()
                plt.savefig('model_V1.1_loss.jpg')

                # 取样本对之进行预测
                X_test = []
                for i in range(reference_num, inputs_data.shape[0]):
                    X_test.append(inputs_data[i - reference_num:i, 0])
                X_test = np.array(X_test)

                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                prediction_price = lstm_model.predict(X_test)
                prediction_price = scaler.inverse_transform(prediction_price)

                # 可视化
                train_data = new_dataset[:train_num]
                valid_data = new_dataset[train_num:]
                valid_data['Predictions'] = prediction_price


                try:
                    mse = mean_squared_error(valid_data['Predictions'], valid_data['Low'])  # 计算测试的mse
                except:
                    mse = mean_squared_error(valid_data['Predictions'], valid_data['High'])  # 计算测试的ms
                print(mse)

                lstm_model.save("lstm_model.h5")

                print('{}数据已被训练'.format(filename))

        lstm_model.save("lstm_model.h5")
    else:
        lstm_model = load_model("lstm_model.h5")
        filename = 'GFKJ.xlsx'

        df = stock_pred.Read_data(filename)
        if price_mode == 1:
            scaler, new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = stock_pred.High_data_preprocess(
                df, reference_num, train_test_ratio)
        elif price_mode == 0:
            scaler, new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = stock_pred.Low_data_preprocess(df,
                                                                                                                     reference_num,
                                                                                                                     train_test_ratio)
        train_num = stock_pred.get_date(df)
        while train_num < reference_num:
            print('error！请重新输入日期')
            train_num = stock_pred.get_date(df)

        inputs_data = new_dataset[train_num - reference_num:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        # 取样本对之进行预测
        X_test = []
        for i in range(reference_num, inputs_data.shape[0]):
            X_test.append(inputs_data[i - reference_num:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        prediction_price, df, test = stock_pred.predict(df, X_test, new_dataset, train_num, scaler, price_mode, lstm_model,
                                             reference_num)
        print(prediction_price)

        # try:
        #     mse= mean_squared_error(valid_data['Predictions'] ,valid_data['Low'])   #计算测试的mse
        # except:
        #     mse= mean_squared_error(valid_data['Predictions'] ,valid_data['High'])  #计算测试的ms
        # print(mse)
