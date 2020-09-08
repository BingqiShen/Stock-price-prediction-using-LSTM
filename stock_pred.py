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
from sklearn.metrics import mean_squared_error

def Read_data(filename):
    # 读取excel文件数据
    df = pd.read_excel(filename)
    df.head()

    # df.index里保存日期
    df["Date"] = pd.to_datetime(df.Date,format="%Y/%m/%d")
    df.index = df['Date']

    # plt.figure(figsize = (16,8))
    # plt.plot(df["High"],label='High Price history')
    # plt.plot(df["Low"],label='Low Price history')
    # plt.show()
    return df

########################################################################################################################
# 定义数据预处理函数

# 最高价数据处理
def High_data_preprocess(df,reference_num,train_test_ratio):
    # 按照时间顺序对单日最高价格数据进行排序
    data = df.sort_index(ascending=True,axis=0)
    high_dataset = pd.DataFrame(index=range(0,len(df)),columns=['Date','High'])

    train_num = math.ceil(len(data) * train_test_ratio)
    for i in range(0,len(data)):
        high_dataset["Date"][i] = data['Date'][i]
        high_dataset["High"][i] = data["High"][i]

    high_dataset.index = high_dataset.Date
    high_dataset.drop("Date",axis=1,inplace=True)

    # 对数据进行归一化处理
    final_dataset = high_dataset.values
    train_data = final_dataset[0:train_num,:]
    valid_data = final_dataset[train_num:,:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data,y_train_data = [],[]

    for i in range(reference_num,len(train_data)):
        x_train_data.append(scaled_data[i-reference_num:i,0])
        y_train_data.append(scaled_data[i,0])

    x_train_data,y_train_data = np.array(x_train_data),np.array(y_train_data)
    x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    # 利用随机种子打乱顺序
    np.random.seed(116)
    np.random.shuffle(x_train_data)
    np.random.seed(116)
    np.random.shuffle(y_train_data)


    return scaler,high_dataset, train_data, valid_data, x_train_data, y_train_data, train_num

# 最低价数据处理
def Low_data_preprocess(df,reference_num,train_test_ratio):
    # 按照时间顺序对单日最高价格数据进行排序
    data = df.sort_index(ascending=True,axis=0)
    low_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Low'])

    train_num = math.ceil(len(data) * train_test_ratio)
    for i in range(0,len(data)):
        low_dataset["Date"][i] = data['Date'][i]
        low_dataset["Low"][i] = data["Low"][i]

    low_dataset.index = low_dataset.Date
    low_dataset.drop("Date",axis=1,inplace=True)

    # 对数据进行归一化处理
    final_dataset = low_dataset.values
    train_data = final_dataset[0:train_num,:]
    valid_data = final_dataset[train_num:,:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data,y_train_data = [],[]

    for i in range(reference_num,len(train_data)):
        x_train_data.append(scaled_data[i-reference_num:i,0])
        y_train_data.append(scaled_data[i,0])

    x_train_data,y_train_data = np.array(x_train_data),np.array(y_train_data)
    x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    # 利用随机种子打乱顺序
    np.random.seed(116)
    np.random.shuffle(x_train_data)
    np.random.seed(116)
    np.random.shuffle(y_train_data)


    return scaler,low_dataset, train_data, valid_data, x_train_data, y_train_data, train_num

########################################################################################################################
# 定义预测函数
def predict(df_nse,X_test,new_data,train_num,scaler,price_mode,lstm_model,reference_num):
    # 定义预测天数
    global prediction_num
    prediction_num = 5

    # 只是为了产生train_data和valid_data，方便调用
    prediction_price = lstm_model.predict(X_test)
    prediction_price = scaler.inverse_transform(prediction_price)
    train_data = new_data[:train_num]
    valid_data = new_data[train_num:]
    valid_data['Predictions'] = prediction_price

    x = X_test[len(new_data) - train_num - 1, :, :]
    x = np.reshape(x, (1, reference_num, 1))

    for i in range(1, prediction_num):
        prediction_price = lstm_model.predict(x)
        x = np.reshape(x, (1, reference_num))
        x = np.column_stack((x, prediction_price))
        x = np.reshape(x, (1, reference_num + 1, 1))
        x = x[:, 1:reference_num + 1, :]

    prediction_price = np.reshape(x, (reference_num, 1))
    prediction_price = prediction_price[
                            len(prediction_price) - prediction_num:len(prediction_price), :]
    prediction_price = scaler.inverse_transform(prediction_price)

    test = new_data
    df = df_nse
    if price_mode == 1:
        for i in range(1, prediction_num + 1):
            test = test.append([{'High': float(prediction_price[i - 1, :])}], ignore_index=True)
            df = df.append([{'Date': pd.Timestamp(2020, 7, 22 + i)}], ignore_index=True)

        test = test[len(new_data):]



        plt.plot(df_nse['Date'][0:train_num], train_data["High"], color='blue',label = 'train data')
        plt.plot(df_nse['Date'][train_num:], valid_data['High'],color = 'green',label = 'valid data for real')
        plt.plot(df_nse['Date'][train_num:], valid_data["Predictions"],color = 'yellow',label = 'valid data for prediction')
        plt.plot(df['Date'][len(new_data):], test["High"], color='red',label = 'prediction')
        plt.title('Prediction for High price', fontsize='large', fontweight='bold')
        plt.legend()
        plt.show()
    elif price_mode == 0:
        for i in range(1, prediction_num + 1):
            test = test.append([{'Low': float(prediction_price[i - 1, :])}], ignore_index=True)
            df = df.append([{'Date': pd.Timestamp(2020, 7, 22 + i)}], ignore_index=True)

        test = test[len(new_data):]

        plt.plot(df_nse['Date'][0:train_num], train_data["Low"], color='blue', label='train data')
        plt.plot(df_nse['Date'][train_num:], valid_data['Low'], color='green', label='valid data for real')
        plt.plot(df_nse['Date'][train_num:], valid_data["Predictions"], color='yellow',
                 label='valid data for prediction')
        plt.plot(df['Date'][len(new_data):], test["Low"], color='red', label='prediction')
        plt.title('Prediction for Low price', fontsize='large', fontweight='bold')
        plt.legend()
        plt.show()

    return prediction_price,df,test

def get_date(df):
    print('请输入指定年份：')
    year = int(input())
    print('请输入指定月份：')
    month = int(input())
    print('请输入指定日期：')
    date = int(input())

    for i in range(0,len(df)):
        if df['Date'][i] == pd.Timestamp(year, month, date):
            break

    if i == len(df) and pd.Timestamp(year, month, date) != pd.Timestamp(2020, 7, 22):
        i == 0

    return i


if __name__ == '__main__':
    # 训练测试比
    global train_test_ration
    train_test_ratio = 0.7
    # 导入文件
    filename = "DatasetForLSTM/JTL.xlsx"
    # 以当前日期的前90天作为训练数据
    global reference_num
    reference_num = 90

    # 定义训练模式，train_mode = 1时训练模型，train_mode = 0时导入模型进行预测
    train_mode = 0
    # 定义价格模式，price_mode = 1时导入最高价，price_mode = 0时导入最低价
    price_mode = 1

    df = Read_data(filename)
    if price_mode == 1:
        scaler,new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = High_data_preprocess(df,reference_num,train_test_ratio)
    elif price_mode == 0:
        scaler,new_dataset, train_data, valid_data, x_train_data, y_train_data, train_num = Low_data_preprocess(df,reference_num,train_test_ratio)



    if train_mode == 1:
        inputs_data = new_dataset[len(new_dataset) - len(valid_data) - reference_num:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        # 构造stacked LSTM模型
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        #         lstm_model.add(Dropout(0.2))
        #         lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
        #         lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        #         lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=50))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1))

        lstm_model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
        history = lstm_model.fit(x_train_data, y_train_data, epochs=50, batch_size=4, verbose=2, validation_split=0.3,
                                 shuffle=True)

        epochs = range(len(history.history['loss']))

        plt.figure()
        plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
        plt.title('Traing and Validation loss')
        plt.legend()
        plt.savefig('model_V1.1_loss.jpg')

        # 保存LSTM模型
        lstm_model.save("{}_saved_lstm_model.h5".format(filename))

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

        # try:
        #     plt.plot(train_data["High"])
        #     plt.plot(valid_data[['High',"Predictions"]])
        #     plt.title('High price history',fontsize='large',fontweight='bold')
        #     plt.show()
        # except:
        #     plt.plot(train_data["Low"])
        #     plt.plot(valid_data[['Low', "Predictions"]])
        #     plt.title('Low price history', fontsize='large', fontweight='bold')
        #     plt.show()

        try:
            mse= mean_squared_error(valid_data['Predictions'] ,valid_data['Low'])   #计算测试的mse
        except:
            mse= mean_squared_error(valid_data['Predictions'] ,valid_data['High'])  #计算测试的ms
        print(mse)

    elif train_mode ==0:
        # 加载LSTM模型
        lstm_model = load_model("lstm_model.h5")

        train_num = get_date(df)
        while train_num < reference_num:
            print('error！请重新输入日期')
            train_num = get_date(df)

        inputs_data = new_dataset[train_num - reference_num:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        # 取样本对之进行预测
        X_test = []
        for i in range(reference_num,inputs_data.shape[0]):
            X_test.append(inputs_data[i-reference_num:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

        prediction_price,df,test = predict(df,X_test,new_dataset,train_num,scaler,price_mode,lstm_model,reference_num)
        print(prediction_price)
