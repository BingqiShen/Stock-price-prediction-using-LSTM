#-*- coding : utf-8 -*-
# coding: unicode_escape

import dash
import dash_core_components as dcc   #交互式组件，用于绘图
import dash_html_components as html  #代码转html，与网页相关，如用它实现Title显示及一些与用户的交互操作
from dash.dependencies import Input, Output #回调
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import stock_pred


# 训练测试比
global train_test_ratio
train_test_ratio = 0.7
# 以当前日期的前90天作为训练数据
global reference_num
reference_num = 90
# 导入文件
filename1 = "DatasetForLSTM/GZMT.xlsx"
filename2 = "DatasetForLSTM/JNFD.xlsx"
filename3 = "DatasetForLSTM/JTL.xlsx"
filename4 = "DatasetForLSTM/LSCZ.xlsx"
filename5 = "DatasetForLSTM/SNYG.xlsx"
filename6 = "DatasetForLSTM/BFXT.xlsx"
filename7 = "DatasetForLSTM/BGGF.xlsx"
filename8 = "DatasetForLSTM/GSYH.xlsx"
filename9 = "DatasetForLSTM/ZGLT.xlsx"
filename10 = "DatasetForLSTM/ZGPA.xlsx"
filename11 = "DatasetForLSTM/HRYY.xlsx"
filename12 = "DatasetForLSTM/JSYH.xlsx"
filename13 = "DatasetForLSTM/WLY.xlsx"
filename14 = "DatasetForLSTM/ZGSH.xlsx"
filename15 = "DatasetForLSTM/ZGSY.xlsx"

# 加载LSTM模型
model1 = load_model("Model/GZMT.xlsx_saved_lstm_model.h5")
model2 = load_model("Model/JNFD.xlsx_saved_lstm_model.h5")
model3 = load_model("Model/JTL.xlsx_saved_lstm_model.h5")
model4 = load_model("Model/LSCZ.xlsx_saved_lstm_model.h5")
model5 = load_model("Model/SNYG.xlsx_saved_lstm_model.h5")
model6 = load_model("Model/BFXT.xlsx_saved_lstm_model.h5")
model7 = load_model("Model/BGGF.xlsx_saved_lstm_model.h5")
model8 = load_model("Model/GSYH.xlsx_saved_lstm_model.h5")
model9 = load_model("Model/ZGLT.xlsx_saved_lstm_model.h5")
model10 = load_model("Model/ZGPA.xlsx_saved_lstm_model.h5")

model = model1

def get_date(df,year,month,date):
    year = int(year)
    month = int(month)
    date = int(date)
    for i in range(0,len(df)):
        if df['Date'][i] == pd.Timestamp(year, month, date):
            break
    if i == len(df) and pd.Timestamp(year, month, date) != pd.Timestamp(2020, 7, 22):
        i == 0
    return i


def sbq(filename,model,price_mode,year,month,date):
    # 数据预处理
    df_before = stock_pred.Read_data(filename)
    if price_mode == 1:
        scaler,new_data, train, valid, x_train_data, y_train_data, train_num = stock_pred.High_data_preprocess(df_before,reference_num,train_test_ratio)
    else:
        scaler, new_data, train, valid, x_train_data, y_train_data, train_num = stock_pred.Low_data_preprocess(
            df_before, reference_num, train_test_ratio)

    train_num = get_date(df_before,year,month,date)

    inputs = new_data[train_num-reference_num:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    # 取样本对之进行预测
    X_test = []
    for i in range(reference_num,inputs.shape[0]):
        X_test.append(inputs[i-reference_num:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    prediction_price = model.predict(X_test)
    prediction_price = scaler.inverse_transform(prediction_price)

    train = new_data[:train_num]
    valid = new_data[train_num:]
    valid['Predictions'] = prediction_price


    prediction_price,df_after,test = stock_pred.predict(df_before,X_test,new_data,train_num,scaler,price_mode,model,reference_num)

    return valid,test,df_after,new_data

valid,test,df_after,new_data = sbq(filename1,model,price_mode=1,year=2018,month=8,date=29)


app = dash.Dash()
server = app.server

# 对网页页面进行设置
app.layout = html.Div([
    # 定义网页标题
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    # 创建选项卡
    dcc.Tabs(id="tabs", children=[
        # 第一个选项名为'Actual, Valid and Prediction Stock High Data'
        dcc.Tab(label='Actual, Valid and Prediction Stock Data',children=[
            html.Div([
                dcc.Input(id='Year', value='2018', type='text'),
                dcc.Input(id='Month', value='8', type='text'),
                dcc.Input(id='Date', value='29', type='text'),

                # 实际数据部分
                html.H2("Actual and Valid price",style={"textAlign": "center"}),
                # 创建下拉列表1，并创建列表选项，此列表统计股价
                dcc.Dropdown(id='dropdown1',
                             options=[{'label': 'GZMT', 'value': 'GZMT'},
                                      {'label': 'JNFD', 'value': 'JNFD'},
                                      {'label': 'JTL', 'value': 'JTL'},
                                      {'label': 'LSCZ', 'value': 'LSCZ'},
                                      {'label': 'BFXT', 'value': 'BFXT'},
                                      {'label': 'BGGF', 'value': 'BGGF'},
                                      {'label': 'GSYH', 'value': 'GSYH'},
                                      {'label': 'ZGLT', 'value': 'ZGLT'},
                                      {'label': 'ZGPA', 'value': 'ZGPA'},
                                      {'label': 'HRYY', 'value': 'HRYY'},
                                      {'label': 'JSYH', 'value': 'JSYH'},
                                      {'label': 'ZGSY', 'value': 'ZGSY'},
                                      {'label': 'WLY', 'value': 'WLY'},
                                      {'label': 'ZGSH', 'value': 'ZGSH'},
                                      {'label': 'SNYG', 'value': 'SNYG'}],
                             # 默认选贵州茅台
                             multi=True, value=['GZMT'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 创建下拉列表highlow-1，并创建列表选项，此列表选择最高价或最低价
                dcc.Dropdown(id='dropdown-highlow-1',
                             options=[{'label': 'High', 'value': 'High'},
                                      {'label': 'Low', 'value': 'Low'}],
                             # 默认选High
                             multi=True,value=['High'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 创建下拉列表，并创建列表选项，此列表用于选择模型
                dcc.Dropdown(id='dropdown-model-1',
                             options=[{'label': 'model_1', 'value': 'model_1'},
                                      {'label': 'model_2', 'value': 'model_2'},
                                      {'label': 'model_3', 'value': 'model_3'},
                                      {'label': 'model_4', 'value': 'model_4'},
                                      {'label': 'model_5', 'value': 'model_5'},
                                      {'label': 'model_6', 'value': 'model_6'},
                                      {'label': 'model_7', 'value': 'model_7'},
                                      {'label': 'model_8', 'value': 'model_8'},
                                      {'label': 'model_9', 'value': 'model_9'},
                                      {'label': 'model_10', 'value': 'model_10'}],
                             # 默认选model_1
                             multi=True,value=['model_1'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 绘图1，图1名称为'Actual Data'，数据选取实际验证数据valid中的"High"数组
                dcc.Graph(
                    id="Actual and Valid Data",
                        ),


                # 预测数据部分
                html.H2("LSTM predicted price",style={"textAlign": "center"}),
                # 创建下拉列表3，并创建列表选项，此列表统计股价
                dcc.Dropdown(id='dropdown3',
                             options=[{'label': 'GZMT', 'value': 'GZMT'},
                                      {'label': 'JNFD', 'value': 'JNFD'},
                                      {'label': 'JTL', 'value': 'JTL'},
                                      {'label': 'LSCZ', 'value': 'LSCZ'},
                                      {'label': 'BFXT', 'value': 'BFXT'},
                                      {'label': 'BGGF', 'value': 'BGGF'},
                                      {'label': 'GSYH', 'value': 'GSYH'},
                                      {'label': 'ZGLT', 'value': 'ZGLT'},
                                      {'label': 'ZGPA', 'value': 'ZGPA'},
                                      {'label': 'HRYY', 'value': 'HRYY'},
                                      {'label': 'JSYH', 'value': 'JSYH'},
                                      {'label': 'ZGSY', 'value': 'ZGSY'},
                                      {'label': 'WLY', 'value': 'WLY'},
                                      {'label': 'ZGSH', 'value': 'ZGSH'},
                                      {'label': 'SNYG', 'value': 'SNYG'}],
                             # 默认选贵州茅台
                             multi=True, value=['GZMT'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 创建下拉列表highlow-1，并创建列表选项，此列表选择最高价或最低价
                dcc.Dropdown(id='dropdown-highlow-3',
                             options=[{'label': 'High', 'value': 'High'},
                                      {'label': 'Low', 'value': 'Low'}],
                             # 默认选High
                             multi=True,value=['High'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 创建下拉列表，并创建列表选项，此列表用于选择模型
                dcc.Dropdown(id='dropdown-model-2',
                             options=[{'label': 'model_1', 'value': 'model_1'},
                                      {'label': 'model_2', 'value': 'model_2'},
                                      {'label': 'model_3', 'value': 'model_3'},
                                      {'label': 'model_4', 'value': 'model_4'},
                                      {'label': 'model_5', 'value': 'model_5'},
                                      {'label': 'model_6', 'value': 'model_6'},
                                      {'label': 'model_7', 'value': 'model_7'},
                                      {'label': 'model_8', 'value': 'model_8'},
                                      {'label': 'model_9', 'value': 'model_9'},
                                      {'label': 'model_10', 'value': 'model_10'}],
                             # 默认选model_1
                             multi=True,value=['model_1'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                # 绘图3，图3名称为'Predicted Data'，数据选取实际验证数据valid中的"High"数组
                dcc.Graph(
                    id="Predicted Data",
                        )
            ])
        ])
    ])
])

def get_model(dropdown_model):
    if dropdown_model == ['model_1']:
        return model1
    elif dropdown_model == ['model_2']:
        return model2
    elif dropdown_model == ['model_3']:
        return model3
    elif dropdown_model == ['model_4']:
        return model4
    elif dropdown_model == ['model_5']:
        return model5
    elif dropdown_model == ['model_6']:
        return model6
    elif dropdown_model == ['model_7']:
        return model7
    elif dropdown_model == ['model_8']:
        return model8
    elif dropdown_model == ['model_9']:
        return model9
    elif dropdown_model == ['model_10']:
        return model10
    else:
        return model1

def get_data(selected_dropdown,year,month,date,model,price_mode):
    if selected_dropdown == ['GZMT']:
        valid, test, df_after, new_data = sbq(filename1,model,price_mode,year,month,date)
    elif selected_dropdown == ['JNFD']:
        valid, test, df_after, new_data = sbq(filename2,model,price_mode,year,month,date)
    elif selected_dropdown == ['JTL']:
        valid, test, df_after, new_data = sbq(filename3,model,price_mode,year,month,date)
    elif selected_dropdown == ['LSCZ']:
        valid, test, df_after, new_data = sbq(filename4,model,price_mode,year,month,date)
    elif selected_dropdown == ['SNYG']:
        valid, test, df_after, new_data = sbq(filename5,model,price_mode,year,month,date)
    elif selected_dropdown == ['BFXT']:
        valid, test, df_after, new_data = sbq(filename6,model,price_mode,year,month,date)
    elif selected_dropdown == ['BGGF']:
        valid, test, df_after, new_data = sbq(filename7, model, price_mode,year,month,date)
    elif selected_dropdown == ['GSYH']:
        valid, test, df_after, new_data = sbq(filename8,model,price_mode,year,month,date)
    elif selected_dropdown == ['ZGLT']:
        valid, test, df_after, new_data = sbq(filename9,model,price_mode,year,month,date)
    elif selected_dropdown == ['ZGPA']:
        valid, test, df_after, new_data = sbq(filename10,model,price_mode,year,month,date)
    elif selected_dropdown == ['HRYY']:
        valid, test, df_after, new_data = sbq(filename11,model,price_mode,year,month,date)
    elif selected_dropdown == ['JSYH']:
        valid, test, df_after, new_data = sbq(filename12, model, price_mode,year,month,date)
    elif selected_dropdown == ['WLY']:
        valid, test, df_after, new_data = sbq(filename13,model,price_mode,year,month,date)
    elif selected_dropdown == ['ZGSH']:
        valid, test, df_after, new_data = sbq(filename14,model,price_mode,year,month,date)
    elif selected_dropdown == ['ZGSY']:
        valid, test, df_after, new_data = sbq(filename15,model,price_mode,year,month,date)
    else:
        valid, test, df_after, new_data = sbq(filename1, model,price_mode,year,month,date)
    return valid, test, df_after, new_data


# 定义回调函数，使用‘@app.callback()'参数装饰器来装饰该回调函数，输出绑定图id，输入绑定滑块值
@app.callback(Output("Actual and Valid Data", 'figure'),
              [Input('dropdown1', 'value'),Input('dropdown-highlow-1', 'value'),Input('dropdown-model-1', 'value'),
               Input('Year', 'value'),Input('Month', 'value'),Input('Date', 'value')])

# 更新图1
def update_graph(selected_dropdown,dropdown_highlow,dropdown_model,year,month,date):
    dropdown = {"GZMT": "GZMT", "JNFD": "JNFD", "JTL": "JTL", "LSCZ": "LSCZ", "SNYG": "SNYG",
                "BFXT": "BFXT", "BGGF": "BGGF", "GSYH": "GSYH", "ZGLT": "ZGLT", "ZGPA": "ZGPA",
                "HRYY": "HRYY", "JSYH": "JSYH", "WLY": "WLY", "ZGSH": "ZGSH", "ZGSY": "ZGSY"}

    model = get_model(dropdown_model)
    if dropdown_highlow == ['Low']:
        valid, test, df_after, new_data = get_data(selected_dropdown,year,month,date,model,price_mode=0)
        temp = valid['Low']
    else:
        valid, test, df_after, new_data = get_data(selected_dropdown,year,month,date,model,price_mode=1)
        temp = valid['High']

    trace1 = go.Scatter(
        name = 'Actual data',
        x = valid.index,
        y = temp,
        mode='lines',
        opacity=0.7,
    )

    trace2 = go.Scatter(
        name='Valid data',
        x=valid.index,
        y=valid['Predictions'],
        mode='lines',
        opacity=0.7,
    )

    data = [trace1,trace2]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                        height=600,
                        title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                        xaxis={"title":"Date",
                               'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                   'step': 'month',
                                                                   'stepmode': 'backward'},
                                                                  {'count': 6, 'label': '6M',
                                                                   'step': 'month',
                                                                   'stepmode': 'backward'},
                                                                  {'step': 'all'}])},
                               'rangeslider': {'visible': True}, 'type': 'date'},
                         yaxis={"title":"Price"})}
    return figure




@app.callback(Output('Predicted Data', 'figure'),
              [Input('dropdown3', 'value'),Input('dropdown-highlow-3', 'value'),Input('dropdown-model-2', 'value'),
               Input('Year', 'value'),Input('Month', 'value'),Input('Date', 'value')])
# 更新图3
def update_graph(selected_dropdown,dropdown_highlow,dropdown_model,year,month,date):
    dropdown = {"GZMT": "GZMT", "JNFD": "JNFD", "JTL": "JTL", "LSCZ": "LSCZ", "SNYG": "SNYG",
                "BFXT": "BFXT", "BGGF": "BGGF", "GSYH": "GSYH", "ZGLT": "ZGLT", "ZGPA": "ZGPA",
                "HRYY": "HRYY", "JSYH": "JSYH", "WLY": "WLY", "ZGSH": "ZGSH", "ZGSY": "ZGSY"}

    model = get_model(dropdown_model)
    if dropdown_highlow == ['Low']:
        valid, test, df_after, new_data = get_data(selected_dropdown,year,month,date,model,price_mode=0)
        temp = test['Low']
    else:
        valid, test, df_after, new_data = get_data(selected_dropdown,year,month,date,model,price_mode=1)
        temp = test['High']


    figure = {'data': [go.Scatter
                       (x = df_after['Date'][len(new_data):],
                        y = temp,
                        mode='lines',
                        opacity=0.7,
                        # name=f'High {dropdown[stock]}',textposition='bottom center'
                        )],
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                        height=600,
                        title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                        xaxis={"title":"Date",
                               'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                   'step': 'month',
                                                                   'stepmode': 'backward'},
                                                                  {'count': 6, 'label': '6M',
                                                                   'step': 'month',
                                                                   'stepmode': 'backward'},
                                                                  {'step': 'all'}])},
                               'rangeslider': {'visible': True}, 'type': 'date'},
                         yaxis={"title":"Price"})}
    return figure




if __name__=='__main__':
    app.run_server(debug=True)