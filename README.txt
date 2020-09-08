运行环境：
python3.7 + pycharm2019.3

导入库：
dash==1.13.4
dash-core-components==1.10.1
dash-html-components==1.0.3
Keras==2.1.5
matplotlib==3.3.0
numpy==1.19.1
pandas==0.25.1
plotly==4.9.0
sklearn==0.0
tensorflow==2.2.0

操作说明：
1、获取股票预测模型：
（1）通过更改stock_pred.py line 209，选取不同股票数据进行导入。
（2）将stock_pred.py line 217 train_mode参数置1。
（3）运行程序，可得到训练模型。
注：可通过更改stock_pred.py line 188 train_test_ratio参数来调整训练测试比

2、预测未来五天股价：
（1）将stock_pred.py line 217 train_mode参数置0
（2）更改stock_pred.py line 299的导入路径，以选择不同的模型进行预测
（3）运行程序，得到预测结果与股价图。

3、网页版显示：
（1）运行程序后，进入http://127.0.0.1:8050网页
（2）通过更改输入文本框中的日期改变指定日期
（3）通过选择下拉列表中的选项来选择所预测的股票、最高价或最低价以及导入模型。

注：简写股票对应关系：		代号模型对应关系：
GZMT——贵州茅台			model_1——贵州茅台 训练所得
JNFD——节能风电			model_2——节能风电 训练所得
JTL——京天利			model_3——京天利    训练所得
LSCZ——兰石重装			model_4——兰石重装 训练所得
SNYG——苏宁易购			model_5——苏宁易购 训练所得
BFXT——北方稀土			model_6——北方稀土 训练所得
BGGF——宝钢股份			model_7——宝钢股份 训练所得
GSYH——工商银行			model_8——工商银行 训练所得
ZGLT——中国联通			model_9——中国联通 训练所得
ZGPA——中国平安			model_10——中国平安 训练所得
HRYY——恒瑞医药
JSYH——建设银行
WLY——五粮液
ZGSH——中国神华
ZGSY——中国石油
