# 重点代码 readme
【代码文件】
1.presswithididentification.py：实时压力阵列读入，自动识别步态周期，计算特征，推理结果，存入文件
2.gpio.py：实现了根据步态识别结果对于LED、蜂鸣器的控制，并加入了对于摄像头的控制
3.client.py：客户端发送数据
4.server.py：服务器端接收数据，HTML方法日志交互
5.display_recognition_result.py：实现了通过HDMI连接显示屏进行识别结果可视化

【模型文件】
scaler.pkl、pca_model.pkl、random_forest_model.pkl：由pca+随机森林方法保存的分类模型
