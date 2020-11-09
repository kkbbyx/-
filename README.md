# -
训练模型 python train.py

测试Flask python server.py

上线服务器 gunicorn -w 3 -t 30 -b 0.0.0.0:5000 app:app

服务器地址  42.194.146.191:5000
