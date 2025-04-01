import numpy as np
import matplotlib.pyplot as plt

def data_read(dir_path):
    with open(dir_path,'r') as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(',')
        data_range = len(data)
    return np.asarray(data, float),data_range

if __name__ == "__main__":

    train_loss_path = r'./output/train_acc.txt'
    val_loss_path = r'./output/val_acc.txt'
    y_train_loss, data_range = data_read(train_loss_path)  # loss值，即y轴
    y_val_loss, data_range = data_read(val_loss_path)
    x_train_loss = range(1,data_range + 1)			 # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')    # x轴标签
    plt.ylabel('acc')     # y轴标签
	
	# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
	# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train acc")
    plt.plot(x_train_loss, y_val_loss, linewidth=1, linestyle="solid",color='blue', label="val acc")
    plt.legend()
    plt.title('acc curve')
    plt.savefig("./acc.jpg")
    plt.show()
