#-*- coding:UTF8-*-
"""
created on Saturday 18th July 15:56 2020
@function:simple BP Neural Network
@creator:张平路

"""
import numpy as np
from sklearn import datasets   #导入手写数据集作为测试样本
from sklearn.model_selection import train_test_split
import math
import operator
import numpy as np
import struct
import matplotlib.pyplot as plt



# 测试集文件
train_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
train_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'



def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


"""
@function: set up a class to restore parameters
@parameter:input_size - the num of input
           hidden_size - the num of hidden 
           output_size - the number of classes need to be sorted
@return:none
"""
class network(object):
    def __init__(self,input_size,hidden_size,output_size,std = 1e-7):
        self.params = {}
        self.params['w1'] = std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = std*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
"""
@function:transform y
@parameter:y - a matrix
@return: Y - a transformed matrix
"""
def tranformed_y(y,output_size):
    Y = np.zeros([y.shape[0],output_size])
    for i in range(y.shape[0]):
        Y[i][int(y[i])] = 1
    return Y
"""
@function:define the loss function
@parameter:nt - the class of parameters
           X_train
           y_train
@return:E - the degree of loss
"""
def loss_function(nt,X_train,y_train):
    w1 = nt.params['w1']
    b1 = nt.params['b1']
    w2 = nt.params['w2']
    b2 = nt.params['b2']
    hidden = sigmoid(np.dot(X_train,w1)+b1)
    hidden = sigmoid(np.dot(hidden,w2)+b2)
    delta = y_train - hidden
    E = np.dot(delta,delta.T)*1/2
    return E
"""
@function:sigmoid function
@parameter: x  - independent variable
@return； y  - dependent variale
"""
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y
def Relu(x):
    for i in range(x.shape[1]):
        if x[0][i]<= 0:x[0][i] =0
    return x
"""
@function:BP
@parameter:nt - the class of network
           X_train - matrix
           y_train - too
           X_test - too
           y_test - too
           yita - the study rate
           
@return:
"""
def BP(nt,X_train, y_train,X_test, y_test,yita = 1e-2):
    w1 = nt.params['w1']
    b1 = np.mat(nt.params['b1'])
    w2 = nt.params['w2']
    b2 = np.mat(nt.params['b2'])
    maxiter = X_train.shape[0] #最大迭代次数

    for i in range(maxiter):
        h1 = np.array(np.dot(X_train[i], w1))
        print(y_train[i])
        hidden1 = np.mat(sigmoid(np.dot(X_train[i], w1) - b1))
        hidden2 = np.mat(sigmoid(np.dot(hidden1, w2) - b2))#神经网络得出的结果
        a = np.multiply(hidden2,1-hidden2)
        g = np.multiply(a,y_train[i]-hidden2)
        b = np.dot(g,np.transpose(w2))
        c = np.multiply(hidden1,1-hidden1)
        e = np.multiply(b,c)
        w2 += yita*np.dot(hidden1.T,g)
        b2 -= yita*g
        b1 -=yita*e
        X_temp = np.mat(X_train[i])
        w1 +=yita*np.dot(X_temp.T,e)
        loss = loss_function(nt,X_train[i],y_train[i])
        print("第%d次迭代后的损失为%f"%(i,loss))
    y = []
    right = 0
    #print(w2)
    for i in range(X_train.shape[0]):
        hidden1 = sigmoid(np.dot(X_train[i], w1) - b1)
        yi = sigmoid(np.dot(hidden1, w2) - b2) # 神经网络得出的结果
        #yi = np.array(yi)
        b = np.argmax(yi[0])
        #print(X_train[i])
       #m,n = max(enumerate(yi),key=operator.itemgetter(1))
        print(b)
        if np.argmax(y_train[i])== b:
            right +=1

        #max = 0
        index = 0
        #for j in range(yi.shape[1]):
           # if yi[j]>max:
                ##
        # index  = j
        #y.append(index)


    print(right)
    print(w1)










    return

"""
@function:decide the hidden size
@parameter:input_size - the num of input
           output_size - the num of output
           alpha - parameter
@return: hidden_size
"""
def hidden_select(inpurt_size,ouput_size,alpha):
    hidden_size = math.ceil(math.sqrt(input_size+output_size)+alpha)
    return  hidden_size

"""
@function:main function
@parameter:none
@return: none
"""
if __name__ == '__main__':
    digit = datasets.load_digits()#read the data of digits
    images = digit.images
    targets = digit.target
    X_train,X_test,y_train,y_test = train_test_split(images,targets,test_size=0.2,random_state=0)#split the data into two parts
    X_train_axis_0  =X_train.shape[0]
    X_train = X_train.reshape(X_train_axis_0,-1)
    X_test_axis_0  = X_test.shape[0]
    X_test = X_test.reshape(X_test_axis_0,-1)

    input_size = X_train.shape[1]
    output_size = 10
    y_test = tranformed_y(y_test,output_size)
    y_train = tranformed_y(y_train,output_size)
    hidden_size  = hidden_select(input_size,output_size,2)



   # BP(nt,X_train,y_train,X_test,y_test)
    train_images = load_train_images()

    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    #for i in range(10):
     #   print(train_labels[i])
      #  plt.imshow(train_images[i], cmap='gray')
       # plt.pause(0.000001)
        #plt.show()
    #print('done')
    train_images = train_images.reshape((10000,784))
    train_labels = tranformed_y(train_labels,10)
    input_size = train_images.shape[1]
    output_size = 10

    hidden_size = hidden_select(input_size, output_size, 2)
    nt = network(input_size, hidden_size, output_size)
    BP(nt, train_images, train_labels, train_images, train_labels)
    print(train_labels)






