#-*-coding:UTF8-*-
"""
created on July 22th 11:50 2020
@author:张平路
@function: realize the random forest

"""


import C45
import numpy as np
import json
import math
import random


"""
@functon:the access to the programme
@parameter:none
@return:none
"""
if __name__ == '__main__':
    fr = open('lenses.txt')
    DataSet = [i.strip().split('  ') for i in fr]

    label = DataSet[0][1:-1]
    label_full = DataSet[0][1:-1]
    dataset = [row[1:] for row in DataSet[1:]]
    data_full = [row[1:] for row in DataSet[1:]]
    row_num = len(data_full)
    selected_row_num = int(math.log(row_num, 2)) + 1
    col_num = len(label_full) - 1
    selected_col_num = int(math.log(col_num, 2)) + 1
    result = [[] for i in range(len(dataset))]


    for m in range(100):#100 decisiontree


        row = []
        for i in range(selected_row_num):
            row.append(random.randint(0,np.shape(dataset)[0]-1))


        col = []
        for i in range(selected_col_num):
            col.append(random.randint(0, np.shape(dataset)[1] - 2))
        

        data = []
        data_temp_full = []
        temp = []
        # print(row,col)
        for i in range(len(row)):
            for j in range(len(col)):
                temp.append(dataset[row[i]][col[j]])
                # print('temp',temp)
            temp.append(dataset[row[i]][-1])

            data.append(temp)
            data_temp_full.append(temp)
            temp = []
            # print('data',data)
        label_temp = []
        label_temp_full = []
        for i in range(len(col)):
            label_temp.append(label[col[i]])
            label_temp_full.append(label[col[i]])



        mytree = C45.createTree(data, label_temp, data_temp_full, label_temp_full)
        # print(json.dumps(mytree, ensure_ascii=False, indent=4))
        predict1 = []
        if type(mytree).__name__ == 'str':
            for i in range(len(data_temp_full)):
                predict1.append(mytree)
        else:
            key = list(mytree.keys())[0]
            value = list(mytree.values())[0]

            for i in range(len(data_temp_full)):
                predict1.append(C45.predict(data_temp_full[i][:-1], data_temp_full[i][-1], label_temp_full, key, value)[0][0])

        for i in range(selected_row_num):
            result[row[i]].append(predict1[i])
    #print(result)
    right = 0
    for i in range(len(result)):
        counts = np.bincount(result[i])

        result[i] = str(np.argmax(counts))
        if result[i] == dataset[i][-1]:
            right+=1
    print("精度为：",float(right/len(result)))




    #build 100 trees
    #for i in range(100):



    #mytree = C45.createTree(dataset,label,data_full,label_full)


    #key = list(mytree.keys())[0]
    #value = list(mytree.values())[0]
    #predict1 = []
    #print(label)

    #for i in range(len(data_full)):
        #predict1.append(C45.predict(data_full[i][:-1], data_full[i][-1], label_full, key, value)[0][0])
    #print(predict1)

