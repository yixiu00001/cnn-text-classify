
# coding: utf-8

# In[1]:

#得到准确率、召回率、以及数量统计的list
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

def testPrecision(predict_label, true_label):
        
    A = dict.fromkeys(true_label,0)  #预测正确的各个类的数目
    B = dict.fromkeys(true_label,0)   #测试数据集中实际的各个类的数目
    C = dict.fromkeys(predict_label,0) #预测结果中各个类的数目
    for i in range(0,len(true_label)):
        B[true_label[i]] += 1
        C[predict_label[i]] += 1
        if true_label[i] == predict_label[i]:
            A[true_label[i]] += 1
    #print("predict is right : ",A)
    print("true count : ", B)
    #print("predict count : ", C)
    
    key_list = []
    true_count_list = []
    precision_list = []
    recall_list = []
    f_list = []
    
    for i in range(21):
        key_list.append(i)
        true_count_list.append(int(B[float(i)]))
        try:
            precision = A[float(i)]*1.0/C[float(i)]
            precision_list.append(precision)
        except:
            precision_list.append(0.00001)
        try:
            recall = A[float(i)]*1.0/B[float(i)]
            recall_list.append(recall)
        except:
            recall_list.append(0.00001)
        try:
            f = precision * recall * 2 / (precision + recall)
            f_list.append(f)
        except:
            f_list.append(0.00001)
            
        
    return precision_list, recall_list, f_list, key_list, true_count_list


# In[3]:

#矩阵的优化显示
def matrixDisplay(matrix):
    shape = matrix.shape
    #求矩阵中的最大位数：
    max_digits = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if len(str(matrix[i][j]))> max_digits:
                max_digits = len(str(matrix[i][j]))
    #拼接矩阵中的行，转为字符串，输出
    for i in range(shape[0]):
        row_str = ""
        for j in range(shape[1]):
            if len(str(matrix[i][j]))< max_digits:
                space_str = ""
                for u in range(max_digits-len(str(matrix[i][j]))):
                    space_str = space_str + " "
                row_str = row_str + str(matrix[i][j])+ space_str +"  "
            else:
                row_str = row_str + str(matrix[i][j])+"  "
        print(row_str)  


# In[4]:

#条形图的输出
import matplotlib.pyplot as plt
import numpy as np
def comparisonPlot(label,y1,y2): 
    index = np.arange(0,21,1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    names = (u'F-Score', u'Quantity')
    subjects = index
    bar_width = 0.35

    ax1.bar(index, y1, bar_width, color='#0072BC', label=names[0])

    ax1.set_xticks(index)  
    ax1.set_xticklabels(label, rotation=0)
    ax1.set_ylabel("The F-Score of each firsttype")
    ax1.set_xlabel("First Type Index")
    
    ax2 = ax1.twinx()  # this is the important function
    ax2.bar(index + bar_width, y2, bar_width, color='#ED1C24', label=names[1])
    ax1.set_xticks(index + bar_width)  
    ax2.set_ylabel("The quantity of each firsttype")
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, -0.12))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.65, -0.12))
    plt.show()


# In[ ]:



