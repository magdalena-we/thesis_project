import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def apply_sigmoid(x_list):

    y_list = [] 
    for x in x_list:
        y = sigmoid(x)
        y_list.append(y)
    
    return y_list


#x_list = []
#y_list = apply_sigmoid(x_list)
#print(y_list)