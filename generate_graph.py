import cvxpy as cp
import numpy as np
import csv
import warnings
from PIL import Image
import math

#uncomment to visualize graph
#import matplotlib.pyplot as plt
#import networkx as nx


np.random.seed(8)

warnings.filterwarnings("ignore")

neighbors=5
train_set_size=700
test_set_size=20



file = open("Sacramentorealestatetransactions_Normalized.csv", "rU")
file.readline()


full_data=np.zeros((985,6))
addr=[]
i=0
for l in file:

    h=l.split(",")

    if float(h[4]) != 0 and float(h[12]) != 0:

        #store beds constant
        full_data[i][0]=float(h[4])

        #store baths constant
        full_data[i][1]=float(h[5])

        #store sqft constant
        full_data[i][2]=float(h[6])

        #store normalized actual price
        full_data[i][3]=float(h[12])

        #store latitude
        full_data[i][4]=float(h[10])

        #store longitude
        full_data[i][5]=float(h[11])

        addr.append(h[0])
        #train_data[i][4]=float
        i+=1


full_count=i
full_data=full_data[:full_count]

train_data=np.zeros((train_set_size,3))
train_list=[]
train_labels=np.zeros(train_set_size)
train_locations=[]

for k in range(train_set_size):

    i=np.random.randint(full_count)

    train_data[k]=full_data[i][:3]
    train_labels[k]=full_data[i][3]
    train_locations.append((full_data[i][4],full_data[i][5]))
    train_list.append((i,addr[i]))

    full_data=np.delete(full_data,i,0)



    full_count=full_count-1

test_data=np.zeros((test_set_size,3))
test_list=[]
test_labels=np.zeros(test_set_size)
test_locations=[]

for k in range(test_set_size):
    i=np.random.randint(full_count)

    test_data[k]=full_data[i][:3]
    test_labels[k]=full_data[i][3]
    test_locations.append((full_data[i][4],full_data[i][5]))
    test_list.append((i,addr[i]))

    full_data=np.delete(full_data,i,0)
    full_count=full_count-1


#make adjacency matrix for adj_mat_train
adj_mat_train=np.zeros((train_set_size,train_set_size))

#get the 5 nearest nodes to each other node
distances=[]

for i in range(train_set_size):
    distances.append([])
    for j in range(train_set_size):

        if i != j:
            distances[i].append((math.sqrt((train_locations[i][0]-train_locations[j][0])**2+(train_locations[i][1]-train_locations[i][1])**2),j))

    distances[i].sort(key = lambda node: node[0])



for i in range(train_set_size):
    for j in range(neighbors):

        neighbor=distances[i][j][1]
        if distances[i][j][0] == 0:
            adj_mat_train[i][neighbor]=0
            adj_mat_train[neighbor][i]=0

        else:
            adj_mat_train[i][neighbor]=5/distances[i][j][0]
            adj_mat_train[neighbor][i]=5/distances[i][j][0]


#make adjacency matrix for adj_mat_test
adj_mat_test=np.zeros((test_set_size,test_set_size))
edge_pairs_test=[]

#get the 5 nearest nodes to each other node
distances=[]

for i in range(test_set_size):
    distances.append([])
    for j in range(test_set_size):

        if i != j:
            distances[i].append((math.sqrt((test_locations[i][0]-test_locations[j][0])**2+(test_locations[i][1]-test_locations[i][1])**2),j))

    distances[i].sort(key = lambda node: node[0])


for i in range(test_set_size):
    for j in range(neighbors):

        neighbor=distances[i][j][1]
        if distances[i][j][0] == 0:
            adj_mat_test[i][neighbor]=0
            adj_mat_test[neighbor][i]=0
        else:
            adj_mat_test[i][neighbor]=5/distances[i][j][0]
            adj_mat_test[neighbor][i]=5/distances[i][j][0]


edge_pairs_train=[]

for i in range(train_set_size):
    for j in range(i,train_set_size):

        if adj_mat_train[i][j]!=0:
            edge_pairs_train.append((i,j))


edge_pairs_test=[]

for i in range(test_set_size):
    for j in range(i,test_set_size):

        if adj_mat_test[i][j]!=0:
            edge_pairs_test.append((i,j))


#uncomment to visualize graph
"""
g=nx.from_numpy_matrix(adj_mat_train)
nx.draw(g, with_labels=True, node_size=100, alpha=1, linewidths=10)
plt.show()
"""
