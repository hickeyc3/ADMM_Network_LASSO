import cvxpy as cp
import numpy as np
import csv
import snap
import warnings
from PIL import Image
import math

warnings.filterwarnings("ignore")



neighbors=5
train_set_size=20
test_set_size=20


file = open("Sacramentorealestatetransactions_Normalized.csv", "rU")
file.readline()

i=0
for line in file:

    h=l.split(",")

    full_data=np.append(full_data,np.zeros())
    #store beds constant
    full_data[i][0]=float(h[4])

    #store baths constant
    train_data[i][1]=float(h[5])

    #store sqft constant
    train_data[i][2]=float(h[6])

    #store actual price
    labels[i]=float(h[9])




exit()




nodes = G.GetNodes()
edges = G.GetEdges()

house_list=[]

for NI in G.Nodes():
        house_list.append(NI.GetId())

house_list.sort()
house_dict={}
for i in range(len(house_list)):
    house_dict[house_list[i]]=i


file = open("Sacramentorealestatetransactions_Normalized.csv", "rU")
file.readline()

line_count=0
house_count=0

#store relevant house data for features and actual price
train_data=np.zeros((len(house_list),3))
test_data=np.zeros((len(house_list),3))
labels=np.zeros(len(house_list))
for l in file:
    line_count+=1

    if house_list[house_count]==line_count:
        h=l.split(",")


        #store beds constant
        train_data[house_count][0]=float(h[4])

        #store baths constant
        train_data[house_count][1]=float(h[5])

        #store sqft constant
        train_data[house_count][2]=float(h[6])

        #store actual price
        labels[house_count]=float(h[9])


        house_count+=1

    if house_count==len(house_list):
        break

#create adjacency matrix for final graph
adj_mat=np.zeros((nodes,nodes))

edge_list=np.zeros((edges,3))
edge_pairs=[]
k=0

#edgeWeights is a hash table with (int,int) pair keys and float values
#store all of snap structures edge weights in 2-D numpy adjacency matrix
for e in edgeWeights:

    i=e.GetVal1()
    j=e.GetVal2()

    edge_pairs.append((int(house_dict[i]),int(house_dict[j])))
    weight=edgeWeights(e)

    edge_list[k][0]=i
    edge_list[k][1]=j
    edge_list[k][2]=weight

    k+=1



for i in edge_list:
    adj_mat[house_dict[i[0]]][house_dict[i[1]]]=i[2]
    adj_mat[house_dict[i[1]]][house_dict[i[0]]]=i[2]
