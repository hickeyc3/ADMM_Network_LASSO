import cvxpy as cp
import numpy as np
import csv
import snap
import warnings
from PIL import Image
import math

warnings.filterwarnings("ignore")
neighbors=3
remove_count_test=980
remove_count_val=0


#Generate graph, edge weights
file = open("Sacramentorealestatetransactions_Normalized.csv", "rU")
file.readline()
G =snap.TUNGraph.New()
locations = snap.TIntFltPrH()
dataset = snap.TIntFltVH()
counter = 0
for line in file:
    G.AddNode(counter)
    temp = snap.TFltPr(float(line.split(",")[10]),float(line.split(",")[11]))
    locations.AddDat(counter, temp)
    tempData = snap.TFltV()
    tempData.Add(float(line.split(",")[4]))
    tempData.Add(float(line.split(",")[5]))
    tempData.Add(float(line.split(",")[6]))

    if(line.split(",")[7] == "Residential"):
    	tempData.Add(1)
    elif(line.split(",")[7] == "Condo"):
    	tempData.Add(2)
    elif(line.split(",")[7] == "Multi-Family"):
    	tempData.Add(3)
    else:
    	tempData.Add(4)
    tempData.Add(float(line.split(",")[12])*10) #12 for normalized; 9 for raw
    dataset.AddDat(counter, tempData)
    counter += 1



#Remove random subset of nodes for test and validation sets
testList = snap.TIntV()
for i in range(remove_count_test):
	temp = G.GetRndNId()
	G.DelNode(temp)
	testList.Add(temp)

validationList = snap.TIntV()
for i in range(remove_count_val):
	temp = G.GetRndNId()
	G.DelNode(temp)
	validationList.Add(temp)

#For each node, find closest neighbors and add edge, weight = 5/distance
edgeWeights = snap.TIntPrFltH()
for NI in G.Nodes():
	distances = snap.TIntFltH()
	lat1 = locations.GetDat(NI.GetId()).GetVal1()
	lon1 = locations.GetDat(NI.GetId()).GetVal2()
	for NI2 in G.Nodes():
		if(NI.GetId() != NI2.GetId()):
			lat2 = locations.GetDat(NI2.GetId()).GetVal1()
			lon2 = locations.GetDat(NI2.GetId()).GetVal2()
			dlon = math.radians(lon2 - lon1)
			dlat = math.radians(lat2 - lat1)
			a2 = math.pow(math.sin(dlat/2),2) + math.cos(lat1)*math.cos(lat2) * math.pow(math.sin(dlon/2),2)
			c = 2 * math.atan2( math.sqrt(a2), math.sqrt(1-a2) )
			dist = 3961 * c
			distances.AddDat(NI2.GetId(), dist)

	distances.Sort(False, True)
	it = distances.BegI()
	for j in range(neighbors):
		if (not G.IsEdge(NI.GetId(), it.GetKey())):
			G.AddEdge(NI.GetId(), it.GetKey())
			#Add edge weight
			temp = snap.TIntPr(min(NI.GetId(), it.GetKey()), max(NI.GetId(), it.GetKey()))
			edgeWeights.AddDat(temp, 1/(it.GetDat()+ 0.1))
		it.Next()

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
house_data=np.zeros((len(house_list),3))
prices=np.zeros(len(house_list))
for l in file:
    line_count+=1

    if house_list[house_count]==line_count:
        h=l.split(",")


        #store beds constant
        house_data[house_count][0]=float(h[4])

        #store baths constant
        house_data[house_count][1]=float(h[5])

        #store sqft constant
        house_data[house_count][2]=float(h[6])

        #store actual price
        prices[house_count]=float(h[9])


        house_count+=1

    if house_count==len(house_list):
        break

#create adjacency matrix for final graph
adj_mat=np.zeros((nodes,nodes))

edge_list=np.zeros((edges,3))

k=0

#edgeWeights is a hash table with (int,int) pair keys and float values
#store all of snap structures edge weights in 2-D numpy adjacency matrix
for e in edgeWeights:

    i=e.GetVal1()
    j=e.GetVal2()
    weight=edgeWeights(e)

    edge_list[k][0]=i
    edge_list[k][1]=j
    edge_list[k][2]=weight

    k+=1



for i in edge_list:
    adj_mat[house_dict[i[0]]][house_dict[i[1]]]=i[2]
    adj_mat[house_dict[i[1]]][house_dict[i[0]]]=i[2]
"""
#Finally put the weight undirected graph into the form of an adjacency matrix
i=0
while i < len(edge_list):
    curr=house_dict[edge_list[i][0]]

    while i < len(edge_list) and curr==house_dict[edge_list[i][0]]:

        adj_mat[curr][house_dict[edge_list[i][1]]]=edge_list[i][2]
        adj_mat[house_dict[edge_list[i][1]]][curr]=edge_list[i][2]
        curr=house_dict[edge_list[i][0]]
        i+=1
"""
#print nodes, edges
