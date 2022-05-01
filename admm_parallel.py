import cvxpy as cp
import numpy as np
import multiprocessing
from functools import partial
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv
import warnings
from PIL import Image


#This is just for my part on the housing experiment
# Each part needs the respective adjacency matrix and then the constants
# needed for the f_i(x_i) needed for the objective, I also have a list of house
# numbers for identification.
from generate_graph import generate_graph

train_set_size=20
test_set_size=10

adj_mat_train, train_data, train_labels, edge_pairs_train, test_data, test_labels = generate_graph(train_set_size,test_set_size)


mu=.5
rho=.001

#max allowable number of processes to be spawned
procs=100


#===============================================================================
#   update_x(i,graph,data_mat,z,u) - allows x update to run in parallel
#   i     - node the update is being run for
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - train_data constants matrix for regresssion problem (swap for your experiment)
#   labels - labels of each house, the actual data point values for each experiment (swap for you experiment)
#   z       - current z
#   u       - current u
#================================================================================
def update_x(i : int,
            graph : np.array,
            data_mat : np.array,
            labels : np.array,
            z : np.array,
            u : np.array) -> np.array:

    x=cp.Variable(len(data_mat[0]))
    x_offset=cp.Variable(1)

    #Change for each experiment, this is the f_i(x_i) in the obj func
    #################################################
    obj = cp.sum_squares(data_mat[i] @ x+x_offset-labels[i])+ mu*cp.sum_squares(x)
    ################################################


    #add terms to obj func for each neighbor
    for j in range(len(graph[i])):
        if graph[i][j] != 0:
            obj += rho*mu*cp.sum_squares(cp.hstack((x,x_offset))-z[i][j]+u[i][j])


    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return np.hstack((x.value,x_offset.value))

#===============================================================================
#   update_z(edge,graph,x,u,l) - allows z update to run in parallel
#   edge     - edge the update is being run for
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   x       - current x
#   u       - current u
#================================================================================
def update_z(edge : tuple,
            graph : np.array,
            x : np.array,
            u : np.array,
            l : float) -> np.array:

    i=edge[0]
    j=edge[1]
    t=(l*graph[i][j])/(rho*LA.norm(x[i]+u[i][j]-x[j]-u[j][i]))
    theta=max(.5, 1-t)

    z_ij=theta*(x[i]+u[i][j])+(1-theta)*(x[j]+u[j][i])
    z_ji=(1-theta)*(x[i]+u[i][j])+theta*(x[j]+u[j][i])


    return z_ij,z_ji

#===============================================================================
#   update_u(edge,x,z,u_copy) - allows u update to run in parallel
#   edge    - edge the update is being run for
#   z       - current z
#   x       - current x
#   u_copy   - shallow copy of current u to get current u[i][j] value
#================================================================================
def update_u(edge : tuple,
            x : np.array,
            z : np.array,
            u_copy : np.array) -> np.array:

    i=edge[0]
    j=edge[1]

    return u_copy[i][j]+(x[i]-z[i][j])



#===============================================================================
#   admm (graph,data_mat,labels,l) - run convex fomrulation for each node i independently
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - train_data constants matrix for regresssion problem (swap for your experiment)
#   labels - labels of each house, the actual data point values for each experiment (swap for you experiment)
#   l - lambda, regularization term updated at each path
#================================================================================
def admm(graph : np.array,
        edges : list,
        data_mat : np.array,
        labels : np.array,
        l : float) -> np.array:


    #seems to diverge if any smaller?
    epsilon=10**(-7)


    n=np.shape(graph)[0]
    nodes=np.linspace(0,n-1,n,dtype='int')

    A=np.hstack((data_mat,np.ones((n,1))))


    #store space for z_i,j and u_i,j for the j neighbors of node i
    z_k=np.zeros((n,n,len(data_mat[0])+1))
    u_k=np.zeros((n,n,len(data_mat[0])+1))

    r=np.zeros(n)

    old_r=np.ones(n)

    while LA.norm(old_r-r)>epsilon:


        #do x-update for each node, so this is n processes in parallel
        update_instance = partial(update_x, graph=graph,data_mat=data_mat,labels=labels,z=z_k,u=u_k)

        x_update_pool = multiprocessing.Pool(processes=procs)

        outputs = x_update_pool.map(update_instance, nodes)

        x_k=np.array(outputs)


        #now that we have our new iterate x values for all the nodes, we update z and u

        z_update_instance=partial(update_z, graph=graph,x=x_k,u=u_k,l=l)

        z_update_pool = multiprocessing.Pool(processes=procs)

        z_outputs = z_update_pool.map(z_update_instance, edges)


        z=np.array(z_outputs)

        #unpack pooled z update
        for e in range(len(edges)):
            z_k[edges[e][0]][edges[e][1]]=z[e][0]
            z_k[edges[e][1]][edges[e][0]]=z[e][1]


        u_copy=np.copy(u_k)
        u_update_instance=partial(update_u,x=x_k,z=z_k,u_copy=u_copy)

        u_update_pool = multiprocessing.Pool(processes=procs)

        u_outputs = u_update_pool.map(u_update_instance, edges)

        u=np.array(u_outputs)

        #unpack pooled u update
        for e in range(len(edges)):
            u_k[edges[e][0]][edges[e][1]]=u[e][0]


        old_r=np.copy(r)

        #compute primal residuals
        for i in range(n):
            r[i]=A[i]@x_k[i]-labels[i]


    return x_k

#===============================================================================
#  global_average(labels) : baseline method to compute global average of house labels
#================================================================================
def global_average(labels):
    return np.sum(labels)/len(labels)

#===============================================================================
#  geographic (graph,data_mat,labels) - geographic (lambda=0) baseline method,
#   same arguments as regularization_path
#================================================================================
def geographic( graph : np.array,
                edges : list,
                data_mat : np.array,
                labels : np.array) -> np.array:

    return admm(graph,edges,data_mat,labels,0)

#===============================================================================
#  regularization_path (graph,data_mat,labels) - runs ADMM with increasing lambda values
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - train_data constants matrix for regresssion problem (swap for your experiment)
#   labels - labels of each house, the actual data point values for each experiment (swap for you experiment)
#================================================================================
def regularization_path( graph : np.array,
                        edges : list,
                        data_mat : np.array,
                        labels : np.array) -> np.array:


    n=np.shape(graph)[0]
    A=np.hstack((train_data,np.ones((n,1))))


    #initial lambda
    #l=3000
    l=3000
    percent_error=np.zeros(n)

    errors=np.empty(0)
    #while LA.norm(old_x-x) > 0:

    x=0
    for k in range(8):

        x_old=np.copy(x)
        x=admm(graph,edges,data_mat,labels,l)

        for i in range(n): percent_error[i]=np.abs(A[i]@x[i]-labels[i])/labels[i]
        errors=np.append(errors,LA.norm(percent_error))
        l+=1

        if LA.norm(x_old-x)<10**(-4) or (k > 1 and errors[k]>errors[k-1]):
            return x_old

    return x


#example on how to run for a node with random adjacency matrix and constants,
#try to have it in this form
np.random.seed(8)
"""
adj_mat=np.random.rand(5,5)
adj_mat=adj_mat @ adj_mat.T
for i in range(5): adj_mat[i][i]=0.0

adj_mat[3][2]=0
adj_mat[2][3]=0
adj_mat[4][1]=0
adj_mat[1][4]=0
train_data=np.random.rand(5,3)
labels=np.random.rand(5)
edge_pairs=[]

for i in range(5):
    for j in range(i,5):

        if adj_mat[i][j]!=0:
            edge_pairs.append((i,j))
"""


# run admm with
#x=admm(adj_mat_train,edge_pairs_train,train_data,train_labels,.5)

x_reg=regularization_path(adj_mat_train,edge_pairs_train,train_data,train_labels)
x_geo=geographic(adj_mat_train,edge_pairs_train,train_data,train_labels)
avg=global_average(train_labels)

geo_errors=np.empty(0)
reg_errors=np.empty(0)
global_avg_errors=np.empty(0)

for i in range(len(test_labels)):
    if test_labels[i] != 0:
        geo_errors=np.append(geo_errors,np.abs(test_data[i]@x_geo[i][:-1]+x_geo[i][-1]-test_labels[i])**2)
        reg_errors=np.append(reg_errors,np.abs(test_data[i]@x_reg[i][:-1]+x_reg[i][-1]-test_labels[i])**2)
        global_avg_errors=np.append(global_avg_errors,(avg-test_labels[i])**2)



err=np.array([np.sum(geo_errors)/test_set_size,np.sum(reg_errors)/test_set_size,np.sum(global_avg_errors)/test_set_size])
#np.savetxt(f"errors_size_{train_set_size}", err, delimiter = ",")
