import cvxpy as cp
import numpy as np
import multiprocessing
from functools import partial
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv
import snap
import warnings
from PIL import Image




#This is just for my part on the housing experiment
# Each part needs the respective adjacency matrix and then the constants
# needed for the f_i(x_i) needed for the objective, I also have a list of house
# numbers for identification.
#from generate_graph import adj_mat, train_data, labels, test_data, edge_pairs


mu=.5
rho=.001

#================================================================
#   print_graph(g) - takes in snap graph to print out, also
#   saves .png to current directory
#===============================================================
def print_graph(g : snap.PNGraph) -> None:
    labels = {}
    for NI in g.Nodes():
            labels[NI.GetId()] = str(NI.GetId())
    g.DrawGViz(snap.gvlDot, "output.png", " ", labels)
    img = Image.open('output.png')
    img.show()


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
    epsilon=10**(-8)


    n=np.shape(graph)[0]
    nodes=np.linspace(0,n-1,n,dtype='int')

    A=np.hstack((data_mat,np.ones((n,1))))


    #store space for z_i,j and u_i,j for the j neighbors of node i
    z_k=np.zeros((n,n,len(data_mat[0])+1))
    u_k=np.zeros((n,n,len(data_mat[0])+1))

    r=np.ones(n)
    del_r=np.ones(n)

    while LA.norm(del_r)>epsilon:


        #do x-update for each node, so this is n processes in parallel
        update_instance = partial(update_x, graph=graph,data_mat=data_mat,labels=labels,z=z_k,u=u_k)

        x_update_pool = multiprocessing.Pool(processes=n)

        outputs = x_update_pool.map(update_instance, nodes)

        x_k=np.array(outputs)


        #now that we have our new iterate x values for all the nodes, we update z and u

        z_update_instance=partial(update_z, graph=graph,x=x_k,u=u_k,l=l)

        z_update_pool = multiprocessing.Pool(processes=len(edges))

        z_outputs = z_update_pool.map(z_update_instance, edges)


        z=np.array(z_outputs)

        #unpack pooled z update
        for e in range(len(edges)):
            z_k[edges[e][0]][edges[e][1]]=z[e][0]
            z_k[edges[e][1]][edges[e][0]]=z[e][1]


        u_copy=np.copy(u_k)
        u_update_instance=partial(update_u,x=x_k,z=z_k,u_copy=u_copy)

        u_update_pool = multiprocessing.Pool(processes=len(edges))

        u_outputs = u_update_pool.map(u_update_instance, edges)

        u=np.array(u_outputs)

        #unpack pooled u update
        for e in range(len(edges)):
            u_k[edges[e][0]][edges[e][1]]=u[e][0]

        old_r=np.copy(r)
        #compute residuals
        for i in range(n):
            r[i]=A[i]@x_k[i]-labels[i]

        #break loop if we aren't improving any further for this lambda
        del_r=old_r-r

    return x_k,z_k,u_k

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

    x,z,u=admm(graph,edges,data_mat,labels,0)

    return x

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
    """
    i=np.random.random_integers(n-1)
    j=np.copy(i)
    while j == i:
        j=np.random.random_integers(n-1)

    x_values,z,u=admm(adj_mat,edges,data_mat,labels,0)

    x=(x_values[i]+x_values[j])/2

    #Change this for each experiment, the gradient evaluations of f_i and f_j
    #########################################################################
    df_i = 2*(data_mat[i] @ x[:-1]+x[-1]-labels[i])*np.hstack((data_mat[i],1))+mu*2*(np.hstack((x[:-1],0)))


    df_j = 2*(data_mat[j] @ x[:-1]+x[-1]-labels[j])*np.hstack((data_mat[j],1))+mu*2*(np.hstack((x[:-1],0)))
    ########################################################################
    """


    #l = .01*(LA.norm(df_i)+LA.norm(df_j))/(2*graph[i][j])
    #initial lambda
    l=.005
    percent_error=np.zeros(n)

    errors=np.empty(0)
    #while LA.norm(old_x-x) > 0:


    for k in range(60):

        x_old=np.copy(x)
        x,z,u=admm(adj_mat,train_data,labels,l)

        for i in range(n): percent_error[i]=np.abs(A[i]@x[i]-labels[i])/labels[i]
        errors=np.append(errors,LA.norm(percent_error))
        l+=.001
        print("l = ",l)
    #    print("error = ",errors[k])
        #print(LA.norm(x_old-x,1))
        if k % 10 == 0:
            print("Regularization path ",k,"% complete")

        if k > 1 and errors[k]>errors[k-1]:
            return x_old,errors

    return x,errors


#example on how to run for a node with random adjacency matrix and constants,
#try to have it in this form
np.random.seed(8)

"""
adj_mat=np.random.rand(5,5)
adj_mat=adj_mat @ adj_mat.T
for i in range(5): adj_mat[i][i]=0.0
train_data=np.random.rand(5,3)
labels=np.random.rand(5)


x=regularization_path(adj_mat,train_data,labels)
n=np.shape(adj_mat)[0]
A=np.hstack((train_data,np.ones((n,1))))
errors=np.zeros(n)


for i in range(n):
    errors[i]=np.abs(A[i]@x[i]-price[i])/labels[i]


x_geo,err_geo=geographic(adj_mat,train_data,labels)
x_mean=global_average(labels)
"""
n=np.shape(adj_mat)[0]
A=np.hstack((train_data,np.ones((n,1))))
"""
x_reg,err_reg=regularization_path(adj_mat,train_data,labels)
x_geo=geographic(adj_mat,train_data,labels)
x_avg=global_average(labels)
"""
