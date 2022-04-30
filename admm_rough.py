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
#from generate_graph import adj_mat, house_data, house_list, house_dict, prices


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
#   update_x(i,graph,data_mat,z,u)) - allows x update to run in parallel
#   i     - node the update is being run for
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - house_data constants matrix for regresssion problem (swap for your experiment)
#   prices - prices of each house, the actual data point values for each experiment (swap for you experiment)
#   z       - current z, not updated in parallel to avoid data hazard
#   u       - current u, not updated in parallel to avoid data hazard
#================================================================================
def update_x(i : int,
            graph : np.array,
            data_mat : np.array,
            prices : np.array,
            z : np.array,
            u : np.array) -> np.array:

    x=cp.Variable(len(data_mat[0]))
    x_offset=cp.Variable(1)

    #Change for each experiment, this is the f_i(x_i) in the obj func
    #################################################
    obj = cp.sum_squares(data_mat[i] @ x+x_offset-prices[i])+ mu*cp.sum_squares(x)
    ################################################


    #add terms to obj func for each neighbor
    for j in range(len(graph[i])):
        if graph[i][j] != 0:
            obj += rho*mu*cp.sum_squares(cp.hstack((x,x_offset))-z[i][j]+u[i][j])


    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return np.hstack((x.value,x_offset.value))



#===============================================================================
#   admm_convex (graph,data_mat,prices,l) - run convex fomrulation for each node i independently
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - house_data constants matrix for regresssion problem (swap for your experiment)
#   prices - prices of each house, the actual data point values for each experiment (swap for you experiment)
#   l - lambda, regularization term updated at each path
#================================================================================
def admm_convex(graph : np.array,
                data_mat : np.array,
                prices : np.array,
                l : float) -> np.array:


    #seems to diverge if any smaller?
    epsilon=10**(-8)


    n=np.shape(graph)[0]
    nodes=np.linspace(0,n-1,n,dtype='int')

    A=np.hstack((data_mat,np.ones((n,1))))


    #store space for z_i,j and u_i,j for the j neighbors of node i
    z=np.zeros((n,n,len(data_mat[0])+1))
    u=np.zeros((n,n,len(data_mat[0])+1))

    r=np.ones(n)
    del_r=np.ones(n)

    while LA.norm(del_r)>epsilon:
    #while LA.norm(r)>epsilon:

        #do x-update for each node, so this is n processes in parallel
        update_instance = partial(update_x, graph=graph,data_mat=data_mat,prices=prices,z=z,u=u)

        x_update_pool = multiprocessing.Pool(processes=n)

        outputs = x_update_pool.map(update_instance, nodes)

        x_k=np.array(outputs)


        #now that we have our new iterate x values for all the nodes, we update z and u
        for i in range(n):
            for j in range(n):
                if graph[i][j] != 0:

                    t=(l*graph[i][j])/(rho*LA.norm(x_k[i]+u[i][j]-x_k[j]-u[j][i]))
                    theta=max(.5, 1-t)

                    z[i][j]=theta*(x_k[i]+u[i][j])+(1-theta)*(x_k[j]+u[j][i])
                    z[j][i]=(1-theta)*(x_k[i]+u[i][j])+theta*(x_k[j]+u[j][i])
                    u[i][j]+=(x_k[i]-z[i][j])



        old_r=np.copy(r)
        #compute residuals
        for i in range(n):
            r[i]=A[i]@x_k[i]-prices[i]


        #break loop if we aren't improving any further for this lambda
        del_r=old_r-r

    return x_k,z,u

#===============================================================================
#  global_average(prices) : baseline method to compute global average of house prices
#================================================================================
def global_average(prices):
    return np.sum(prices)/len(prices)

#===============================================================================
#  geographic (graph,data_mat,prices) - geographic (lambda=0) baseline method,
#   same arguments as regularization_path
#================================================================================
def geographic( graph : np.array,
                data_mat : np.array,
                prices : np.array) -> np.array:


    n=np.shape(graph)[0]
    A=np.hstack((house_data,np.ones((n,1))))

    i=1
    percent_error=np.zeros(n)

    errors=np.empty(0)
    #while LA.norm(old_x-x) > 0:
    while i < 100:

        x,z,u=admm_convex(adj_mat,house_data,prices,0)

        for i in range(n): percent_error[i]=np.abs(A[i]@x[i]-prices[i])/prices[i]
        errors=np.append(errors,LA.norm(percent_error))
        i+=1

        if i % 10 == 0:
            print("Geographic method ",l,"% complete")
        #print (LA.norm(old_x-x))

    return x,errors

#===============================================================================
#  regularization_path (graph,data_mat,prices) - runs ADMM with increasing lambda values
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - house_data constants matrix for regresssion problem (swap for your experiment)
#   prices - prices of each house, the actual data point values for each experiment (swap for you experiment)
#================================================================================
def regularization_path( graph : np.array,
                        data_mat : np.array,
                        prices : np.array) -> np.array:


    n=np.shape(graph)[0]
    A=np.hstack((house_data,np.ones((n,1))))

    i=np.random.random_integers(n-1)
    j=np.copy(i)
    while j == i:
        j=np.random.random_integers(n-1)

    x_values,z,u=admm_convex(adj_mat,house_data,prices,0)

    x=(x_values[i]+x_values[j])/2

    #Change this for each experiment, the gradient evaluations of f_i and f_j
    #########################################################################
    df_i = 2*(data_mat[i] @ x[:-1]+x[-1]-prices[i])*np.hstack((data_mat[i],1))+mu*2*(np.hstack((x[:-1],0)))


    df_j = 2*(data_mat[j] @ x[:-1]+x[-1]-prices[j])*np.hstack((data_mat[j],1))+mu*2*(np.hstack((x[:-1],0)))
    ########################################################################

#    cp.sum_squares(data_mat[i] @ x_values[i]+x_offsets[i]-prices[i])+ mu*cp.sum_squares(x_variables[i])

    #initial lambda
    l = .01*(LA.norm(df_i)+LA.norm(df_j))/(2*graph[i][j])


    #TO DO: fix initial lambda if possible
    l=1
    percent_error=np.zeros(n)

    errors=np.empty(0)
    #while LA.norm(old_x-x) > 0:
    while l < 100:

        x,z,u=admm_convex(adj_mat,house_data,prices,l)

        for i in range(n): percent_error[i]=np.abs(A[i]@x[i]-prices[i])/prices[i]
        errors=np.append(errors,LA.norm(percent_error))
        l+=1

        if l % 10 == 0:
            print("Regularization path ",l,"% complete")
        #print (LA.norm(old_x-x))

    return x,errors


#example on how to run for a node with random adjacency matrix and constants,
#try to have it in this form
np.random.seed(8)


adj_mat=np.random.rand(5,5)
adj_mat=adj_mat @ adj_mat.T
for i in range(5): adj_mat[i][i]=0.0
house_data=np.random.rand(5,3)
prices=np.random.rand(5)


"""

x=regularization_path(adj_mat,house_data,prices)

"""
