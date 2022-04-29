import cvxpy as cp
import numpy as np
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
#   admm_convex (G,c,l) - run convex fomrulation for each node i independently
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



    rho=.001

    n=np.shape(graph)[0]
    A=np.hstack((data_mat,np.ones((n,1))))
    #make a vector of n cvxpy variables, one for each node
    x_variables=[]
    x_offsets=[]


    for _ in range(n):
        x_variables.append(cp.Variable(len(data_mat[0])))
        x_offsets.append(cp.Variable(1))


    #store space for x_k, z_i,j and u_i,j for the j neighbors of node i
    x_k=np.zeros((n,len(data_mat[0])+1))
    z=np.zeros((n,n,len(data_mat[0])+1))
    u=np.zeros((n,n,len(data_mat[0])+1))

    r=np.ones(n)
    del_r=np.ones(n)

    while LA.norm(del_r)>epsilon:
    #while LA.norm(r)>epsilon:


        #do x-update for each node
        for i in range(n):

            #Change for each experiment, this is the f_i(x_i) in the obj func
            #################################################
            obj = cp.sum_squares(data_mat[i] @ x_variables[i]+x_offsets[i]-prices[i])+ mu*cp.sum_squares(x_variables[i])
            ################################################


            #add terms to obj func for each neighbor
            for j in range(len(graph[i])):
                if graph[i][j] != 0:
                    obj += rho*mu*cp.sum_squares(cp.hstack((x_variables[i],x_offsets[i]))-z[i][j]+u[i][j])


            prob = cp.Problem(cp.Minimize(obj))
            prob.solve()

            x_k[i]=np.hstack((x_variables[i].value,x_offsets[i].value))



            #update z and u
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


        del_r=old_r-r
        #print(LA.norm(r))
    return x_k,z,u

def regularization_path(graph, data_mat, prices):

    n=np.shape(graph)[0]
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

    old_x=np.ones(len(x))

    #while LA.norm(old_x-x) > 0:
    while l < 100:
        old_x=np.copy(x)
        x,z,u=admm_convex(adj_mat,house_data,prices,l)
        l+=1

        if l % 10 == 0:
            print(l,"% complete")
        #print (LA.norm(old_x-x))

    return x


#example on how to run for a node with random adjacency matrix and constants,
#try to have it in this form
np.random.seed(8)

adj_mat=np.random.rand(5,5)
adj_mat=adj_mat @ adj_mat.T
for i in range(5): adj_mat[i][i]=0.0
house_data=np.random.rand(5,3)
prices=np.random.rand(5)

x=regularization_path(adj_mat,house_data,prices)

n=np.shape(adj_mat)[0]
A=np.hstack((house_data,np.ones((n,1))))
errors=np.zeros(n)
for i in range(n):
    errors[i]=np.abs(A[i]@x[i]-price[i])/prices[i]
