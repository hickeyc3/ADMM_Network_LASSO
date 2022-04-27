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
# numbers for identification, but you guys won't need that
#from generate_graph import adj_mat, house_data, house_list, house_dict



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
#   neighbors - row of adjacency matrix for node i
#   c - house_data constants for regresssion problem (swap for your experiment)
#   l - lambda, regularization term
#================================================================================
def admm_convex(neighbors : np.array,
                c : np.array,
                l : float) -> np.array:

    theta=.3

    rho=.001

    x=cp.Variable(len(c)-1)
    x_offset=cp.Variable(1)


    #store space for z_i,j and u_i,j for the j neighbors of node i

    z=np.zeros((len(neighbors),len(c)))
    u=np.zeros((len(neighbors),len(c)))


    #this is the loop conditional, once we get the z-update working
    #while LA.norm(r)<=epsilon_pri or LA.norm(s)<=epsilon_dual
    for k in range(1):

        #Change for each experiment, this is the f_i(x_i) in the obj func for the
        # housing experiment
        #################################################
        mu=.5
        obj = cp.sum_squares(c[:-1] @ x+x_offset-c[-1])+ mu*cp.sum_squares(x)
        ################################################


        #add terms to obj func for each neighbor
        for j in range(len(neighbors)):
            if neighbors[j] != 0:
                obj += rho*.5*cp.sum_squares(cp.hstack((x,x_offset))-z[j]+u[j])


        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()

        x_k=np.hstack((x.value,x_offset.value))


        for j in range(len(neighbors)):
            if neighbors[j] != 0:
                #update z???? Right now it's just 0's
                u[j]+=(x_k-z[j])



    return x_k

#example on how to run for a node with random adjacency matrix and constants,
#try to have it in this form
np.random.seed(8)

adj_mat=np.random.rand(5,5)
adj_mat=adj_mat @ adj_mat.T
for i in range(5): adj_mat[i][i]=0.0
house_data=np.random.rand(5,4)

#admm_convex(adj_mat[0], house_data[0],0)
