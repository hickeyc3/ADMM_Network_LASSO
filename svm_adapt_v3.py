import cvxpy as cp
import numpy as np
#import multiprocessing
from multiprocessing import Pool
from functools import partial
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv
import warnings
from PIL import Image
import math
#max allowable number of processes to be spawned
procs=8

#uncomment to visualize graph
import matplotlib.pyplot as plt
import networkx as nx

import os
mypath = '/home/yan/Documents/git/cs520_network_lasso/NetworkLasso-master/svm_adapt'
os.chdir(mypath)

warnings.filterwarnings("ignore")
nodes = 100
samepart = 0.15
diffpart = 0.07
partitions = 5
sizeOptVar = 11
trainSetSize = 8
numtests = trainSetSize
testSetSize = 4
c = 0.75
rho=.0001
#useConvex = 1
numiters = 10
np.random.seed(2)

def generate_graph_svm(nodes,samepart,diffpart,partitions,sizeOptVar,trainSetSize,testSetSize):
    adj_mat=np.zeros((nodes,nodes))
    sizepart = nodes/partitions
    edge_pairs=[]
    correctedges = 0
    for i in range(nodes):
        for j in range(nodes):
            if (i<j):
                if (round(i/sizepart) == round(j/sizepart)):
                    if(np.random.random() >= 1-samepart):
                        adj_mat[i][j]=1
                        edge_pairs.append((i,j))
                        correctedges = correctedges +1
                else:
                    if(np.random.random() >= 1-diffpart):
                        adj_mat[i][j]=1
                        edge_pairs.append((i,j))


    #generate a,w,v for each node
    a_true = np.random.randn(sizeOptVar, partitions)#the true a parameter vector for every partition
    v = np.random.randn(trainSetSize,nodes)#noise for each training sample each node
    vtest = np.random.randn(testSetSize,nodes) #noise for each test sample each node
    trainingSet = np.random.randn(trainSetSize*(sizeOptVar+1), nodes) #First all the x_train, then all the y_train below it
    for i in range(trainSetSize):
        trainingSet[(i+1)*sizeOptVar - 1, :] = 1 #Constant offset
    for i in range(nodes):
        a_part = a_true[:,math.floor(i/sizepart)]#find the true partition w for each node
        for j in range(trainSetSize):
            trainingSet[trainSetSize*sizeOptVar+j,i] = np.sign([np.dot(a_part.transpose(), trainingSet[j*sizeOptVar:(j+1)*sizeOptVar,i])+v[j,i]])

    (x_test,y_test) = (np.random.randn(testSetSize*sizeOptVar, nodes), np.zeros((testSetSize, nodes)))
    for i in range(testSetSize):
        x_test[(i+1)*sizeOptVar - 1, :] = 1 #Constant offset
    for i in range(nodes):
        a_part = a_true[:,math.floor(i/sizepart)]#find the true partition w for each node
        for j in range(testSetSize):
            y_test[j,i] = np.sign([np.dot(a_part.transpose(), x_test[j*sizeOptVar:(j+1)*sizeOptVar,i])+vtest[j,i]])
    sizeData = trainingSet.shape[0]#size of data available for one node, including all x and y
    edges = len(edge_pairs)
    print (nodes, edges, correctedges/float(edges), 1 - float(correctedges)/edges)
    return adj_mat,edge_pairs,trainingSet, x_test, y_test,sizeData

adj_mat,edge_pairs,trainingSet, x_test, y_test,sizeData = generate_graph_svm(nodes,samepart,diffpart,partitions,sizeOptVar,trainSetSize,testSetSize)

G = nx.Graph()
G.add_nodes_from(list(np.arange(nodes)))
G.add_edges_from(edge_pairs)
degrees = [val for (node, val) in G.degree()]
maxdeg = max(degrees)
edges = G.number_of_edges()
#nx.draw(G)
#plt.show()

data_mat = np.concatenate((trainingSet,np.tile([c, numtests,sizeData,rho,sizeOptVar], (nodes,1)).transpose()), axis=0).transpose()
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
            z : np.array,
            u : np.array,
            rho: float,
            l:float) -> np.array:

    tempData_persample = data_mat[i,:]
    rawData = tempData_persample[0:(sizeData)]
    x_train = rawData[0:trainSetSize*sizeOptVar]
    y_train = rawData[trainSetSize*sizeOptVar: trainSetSize*(sizeOptVar+1)]
    #Change for each experiment, this is the f_i(x_i) in the obj func
    #################################################
    #obj = cp.sum_squares(data_mat[i] @ x+x_offset-labels[i])+ mu*cp.sum_squares(x)
    a=cp.Variable(sizeOptVar)
    epsil = cp.Variable((trainSetSize,1))
    constraints = [epsil >= 0]
    f = c*cp.norm(epsil,1)
    for i in range(sizeOptVar - 1):
        f = f + 0.5*cp.square(a[i])
    for i in range(trainSetSize):
        temp = np.asmatrix(x_train[i*sizeOptVar:(i+1)*sizeOptVar])
        constraints = constraints + [y_train[i]*(temp@a) >= 1 - epsil[i]]
    ################################################

    g = 0
    rho = rho + math.sqrt(l)
    # for i in range(int(neighs.size/(2*inputs+1))):
    #     weight = neighs[i*(2*inputs+1)]
    #     if(weight != 0):
    #         u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
    #         z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
    #         g = g + rho/2*cp.square(cp.norm(a - z.reshape(-1,1) + u.reshape(-1,1)))

    #add terms to obj func for each neighbor
    for j in range(len(graph[i])):
        if graph[i][j] != 0:
            g += rho/2*cp.sum_squares(a-z[i][j]+u[i][j])


    objective = cp.Minimize(50*f + 50*g)
    p = cp.Problem(objective, constraints)
    result = p.solve()

    return a.value

#===============================================================================
#   update_z(edge,graph,x,u,l) - allows z update to run in parallel
#   edge     - edge the update is being run for
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   x       - current x
#   u       - current u
#================================================================================
def update_z(edge : tuple,
            graph : np.array,
            ak : np.array,
            u : np.array,
            rho: float,
            l : float) -> np.array:

    i=edge[0]
    j=edge[1]
    t=(l*graph[i][j])/(rho*LA.norm(ak[i]+u[i][j]-ak[j]-u[j][i]))
    theta=max(.5, 1-t)

    z_ij=theta*(ak[i]+u[i][j])+(1-theta)*(ak[j]+u[j][i])
    z_ji=(1-theta)*(ak[i]+u[i][j])+theta*(ak[j]+u[j][i])


    return z_ij,z_ji

#===============================================================================
#   update_u(edge,x,z,u_copy) - allows u update to run in parallel
#   edge    - edge the update is being run for
#   z       - current z
#   x       - current x
#   u_copy   - shallow copy of current u to get current u[i][j] value
#================================================================================
def update_u(edge : tuple,
            ak : np.array,
            z : np.array,
            u_copy : np.array) -> np.array:

    i=edge[0]
    j=edge[1]

    return u_copy[i][j]+(ak[i]-z[i][j])

#===============================================================================
#   admm (graph,data_mat,labels,l) - run convex fomrulation for each node i independently
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - train_data constants matrix for regresssion problem (swap for your experiment)
#   labels - labels of each house, the actual data point values for each experiment (swap for you experiment)
#   l - lambda, regularization term updated at each path
#================================================================================
def admm(graph : np.array,
        edge_pairs : list,
        data_mat : np.array,
        rho: float,
        l : float,
        z_k:np.array,
        u_k:np.array) -> np.array:


    #seems to diverge if any smaller?
    epsilon=10**(-2)


    n=np.shape(graph)[0]
    xupdate_nodes=np.linspace(0,n-1,n,dtype='int')

    #A=np.hstack((data_mat,np.ones((n,1))))


    #store space for z_i,j and u_i,j for the j neighbors of node i
    #z_k=np.zeros((n,n,len(data_mat[0])+1))
    #u_k=np.zeros((n,n,len(data_mat[0])+1))
    #z_k=np.zeros((n,n,sizeOptVar))
    #u_k=np.zeros((n,n,sizeOptVar))

    r=np.zeros(n*trainSetSize)
    v_acc=np.zeros(n*trainSetSize)
    old_r=np.ones(n*trainSetSize)
    iters = 0
    #while LA.norm(old_r-r)>epsilon:
    pool = Pool(processes = min(max(nodes, edges), procs))
    while iters < numiters:

        #do x-update for each node, so this is n processes in parallel

        update_instance = partial(update_x, graph=graph,data_mat=data_mat,z=z_k,u=u_k,rho = rho,l=l)

        #x_update_pool = multiprocessing.Pool(processes=procs)

        outputs = pool.map(update_instance, xupdate_nodes)

        x_k=np.array(outputs)


        #now that we have our new iterate x values for all the nodes, we update z and u

        z_update_instance=partial(update_z, graph=graph,ak=x_k,u=u_k,rho = rho,l=l)

        #z_update_pool = multiprocessing.Pool(processes=procs)

        z_outputs = pool.map(z_update_instance, edge_pairs)


        z=np.array(z_outputs)

        #unpack pooled z update
        for e in range(len(edge_pairs)):
            z_k[edge_pairs[e][0]][edge_pairs[e][1]]=z[e][0]
            z_k[edge_pairs[e][1]][edge_pairs[e][0]]=z[e][1]


        u_copy=np.copy(u_k)
        u_update_instance=partial(update_u,ak=x_k,z=z_k,u_copy=u_copy)

        #u_update_pool = multiprocessing.Pool(processes=procs)

        u_outputs = pool.map(u_update_instance, edge_pairs)

        u=np.array(u_outputs)

        #unpack pooled u update
        for e in range(len(edge_pairs)):
            u_k[edge_pairs[e][0]][edge_pairs[e][1]]=u[e][0]

        #compute primal residuals
        for i in range(n):
            tempData_persample = data_mat[i,:]
            rawData = tempData_persample[0:sizeData]
            x_train_pernode = rawData[0:trainSetSize*sizeOptVar]
            y_train_pernode = rawData[trainSetSize*sizeOptVar: trainSetSize*(sizeOptVar+1)]
            for j in range(trainSetSize):
                x_train_persample = np.asmatrix(x_train_pernode[j*sizeOptVar:(j+1)*sizeOptVar])
                r[i*trainSetSize+j]=abs(x_train_persample@x_k[i]*y_train_pernode[j]-1)
                v_acc[i*trainSetSize+j]=np.sign(x_train_persample@x_k[i])==y_train_pernode[j]
        RawPerf = [LA.norm(r),LA.norm(r-old_r),sum(v_acc)/(n*trainSetSize)]
        Output = ['{:.8f}'.format(elem) for elem in RawPerf]
		#print (Output)
		old_r=np.copy(r)
		iters = iters + 1
    pool.close()
    pool.join()
    return x_k,z_k,u_k

#===============================================================================
#  regularization_path (graph,data_mat,labels) - runs ADMM with increasing lambda values
#   graph - ajacency matrix form of graph, need the whole thing in this version
#   data_mat - train_data constants matrix for regresssion problem (swap for your experiment)
#   labels - labels of each house, the actual data point values for each experiment (swap for you experiment)
#================================================================================
def regularization_path( graph : np.array,
                        edge_pairs : list,
                        data_mat : np.array,
                        rho: float,
                        start_l:float,
                        step_l:float,
                        n_l:int,
                        isMul: int) -> np.array:


    n=np.shape(graph)[0]
    z_k=np.zeros((n,n,sizeOptVar))
    u_k=np.zeros((n,n,sizeOptVar))
    mat_trainPerf = np.zeros((n_l,2))
    mat_testPerf = np.zeros((n_l,2))

    #initial lambda
    #l=start_l
    l=0

    #while LA.norm(old_x-x) > 0:
    for k in range(n_l):
        train_r = np.zeros(n*trainSetSize)
        test_r = np.zeros(n*testSetSize)
        train_acc = np.zeros(n*trainSetSize)
        test_acc = np.zeros(n*testSetSize)
        #x_old=np.copy(x)
        x,z_k,u_k=admm(graph,edge_pairs,data_mat,rho,l,z_k,u_k)

        #compute primal residuals
        for i in range(n):
            tempData_persample = data_mat[i,:]
            rawData = tempData_persample[0:sizeData]
            x_train_pernode = rawData[0:trainSetSize*sizeOptVar]
            y_train_pernode = rawData[trainSetSize*sizeOptVar: trainSetSize*(sizeOptVar+1)]
            for j in range(trainSetSize):
                x_train_persample = np.asmatrix(x_train_pernode[j*sizeOptVar:(j+1)*sizeOptVar])
                train_r[i*trainSetSize+j]=abs(x_train_persample@x[i]*y_train_pernode[j]-1)
                train_acc[i*trainSetSize+j]=np.sign(x_train_persample@x[i])==y_train_pernode[j]

            x_test_pernode = x_test[:,i]
            y_test_pernode = y_test[:,i]
            for j in range(testSetSize):
                x_test_persample = np.asmatrix(x_test_pernode[j*sizeOptVar:(j+1)*sizeOptVar])
                test_r[i*testSetSize+j]=abs(x_test_persample@x[i]*y_test_pernode[j]-1)
                test_acc[i*testSetSize+j]=np.sign(x_test_persample@x[i])==y_test_pernode[j]
        trainPerf = [LA.norm(train_r),sum(train_acc)/(n*trainSetSize)]
        trainPerf_formated = ['{:.4f}'.format(elem) for elem in trainPerf]
        testPerf = [LA.norm(test_r),sum(test_acc)/(n*testSetSize)]
        testPerf_formated = ['{:.4f}'.format(elem) for elem in testPerf]
        mat_trainPerf[k]=trainPerf
        mat_testPerf[k] = testPerf

        print("l = ",l)
        print("train perf:", trainPerf_formated)
        print("test perf:", testPerf_formated)
        print("------------------")
        #print("sum of r = ",sum(r))
    #    print("error = ",errors[k])
        #print(LA.norm(x_old-x,1))
        if(l==0):
            l=start_l
        else:
            if(isMul):
                l=l*step_l
            else:
                l = l+step_l

        # if k > 1 and errors[k]>errors[k-1]:
        #     return x_old,errors

    return x,mat_trainPerf,mat_testPerf

#===============================================================================
#  global_average(labels) : baseline method to compute global average of house labels
#================================================================================
def global_average(data_mat):
        #compute primal residuals
        n = data_mat.shape[0]
        train_r = np.zeros(n)
        test_r = np.zeros(n)
        train_acc = np.zeros(n)
        test_acc = np.zeros(n)
        for i in range(n):
            tempData_persample = data_mat[i,:]
            rawData = tempData_persample[0:sizeData]
            #x_train_pernode = rawData[0:trainSetSize*sizeOptVar]
            y_train_pernode = rawData[trainSetSize*sizeOptVar: trainSetSize*(sizeOptVar+1)]
            avg_y = np.mean(y_train_pernode)
            train_r[i]=np.sum(abs(y_train_pernode*avg_y-1))
            train_acc[i]=np.sum(np.sign(y_train_pernode)==avg_y)


            y_test_pernode = y_test[:,i]
            test_r[i]=np.sum(abs(y_test_pernode*avg_y-1))
            test_acc[i]=np.sum(np.sign(y_test_pernode)==avg_y)

        trainPerf = [LA.norm(train_r),sum(train_acc)/(n*trainSetSize)]
        trainPerf_formated = ['{:.4f}'.format(elem) for elem in trainPerf]
        testPerf = [LA.norm(test_r),sum(test_acc)/(n*testSetSize)]
        testPerf_formated = ['{:.4f}'.format(elem) for elem in testPerf]
        return trainPerf,testPerf


# isMul = 0# isMul =1 update l by multiplication, otherwise by addition
# start_l = 0.2
# step_l = .2
# n_l = 5

isMul = 1# isMul =1 update l by multiplication, otherwise by addition
start_l = 0.001
step_l = 10
n_l = 6

x_reg,reg_trainPerf,reg_testPerf=regularization_path(adj_mat,edge_pairs,data_mat,rho,start_l,step_l,n_l,isMul)

np.save(f'reg_low_trainPerf_sizeOptVar_{sizeOptVar}_testSize_{testSetSize}_trainSize_{trainSetSize}', reg_trainPerf)
np.save(f'reg_low_testPerf_sizeOptVar_{sizeOptVar}_testSize_{testSetSize}_trainSize_{trainSetSize}', reg_testPerf)

avg_trainPerf,avg_testPerf=global_average(data_mat)
np.save(f'avg_trainPerf_sizeOptVar_{sizeOptVar}_testSize_{testSetSize}_trainSize_{trainSetSize}', avg_trainPerf)
np.save(f'avg_testPerf_sizeOptVar_{sizeOptVar}_testSize_{testSetSize}_trainSize_{trainSetSize}', avg_testPerf)
