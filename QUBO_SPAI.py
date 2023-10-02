# Reference
# Computing a Sparse Approximate Inverse for 2D Finite Difference Poisson Problems on Quantum Annealing Machin
# Authors: Sanjay Suresh, Krishnan Suresh
# Email: ksuresh@wisc.edu

from dimod.reference.samplers import ExactSolver
from dwave.system import DWaveSampler, EmbeddingComposite

import numpy as np
import scipy
import pyamg
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import time
from pyqubo import Placeholder,Array
from tqdm import tqdm
#import dwave.inspector

def assignMaterial(materialScenario,gridX,gridY): 
    # We can assign different materials to different 'elements' in the grid
    # The number of nodes is (gridX)*(gridY)
    # The number of elements is (gridX-1)*(gridY-1)
    # Nodes are numbered first along Y and then along X
    # Elements are numbered first along Y and then along X
    
    # materialScenario=0: uniform material 0
    # materialScenario=1: material 0 on left half and material 1 on right half 
    # nodeMaterialSignature: the 4 materials surrounding the node defines its unique signature
    
    nNodes = gridX*gridY
    nElems = (gridX-1)*(gridY-1)
    materialIndex = nElems*[0]# default of material 0
    nodeMaterialSignature = nNodes*[0] 
    if (materialScenario == 1):
        for i in range(int(nElems/2),nElems):
            materialIndex[i] = 1
        for node in range(int(gridX/2)*gridY,int(gridX/2)*gridY+gridY):
            nodeMaterialSignature[node] = 1
        for node in range(int(gridX/2)*gridY+gridY, nNodes):
            nodeMaterialSignature[node] = 2
            
    return [materialIndex,nodeMaterialSignature]


def getAdjacentNodes(gridX,gridY,node):
    # gridX: The number of nodes along X 
    # gridY: The number of nodes along Y 
    # node: The node number; nodes are numbered first along Y and then along X
    # Returns the nodes that are adjacent to node, including itself  
   
    # The number of adjacent nodes can range from 4 to 9
    # We return a list of 9, where -1 denotes an invalid node
    # List order: self, top right, top middle, ... anti-clockwise
    # Convention: nodes are numbered first along Y and then along X
    xi = int(node/(gridY))
    yi = node - (gridY)*xi
    
    nodes = 9*[-1]
    if (xi == 0) and (yi == 0):#Lower left corner
        nodes = [node,node+gridY+1,node+1,-1,-1,-1,-1,-1,node+gridY] 
    elif  (xi == 0) and (yi == gridY-1): #Top left corner
        nodes = [node,-1,-1,-1,-1,-1,node-1,node+gridY-1,node+gridY] 
    elif (xi == gridX-1) and (yi == 0): #Lower right corner
        nodes = [node,-1,node+1,node-gridY+1,node-gridY,-1,-1,-1,-1]
    elif (xi == gridX-1) and (yi == gridY-1):  #top right corner
        nodes = [node,-1,-1,-1,node-gridY,node-gridY-1,node-1,-1,-1]  
    elif (xi == 0): # node on left edge but not on corner
        nodes = [node,node+gridY+1,node+1,-1,-1,-1,node-1,node+gridY-1,node+gridY] 
    elif (xi == gridX-1): # node on right edge but not on corner
        nodes = [node,-1,node+1,node-gridY+1,node-gridY,node-gridY-1,node-1,-1,-1] 
    elif (yi == 0): # node on bottom edge but not on corner
        nodes = [node,node+gridY+1,node+1,node-gridY+1,node-gridY,-1,-1,-1,node+gridY] 
    elif (yi == gridY-1): # node on top edge but not on corner
        nodes = [node,-1,-1,-1,node-gridY,node-gridY-1,node-1,node+gridY-1,node+gridY] 
    else: # interior node     
        nodes = [node,node+gridY+1,node+1,node-gridY+1,node-gridY,node-gridY-1,node-1,node+gridY-1,node+gridY] 
 
    return nodes

def findNodeMapping(gridX,gridY,nodeMaterialSignature):
    # two nodes are congruent if they result in the same column entries in K matrix
    nNodes = gridX*gridY
    nodeCongruency = nNodes*[0]# default: every node is congruent to node 0
    materialNeighborhoodUniqueNodes = []
    uniqueNodes = []
    for node in range(nNodes):
        neighboringNodes = getAdjacentNodes(gridX, gridY, node) 
        materialNeighborhoodOfNode = 9*[-1]
        for i in range(9):
            if (neighboringNodes[i] >= 0):
                materialNeighborhoodOfNode[i] = nodeMaterialSignature[neighboringNodes[i]]
        found = False
        for i in range(len(uniqueNodes)):
            otherNode = uniqueNodes[i]
            materialNeighborhoodOtherNode = materialNeighborhoodUniqueNodes[i]
            if (materialNeighborhoodOfNode == materialNeighborhoodOtherNode):
                nodeCongruency[node] = otherNode
                found = True
                break
        if (not found):
            uniqueNodes.append(node)
            materialNeighborhoodUniqueNodes.append(materialNeighborhoodOfNode)
            nodeCongruency[node] = node

    print('Num of unique nodes: ',len(uniqueNodes))
    return  nodeCongruency

def applyConductivityToK(K,gridX,gridY,nodeMaterialSignature,kMaterial):
    # Given a material Index, and the two conductivites, modify the K matrix
    for col in range(K.shape[0]): # also corresponds to a node in grid
        ns1 = nodeMaterialSignature[col]    
        nonzeros = K.indptr[col+1]- K.indptr[col]
        for i in range(nonzeros):
            row = K.indices[K.indptr[col]+i]# also corresponds to a node in grid
            ns2 = nodeMaterialSignature[row] 
            if (ns1 == 0) or (ns2 == 0):
                kEffective = kMaterial[0]
            elif (ns1 == 2) or (ns2 == 2):
                kEffective = kMaterial[1]
            else:
                kEffective = (kMaterial[0] + kMaterial[1])/2
            
            # below code is same as K[row,col] = kEffective*K[row,col] but much faster 
            K.data[K.indptr[col]+i] = kEffective*K.data[K.indptr[col]+i]

    return K
    
def createModelWithPlaceHolders():
    # When using PyQUBO, creating a model with place holers avoids repeated compiling
    maxSparsity = 5 # Don't modify; finite difference of Poisson problem
    q1= Array.create("q1",shape = maxSparsity,vartype = "BINARY")
    q2= Array.create("q2",shape = maxSparsity,vartype = "BINARY")
    
    c = maxSparsity*[0]#placeholders
    b = maxSparsity*[0]#placeholders
    A = maxSparsity*[0]#placeholders
    x = maxSparsity*[0]# symbolic via qubits  
    for i in range(maxSparsity):
        A[i] = maxSparsity*[0]#placeholders          
    L = Placeholder('L')
    for i in range(maxSparsity):
        c[i] = Placeholder('c[%d]' %i)
        b[i] = Placeholder('b[%d]' %i)
        for j in range(maxSparsity):
            A[i][j] = Placeholder("A[{i}][{j}]".format(i = i, j = j))
               
    for i in range(maxSparsity):
        x[i] = c[i] + L*(-2*q1[i] + q2[i]) 
       
    H = 0
    for  i in range(maxSparsity):
        Ax = 0
        for j in range(maxSparsity):
            Ax = Ax + A[i][j]*x[j]     
        H = H + x[i]*(0.5*Ax) - x[i]*b[i]
    model = H.compile()
    return model


def QUBOApproximateInverse(K,model, boxMaxIteration = 50, boxTolerance = 1e-6,boxL = 1): 
    nDimensions = K.shape[0]
    M = scipy.sparse.csc_matrix.copy(K)  
    dictionary = {}  
    for col in tqdm(range(nDimensions)):
        node = col
        if (exploitNodeCongruency):
            congruentNode = nodeCongruency[node]
            if (congruentNode == node):
                x = QUBOBoxSolve(K,col,M,model,boxMaxIteration = boxMaxIteration,boxTolerance = boxTolerance,boxL = boxL )
                dictionary[node] = x
            else:
                x = dictionary[congruentNode]
        else:  
            x = QUBOBoxSolve(K,col,M,model,boxMaxIteration = boxMaxIteration,boxTolerance = boxTolerance,boxL = boxL )
            
        nonzeros = K.indptr[col+1]- K.indptr[col]    
        for k in range(nonzeros):
            M.data[K.indptr[col]+k] = x[k]
    print("")       
    return M

def QUBOBoxSolve(K, col, M, model, boxL = 1, boxMaxIteration = 50, boxTolerance = 1e-6):
    global totalBoxIterations
    global totalBoxTime
    nonzeros = K.indptr[col+1]- K.indptr[col]
    maxSparsity = 5 # Don't modify; finite difference of Poisson problem
    if (nonzeros > maxSparsity):
        print("Sparsity is:", nonzeros)
        print("Exceeds allowed:", maxSparsity)
        return nonzeros*[0]
        
    qSol1 = maxSparsity*[0]#numerical
    qSol2 = maxSparsity*[0]#numerical
    center = maxSparsity*[0]#numerical
    b = maxSparsity*[0]#numerical
    A = maxSparsity*[0]#numerical
    for i in range(maxSparsity):
        A[i] = maxSparsity*[0]#numerical

    PEMin = 0

    for  i in range(nonzeros):
        row_i = K.indices[K.indptr[col]+i]
        if  (row_i < col): # exploiting symmetry
            continue
        for j in range(nonzeros):
            row_j = K.indices[K.indptr[col]+j]
            if  (row_j < col): # exploiting symmetry
                continue
            A[i][j] = K[row_i,row_j]  
        if (row_i == col):
            b[i] = 1
    modelDictionary = {}
    for  i in range(maxSparsity):
        modelDictionary['b[%d]' %i] = b[i]
        for j in range(maxSparsity):
            modelDictionary["A[{i}][{j}]".format(i = i, j = j)] = A[i][j]
            
    for iteration in range(boxMaxIteration):   
        modelDictionary['L'] =  boxL
        for  i in range(maxSparsity):
            modelDictionary['c[%d]' %i] = center[i]
        
        bqm = model.to_bqm(feed_dict = modelDictionary)
        tStart = time.time()
        if (samplingMethod == 0):
            results = sampler.sample(bqm)
        else:
            results = sampler.sample(bqm, num_reads = nSamples) 
            #dwave.inspector.show(results); input('')
        tEnd = time.time()
        totalBoxTime = totalBoxTime + (tEnd - tStart)
        sample = results.first.sample
        PE = results.first.energy 
        if (boxL < boxTolerance):
            break
        if (PE < PEMin): # Translation
            for i in range(maxSparsity):
                if (i < nonzeros):
                    row = K.indices[K.indptr[col]+i]
                    if (row<col):
                        center[i]= M[col,row]
                        continue
                qSol1[i]= sample["q1["+str(i)+"]"] 
                qSol2[i]= sample["q2["+str(i)+"]"]       
                center[i] = center[i] + boxL*(-2*qSol1[i] + qSol2[i])
            PEMin = PE             
        else:# Contraction only if we don't translate
            boxL = boxL/2
    if (boxL> boxTolerance):
        print("Box method did not converge to desired tolerance")
    #print("avg time:", tTotal/(iteration+1))
    #print("iterations:", iteration)
    
    totalBoxIterations = totalBoxIterations + iteration
    return center

def plotField(gridX,gridY):
    x = np.linspace(0., 1,num=gridX) 
    y = np.linspace(0., gridY/gridX,num=gridY) 
    X, Y = np.meshgrid(x, y) 
    U = np.transpose(np.reshape(u,(gridX, gridY)))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, U, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('field');
   
    
def CGcallBackFunction(u):
     nrmConvergence.append(np.linalg.norm(f-K*u)) 

if __name__ == "__main__":
    #######################################################
    # Default simulation parameters and options
    #######################################################
    
    # Specify the number of unknowns (interior grid nodes)
    # gridX: The number of nodes along X (>= 2) 
    # gridY: The number of nodes along Y (>= 2)
    gridX = 41
    gridY = 31
    gridX = gridX + (gridX % 2) - 1 # convenient to have odd number of nodes
    gridY = gridY + (gridY % 2) - 1 #  convenient to have odd number of nodes
    
    # We can assign different materials to different 'elements' in the grid
    # The number of elements is (gridX-1)*(gridY-1)
    # materialScenario = 0: uniform material 0
    # materialScenario = 1: material 0 in left half and material 1 in right half 
    materialScenario = 0
    kMaterial = [1,1] # the two conductivities
    
    # samplingMethod
    # 0: for exact (symbolic) solve of QUBO
    # 1: for DWAVE quantum annealing of QUBO with nSamples
    samplingMethod = 0 
    nSamples = 100  # relevant only when samplingMethod = 1
    
    # boxTolerance, boxMaxIteration 
    # boxTolerance: typical value of 1e-6
    # boxMaxIteration: typical value of 50
    # boxL: typical value of 1
    boxTolerance = 1e-6
    boxMaxIteration = 50
    boxL = 1
    
    # exploitNodeCongruency: if set to True exploits FD node congruency; reduces computation to constant time independent of grid Size
    exploitNodeCongruency = True
    
    # CGTol: conjugate gradient residual tolerance
    CGtol = 1e-10
  
    #######################################################
    # Modify the default values for each experiment
    #######################################################
    experiment = 1
    
    if (experiment == 1):
        gridX = 41
        gridY = 31
    elif (experiment == 2):
        gridX = 401
        gridY = 301
    elif (experiment == 3):
        gridX = 401
        gridY = 301
        materialScenario = 1 
        kMaterial = [1,100]
    elif (experiment == 4):
         gridX = 401
         gridY = 301
         boxTolerance = 1e-4
    elif (experiment == 5):
         gridX = 401
         gridY = 301
         boxL = 100
         
    #######################################################
    # Main code starts here
    #######################################################
    #%% Compute the K matrix
    K = pyamg.gallery.poisson((gridX,gridY), format='csc')  # 2D Poisson problem on grid
    print('#Nodes:',K.shape[0])
    [materialIndex,nodeMaterialSignature] = assignMaterial(materialScenario,gridX,gridY)
    st = time.time()
    K = applyConductivityToK(K,gridX,gridY,nodeMaterialSignature,kMaterial)
    f = ((kMaterial[0] + kMaterial[1])/2)*np.ones(K.shape[0])/min(gridX,gridY)  # scale to ensure u ~ O(1)
    et = time.time()
    print("Time to prepare K:",et-st)
    
    #print("Condition number of K:", np.linalg.cond(K.todense()))
    #%% CG without preconditioner
    nrmConvergence = []
    st = time.time()
    u, exit_code = cg(K, f, tol = CGtol,callback = CGcallBackFunction)
    et = time.time()
    print("CG Iters:",len(nrmConvergence))
    print("Time to perform CG:",et-st)
    nrmConvergenceCG = nrmConvergence.copy()

    #%% Precompute pyQUBO model with place holders
    st = time.time()
    model = createModelWithPlaceHolders()
    print("Time to create QUBO model:",et-st)
    et = time.time()
   
    st = time.time()
    nodeCongruency = findNodeMapping(gridX,gridY,nodeMaterialSignature)
    et = time.time()
    print("Time to find node congruency:",et-st)
      
    
    #%% Create the sampler 
    if (samplingMethod == 0):
        sampler = ExactSolver()
    else:
        sampler = EmbeddingComposite(DWaveSampler())
             
    #%% Compute SPAI M
    totalBoxIterations = 0
    totalBoxTime = 0
    st = time.time()
    M = QUBOApproximateInverse(K,model,boxTolerance=boxTolerance, boxMaxIteration = boxMaxIteration,boxL = boxL)
    et = time.time()
    print("Time to compute M:",et-st)
    print("totalBoxIterations:",totalBoxIterations)
    print("totalBoxTime:",totalBoxTime)
    
    #%% Q-PCG 
    nrmConvergence = []
    st = time.time()
    u, exit_code = cg(K, f, M = M,tol = CGtol,callback = CGcallBackFunction)
    et = time.time()
    print("Time to perform Q-PCG:",et-st)
    nrmConvergenceQPCG = nrmConvergence.copy()
    
    #%% Plots
    plt.close('all')
    plt.semilogy(range(len(nrmConvergenceQPCG)),nrmConvergenceQPCG,'r:')
    
    print("Q-PCG Iters:",len(nrmConvergenceQPCG))   
    plt.semilogy(range(len(nrmConvergenceCG)),nrmConvergenceCG,'b')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if (samplingMethod == 0):
        plt.legend(["Q-PCG", "CG"], loc ="upper right",fontsize="14")
    else:
        plt.legend(["Q-PCG", "CG"], loc ="upper right",fontsize="14")
        
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Residual error', fontsize=14)
    plt.grid(visible=True)
    plt.show()

    