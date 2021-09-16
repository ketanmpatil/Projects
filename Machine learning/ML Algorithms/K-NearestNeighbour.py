from matplotlib import lines, markers
import numpy as np
import matplotlib.pyplot as plt


def KNN(X,p,K=3): # Where X: Array of Nodes, K: No. of nodes, preferabely odd
    # X = {class:(clasnodes)}
    X_ = X
    classes = list(X_.keys()) # List of Classes
    
    distance_map={}  # distance_map = {dist : class used to cal distance}

    for i in classes:
        for j in X_[i]:
            dist = np.sqrt((j[1]-p[1])**2+(j[0]-p[0])**2) # Euclidean Distance Formula

            distance_map[dist] = i

    keys = list(distance_map.keys())
    keys = sorted(keys)[:K] # Get first K short distances from point

    
    close_classes = [distance_map[i] for i in keys] # Classes 

    belongs_to = {} # Stores count of classes that appeared in keys

    for i in classes:
        belongs_to[i] = close_classes.count(i)

    return list(belongs_to.keys())[list(belongs_to.values()).index(max(belongs_to.values()))] # Returns Class that is closes


# points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7),(0,0)],
#               1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
# p = (5,3)

# print(KNN(points,p))

# plt.plot(points[1],points[0],marker='o',linestyle=' ')
# plt.plot(p[0],p[1],marker='*')

# plt.show()
