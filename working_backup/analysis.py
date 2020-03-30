from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import numpy as np

from itertools import cycle

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# plt.style.use('ggplot')

def load_data(prefix, num_runs):
    spikes_t_arr = []
    spikes_i_arr = []
    I_arr = []
    trace_V_arr = []
    trace_t_arr = []
    labels_arr = []
    for i in range(num_runs):
        spikes_t_arr.append(np.load(prefix+'spikes_t_'+str(i)+'.npy'))
        spikes_i_arr.append(np.load(prefix+'spikes_i_'+str(i)+'.npy'))
        I_arr.append(np.load(prefix+'I_'+str(i)+'.npy'))

        trace_V_arr.append(np.load(prefix+'trace_V_'+str(i)+'.npy'))
        trace_t_arr.append(np.load(prefix+'trace_t_'+str(i)+'.npy'))
        #labels = np.load(prefix+'labels_'+str(i)+'.npy')
        #labels_arr.append(labels)
        labels_arr.append(np.load(prefix+'labels_'+str(i)+'.npy'))

    return spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, labels_arr

def load_data_limited(prefix, num_runs, **kwargs):

    start_run = kwargs.get('start_run',0)
    skip = kwargs.get('skip',1)
    start_skip = kwargs.get('start_skip',0)
    #end_skip = kwargs.get('end_skip',1)

    trace_V_arr = []
    labels_arr = []
    for i in range(start_run,start_run+num_runs):
        tmp = np.load(prefix+'trace_V_'+str(i)+'.npy')
        tmp = tmp[:,start_skip:]
        trace_V_arr.append(tmp)

        tmp = np.load(prefix+'labels_'+str(i)+'.npy')
        tmp = tmp[start_skip:]
        labels_arr.append(tmp)

    return trace_V_arr, labels_arr

def doPCA(trace_V_arr, k = 3):

    n, _, length = np.shape(trace_V_arr)
    print("DOING PCA HERE")
    print(np.shape(trace_V_arr))
    #concatenate data
    data = np.hstack(trace_V_arr)

    # svd decomposition and extract eigen-values/vectors
    pca = PCA(n_components=k)
    print(np.shape(data.T))
    pca.fit(data.T)

    # Save the pca data into each odor/conc
    Xk = pca.transform(data.T)

    pca_arr = []
    for i in range(n):
        pca_arr.append(Xk[length*i:length*(i+1)].T)

    return pca_arr, pca

def applyPCA(PCA, data):
    n, _, length = np.shape(data)
    data = np.hstack(data)

    Xk = PCA.transform(data.T)

    pca_arr = []
    for i in range(n):
        pca_arr.append(Xk[length*i:length*(i+1)].T)

    return pca_arr


def learnSVM(X, y, K = 'linear', v = True):
    print("Doing SVM fit Here: ")
    if K == 'linear':
        clf = LinearSVC(C = 0.4, max_iter=100000, verbose = v)
    else:
        clf = SVC(kernel=K, verbose = v, C = 0.6, gamma = 8.0)
    print(np.shape(X))
    print(np.shape(y))
    clf.fit(X, y)

    return clf

def normalize(X, minim, maxim):
    return (X-minim)/(maxim - minim)

def plotSVM(clf, X, Y, title, name):
    _, ax = plt.subplots()

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min()-.1, X[:, 0].max()+.1
    y_min, y_max = X[:, 1].min()-.1, X[:, 1].max()+.1
    xx2, yy2 = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))
    Z = clf.predict(np.c_[xx2.ravel(), yy2.ravel()])

    Z = Z.reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25, label = str(Y))
    plt.title(title, fontsize = 22)
    # plt.legend()

    ax.axis([x_min, x_max,y_min, y_max])

    plt.ylabel('PCA EV 2', fontsize = 16)
    plt.xlabel('PCA EV 1', fontsize = 16)

    # plt.xticks([])
    # plt.yticks([])

    plt.savefig(name, bbox_inches = 'tight')
    plt.show()

def plotPCA2D(pca_arr, title, name, num_trials, skip = 1):
    marker = cycle(['^','o','s','p'])
    cycol = cycle(['y', 'b', 'r', 'k', 'c'])
    m = next(marker)
    c = next(cycol)
    fig,ax = plt.subplots()
    for i in range(len(pca_arr)):
        d = pca_arr[i]
        m = next(marker)
        c = next(cycol)
        if i%num_trials != 0:
            ax.plot(d[0][::skip], d[1][::skip],
             		 '.'#,
              		 #marker = m,
            		 #color = c
                     )
        if i%num_trials == 0:
            #m = next(marker)
            ax.plot(d[0][::skip], d[1][::skip],
                     '.',
                     #marker = m,
                     #color = c,
                     alpha = 0.5,
                     label = 'base odor '+str((i+num_trials)/num_trials))
    ax.grid(False)
    plt.legend()
    plt.title(title, fontsize = 22)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.savefig(name, bbox_inches = 'tight')
    plt.show()

def getMIM(trace_V_arr):
    #Information component analysis
    # length = len(trace_V_arr[0][0])
    data = np.hstack(trace_V_arr).T
    N = np.shape(data)[1]

    bins = int(np.round(np.log2(N)+1)) #Sturges' formula
    matMI = np.zeros((N, N))

    for ix in np.arange(N):
        if ix%10 == 0: print(ix)
        for jx in np.arange(ix,N):
            matMI[ix,jx] = calc_MI(data[:,ix], data[:,jx], bins)
            #symmetric matrix
            matMI[jx, ix] = matMI[ix,jx]

    np.save('MIM',matMI)

def doInCA(MIM, data, length, num_odors, skip, k = 3):
    # k is the number of dimensions
    w,v = np.linalg.eig(MIM)

    vk = v[:, :k]
    # # Save the InCA data into each odor/conc
    Xk = vk.T.dot(data.T).T[::skip]
    InCAData = []
    # num_odors?
    for i in range(num_odors):
        InCAData.append(Xk[i*length:(i+1)*length])

    return InCAData

def plotInCA3D(InCAData, N, title, name, start = 400):

    # cycol = cycle(['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700'])
    # marker = cycle(['^','o','s','p'])


    fig = plt.figure(figsize = (10,7))
    ax = fig.gca(projection='3d')
    # c = next(cycol)
    # m = next(marker)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.grid(False)


    for j in range(len(InCAData)):
        ax.scatter(InCAData[j][start:,0], InCAData[j][start:,1], InCAData[j][start:,2],
                   s=10#,
                   # color = c,
                   # marker = m
                   )
        # c = next(cycol)
        # m = next(marker)

    plt.title(title, fontsize = 22)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(0, 0)
    # plt.tight_layout()
    plt.show()
    ax.figure.savefig(name, bbox_inches = 'tight')


def plotInCA2D(InCAData, N, title, name, start = 400,x_bounds = None, y_bounds = None):
    '''
    Currently, this creates a pretty bad image -- used 0, 2
    '''
    #marker = cycle(['^','o','s','p'])
    #ycol = cycle(['yellow', 'blue', 'red', 'cyan', 'm', 'g','lime','darkcyan','orange','midnightblue'])
    #m = next(marker)
    #c = next(cycol)


    fig,ax = plt.subplots(figsize = (10,7))
    # c = next(cycol)
    # m = next(marker)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.grid(False)


    for j in range(len(InCAData)):
        ax.scatter(InCAData[j][start:,0], InCAData[j][start:,2],
                   s=10,
                   #color = c,
                   #marker = m
                   alpha = 0.5,
                   label = 'base odor '+str(j)
                   )
        #c = next(cycol)
        # m = next(marker)

    if x_bounds is not None:
        ax.set_xlim(x_bounds)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)

    plt.legend()
    plt.title(title, fontsize = 22)
    # plt.tight_layout()
    plt.show()
    ax.figure.savefig(name, bbox_inches = 'tight')

def calc_MI(X, Y, bins):
    c_xy = np.histogram2d(X, Y, bins)[0]
    MI = mutual_info_score(None, None, contingency=c_xy)
    return MI

def plotPCA3D(PCAdata, N, title, name, el = 30, az = 30, skip = 1, start = 0):

    cycol = cycle(['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700'])
    marker = cycle(['^','o','s','p'])


    fig = plt.figure(figsize = (10,7))
    ax = fig.gca(projection='3d')
    c = next(cycol)
    m = next(marker)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.grid(False)

    for j in range(len(PCAdata)):
        ax.scatter(PCAdata[j][0, start:][::skip],
                   PCAdata[j][1, start:][::skip],
                   PCAdata[j][2, start:][::skip],
                   s=10,
                   color = c,
                   marker = m)
        c = next(cycol)
        m = next(marker)

    plt.title(title, fontsize = 22)
    ax.view_init(elev = el, azim = az)
    ax.figure.savefig(name, bbox_inches = 'tight')
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix,

    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def getMIM(prefix, trace_V_arr):
    #Information component analysis
    # length = len(trace_V_arr[0][0])
    data = np.hstack(trace_V_arr).T

    n = np.shape(data)[0]
    N = np.shape(data)[1]

    bins = int(np.round(np.log2(n)+1)) #Sturges' formula

    matMI = np.zeros((N, N))

    for ix in np.arange(N):
        if ix%100 == 0: print(str(ix/100.) +'%')
        for jx in np.arange(ix,N):
            matMI[ix,jx] = calc_MI(data[:,ix], data[:,jx], bins)
            #symmetric matrix
            matMI[jx, ix] = matMI[ix,jx]

    np.save(prefix+'MIM', matMI)

def doInCA(MIM, trace_V_arr, skip, k = 3):

    data = np.hstack(trace_V_arr).T
    n, num_neurons, length = np.shape(trace_V_arr)

    w,v = np.linalg.eig(MIM)

    vk = v[:, :k]

    # # Save the InCA data into each odor/conc
    Xk = vk.T.dot(data.T).T[::skip]

    inca_arr = []
    for i in range(n):
        inca_arr.append(Xk[length*i:length*(i+1)].T)

    return inca_arr


def calc_MI(X, Y, bins):
    c_xy = np.histogram2d(X, Y, bins)[0]
    MI = mutual_info_score(None, None, contingency=c_xy)
    return MI
