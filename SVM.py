from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats

import pickle
import os.path

import matplotlib.pyplot as plt
from sklearn import metrics

defaultclock.dt = .05*ms

#np.random.seed(22)

N_AL = 1000
in_AL = .1
PAL = 0.5

tr_prefix = 'data_dev/'
# This needs to be updated
te_prefix = 'data_dev/eta10/test_'

# If previous connections exist, use them
S_AL_conn = None
net_conn = os.path.exists(tr_prefix + 'S_AL.npz')
if net_conn:
    print('Using previous connections...')
    S_AL_conn = np.load(tr_prefix + 'S_AL.npz')

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V'],
               S_AL_conn = S_AL_conn
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)

# save network connections if none exist
if not net_conn:
    np.savez(tr_prefix+'S_AL.npz', i=S_AL.i, j = S_AL.j)

# setting initial conditions, these won't be used for DC current
G_AL.scale = 1
G_AL.I_noise = 0
G_AL.td = 0

trace_current = StateMonitor(G_AL,['I_inj'],record=True)
net.add(trace_current)
net.store()

inp = 0.15
p_inj = 1./3.
noise_amp = 0.1 #max noise percentage of inp
noise_test = 2.5*sqrt(3)

# If we need to load more files than odors
# ex: we want to run odors 5-10 for testing
num_odors = 10
num_files = 10

# number of presentations during training/testing cycle
num_train = 1
num_test = 1

run_time = 2500*ms


I_arr = []

# Use previous injected training currents if they exist
inj_curr = os.path.exists(tr_prefix + 'I_0.npy')

if inj_curr:
    print('Using previous odors...')
    for i in range(num_files):
        I_arr.append(np.load(tr_prefix + 'I_' + str(i) + '.npy'))
else:
    #create the base odors
    for i in range(num_odors):
        #I = ex.get_rand_I(N_AL, p = np.random.uniform(0.1, 0.5), I_inp = inp)*nA
        # Editing for Henry
        I = ex.get_rand_I(N_AL, p = 1./3., I_inp = inp)*nA
        I_arr.append(I)

# Since we doin't rescale currents when saving, this adjustment is necessary
I_arr = np.asarray(I_arr)/1e-9*nA

# example of running odors 5-10
#I_arr = I_arr[num_odors:]

run_params_train = dict(num_odors = num_odors,
                        num_trials = num_train,
                        prefix = tr_prefix,
                        inp = inp,
                        noise_amp = noise_amp,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = True)


states = dict(  G_AL = G_AL,
                S_AL = S_AL,
                trace_AL = trace_AL,
                spikes_AL = spikes_AL)


run_params_test = dict( num_odors = num_odors,
                        num_trials = num_test,
                        prefix = te_prefix,
                        inp = inp,
                        noise_amp = noise_test,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = False)

param_labels = ['N_AL', 'in_AL', 'PAL', 'inj', 'eta', 'num_train', 'num_test', 'runtime', 'p_inj']
param_values = [N_AL, in_AL, PAL, inp, noise_test, num_train, num_test, run_time, p_inj]


#ex.createData(run_params_train, I_arr, states, net,start_number=0)
#ex.createData(run_params_test, I_arr, states, net,start_number=0)

# This saves parameters used in the test folders
#np.savetxt(te_prefix + 'params.txt',list(zip(param_labels,param_values)),fmt='%s')


""" PCA/SVM Code

_, _, _, trace_V_arr, _, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
_, _, _, test_V_arr, _, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)

#uncomment these lines to do PCA on the output
# pca_dim = 20
# pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

# print(pca_arr[0])

# X = np.hstack(pca_arr).T
print(np.shape(label_arr))
#print(np.shape(trace_V_arr))

skip = 50

'''
for i in range(len(trace_V_arr[i])):
    trace_V_arr[i] = np.asarray(trace_V_arr[i][:,::skip])
    label_arr[i] = label_arr[i][::skip]
    test_V_arr[i] = np.asarray(test_V_arr[i][:,::skip])
    label_test_arr[i] = label_test_arr[::skip]
print(np.shape(label_arr))
'''
trace_V_arr = trace_V_arr[:3]
label_arr = label_arr[:3]
test_V_arr = test_V_arr[:3]
label_test_arr = label_test_arr[:3]

X = np.hstack(trace_V_arr).T


mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)
y = np.hstack(label_arr)
# barely enough memory to learn the SVM, needs to use swap partition to work
skip = 5
X = X[::skip,:]
y = y[::skip]


# Linear

print('Linear SVM...')
print(X.shape)
clf = anal.learnSVM(X, y)
print('Finished SVM')
# How do I save this??
# test_data = anal.applyPCA(PCA, test_V_arr)
test_V_arr = anal.normalize(test_V_arr, mini, maxi)

y_test = stats.mode(label_test_arr,axis = 1)[0]

pred_arr = []
for i in range(len(test_V_arr)):
    pred = clf.predict(test_V_arr[i].T)
    total_pred = stats.mode(pred)[0]
    print('True: ' + str(y_test[i]), 'pred: ' + str(int(total_pred)))
    pred_arr.append(total_pred)

pred_arr = np.array(pred_arr)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, pred_arr)))

cm = metrics.confusion_matrix(y_test, pred_arr)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(y_test, pred_arr)))
'''
print('Poly SVM...')
clf = anal.learnSVM(X, y, K='poly')

pred_arr = []
for i in range(len(test_V_arr)):
    pred = clf.predict(test_V_arr[i].T)
    total_pred = stats.mode(pred)[0]
    print('True: ' + str(y_test[i]), 'pred: ' + str(int(total_pred)))
    pred_arr.append(total_pred)

expected = y_test
predicted = np.array(pred_arr)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))

cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
'''
"""
# Only works if pca_dim = 2


# title = 'Arbitrary Input Training'
# name = 'training.pdf'
# anal.plotPCA2D(pca_arr, title, name, num_train, skip = 2)
# title = 'Arbitrary Input Training Boundary'
# name = 'boundary_AI.pdf'
# anal.plotSVM(clf, X, y, title, name)
# title = 'Testing Arbitrary Input with Noise ' + str(np.rint(100*noise_test/sqrt(3)))+'%'
# name = 'testing_AI.pdf'
# anal.plotSVM(clf, test_data, label_test_arr, title, name)
