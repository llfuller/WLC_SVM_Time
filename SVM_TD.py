from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal

from scipy.stats import mode
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics

defaultclock.dt = .04*ms


N_AL = 1000
in_AL = .1
PAL = 0.5

# Need to create folders before running
tr_prefix = 'data_dev/td_data/base/'
te_prefix = 'data_dev/td_data/test/test_'
save_prefix = 'data_dev/td_data/'

# Data Path for pre-designed currents
data_path = 'source_data/designed_currents/'
data_file = 'current.dat'

# dt of the source file
dt = 0.02 # ms

# Loading Data file
data = np.loadtxt(data_path + data_file)

# Renormalizing 0 - 1
data = (data - np.min(data))/(np.max(data)-np.min(data))



#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V']
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)

# Make Timed Array
# Different time dependent waveforms for different odors
n_waveforms = 6

t_array = []
for j in range(n_waveforms):
    # Make multiple waveforms from one stimulated current by sampling
    # every 250 ms, creating waveform length of 150 ms
    tmp = TimedArray(data[int(j*250/dt):int((j*250 + 150)/dt)], dt = dt*ms)
    t_array.append(tmp)

# If using the function createData, must be named t_array
trace_current = StateMonitor(G_AL,['I_inj','I_noise', 'scale'],record = True)
net.add(trace_current)

# Can crease a random time delay between neurons so the current waveform is
# slightly shifted
#G_AL.td = np.random.uniform(low = 0, high = 30, size =N_AL)
G_AL.td = 0
net.store()

# current settings
inp = 1.0
noise_amp = 0.1*np.sqrt(3) #max noise as a fraction of inp
noise_test = 0.1*np.sqrt(3)

# Different spatial injections
sp_odors = 1
num_odors = int(sp_odors*n_waveforms)
num_train = 1
num_test = 1

run_time = 100*ms

I_arr = []
#create the base odors
for i in range(sp_odors):
    I = ex.get_rand_I(N_AL, p =1./3., I_inp = inp)
    I_arr.append(I)


run_params_train = dict(num_odors = num_odors,
                        n_waveforms = n_waveforms,
                        sp_odors  = sp_odors,
                        num_trials = num_train,
                        prefix = tr_prefix,
                        inp = inp*nA,
                        noise_amp = noise_amp,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = True)


states = dict(  G_AL = G_AL,
                S_AL = S_AL,
                trace_AL = trace_AL,
                spikes_AL = spikes_AL,
                trace_current = trace_current)


run_params_test = dict( num_odors = num_odors,
                        n_waveforms = n_waveforms,
                        sp_odors = sp_odors,
                        num_trials = num_test,
                        prefix = te_prefix,
                        inp = inp*nA,
                        noise_amp = noise_test,
                        run_time = run_time,
                        N_AL = N_AL,
                        train = False)

# Run to create training and test data
ex.createDataTD(run_params_train, I_arr, states, net, t_array=t_array)
ex.createDataTD(run_params_test, I_arr, states, net, t_array=t_array)


spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(tr_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(te_prefix, num_runs = num_odors*num_test)

pca = False
if pca:
    pca_dim = 2
    pca_arr, PCA = anal.doPCA(trace_V_arr, k = pca_dim)

    X = np.hstack(pca_arr).T
else:
    X = np.hstack(trace_V_arr).T

mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)

y = np.hstack(label_arr)

trace_V_arr = None

# save min and max values for normalization
#mini = np.append(mini,maxi)
np.save(save_prefix + 'mini_maxi_%d.npy'%n_waveforms,[mini,maxi])

clf = anal.learnSVM(X, y)


pickle_file = open(save_prefix+'trained_svm_%d.pickle'%n_waveforms,'wb')
pickle.dump(clf,pickle_file)
pickle_file.close()

#test_data = anal.applyPCA(PCA, test_V_arr)
test_data = test_V_arr
test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(label_test_arr, axis = 1)


pred_arr = []
for i in range(len(test_data)):
    pred = clf.predict(test_data[i].T)
    # mode may end up being more correct
    total_pred = np.rint(mode(pred)[0][0])
    #total_pred = np.rint(np.mean(pred))
    print('total pred: ', total_pred)
    print('True: ' + str(y_test[i]), 'pred: ' + str(int(total_pred)))
    pred_arr.append(total_pred)

expected = y_test
pred_arr = np.array(pred_arr)
#pred_arr = np.reshape(pred_arr, (pred_arr.shape[0],pred_arr.shape[1]))

#print("Classification report for classifier %s:\n%s\n"
#      % (clf, metrics.classification_report(expected, predicted)))

#cm = metrics.confusion_matrix(expected, predicted)
#print("Confusion matrix:\n%s" % cm)

# anal.plot_confusion_matrix(cm)


print("Accuracy={}".format(metrics.accuracy_score(expected, pred_arr)))

# code to append file
f = open(save_prefix + 'acc','ab')
np.savetxt(f,([n_waveforms,metrics.accuracy_score(expected, pred_arr)],),fmt='%d %.3f')
f.close()

# title = 'Arbitrary Input Training'
# name = 'training.pdf'
# anal.plotPCA2D(pca_arr, title, name, num_train, skip = 2)
# title = 'Arbitrary Input Training Boundary'
# name = 'boundary_AI.pdf'
# anal.plotSVM(clf, X, y, title, name)
# title = 'Testing Arbitrary Input with Noise ' + str(np.rint(100*noise_test/sqrt(3)))+'%'
# name = 'testing_AI.pdf'
# anal.plotSVM(clf, test_data, label_test_arr, title, name)
