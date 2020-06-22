from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats
from scipy.stats import mode
import currents_storage as cs
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics


np.random.seed(125)


run_time = 120*ms
defaultclock.dt = .02*ms
test_I_arr_file_loaded = np.load('comparison_files/test_I_arr_file.npz')
I = np.multiply(np.power(10, 9), np.array(test_I_arr_file_loaded['I_arr']))
# print("I is "+str(I[0][:6]))

N_AL = 800
in_AL = .1
PAL = 0.5

# Need to create folders before running
tr_prefix = 'data_dev/td_data/base/'
te_prefix = 'data_dev/td_data/test/test_'
save_prefix = 'data_dev/td_data/'

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
n_waveforms = 10

t_array = []
# for j in range(n_waveforms):
#     # Make multiple waveforms from one stimulated current by sampling
#     # every 250 ms, creating waveform length of 150 ms
#     tmp = TimedArray(I[7], dt = dt*ms)
#     t_array.append(tmp)
np_arrays_of_currents = cs.create_TD_currents(run_time, N_AL, I, defaultclock)
for j in range(n_waveforms):
    t_array.append(TimedArray(np_arrays_of_currents[j], dt = defaultclock.dt))
print("array of current_storage_function: "+str(np_arrays_of_currents[0]))
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
num_train = 1 #if 1, no noise introduced to computation
num_test = 1 #if 1, no noise introduced to computation


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
#
# Loading data
loading_train_prefix = "train/run_3_25_2020_odor_0_many_sine_A_perturbations/"
loading_test_prefix = "test/run_3_25_2020_odor_0_many_sine_A_perturbations/test_"
spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr = anal.load_data(loading_train_prefix, num_runs = num_odors*num_train)
spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr = anal.load_data(loading_test_prefix, num_runs = num_odors*num_test)

print("Check trace shape here")
print(np.shape(trace_V_arr))


# Plotting just for odor 1
# Multiplying by 1000 for units to be in ms
time_array = np.multiply(trace_t_arr[0],1000)
print("CHeck time_array shape here")
print(time_array)
# print("Seeing currents of: "+str(I_arr[0]))
# matplotlib.pyplot.figure()
# matplotlib.pyplot.imshow(I_arr[0])
# matplotlib.pyplot.show()
# print("Shape of currents: "+str(np.shape(I_arr)))
# print("Currents observed: "+str(np.sum(I_arr[0][:][:])))
# print("Currents observed: "+str(np.sum(I_arr[0][:][:])))

for odor_index in range(num_odors):
    fig, axs = plt.subplots(2)
    fig.suptitle('Voltage and Currents of All Neurons')
    for i in range(N_AL):
        axs[0].plot(time_array[500:],np.multiply(1000,trace_V_arr[odor_index][i])[500:])
        axs[0].set(ylabel="V (mV)")
        axs[0].set(xlabel = 'time (ms)')
        axs[1].plot(time_array[500:],I_arr[odor_index][i,500:])
        axs[1].set(ylabel="I (Amps)")
        axs[1].set(xlabel = 'time (ms)')
    plt.savefig('TD_Plots_Low_Internal_Driving_Voltage/fig_'+str(odor_index)+'.png')
#
# -------------------------------------------------------------------
# Preparation for SVM

print("trace_V_arr has dimension "+str(np.shape(trace_V_arr)))
print("test_V_arr has dimension "+str(np.shape(test_V_arr)))
# print("trace_V_arr has dimension "+str(np.shape(trace_V_arr[:][:][1000:])))
print("SHape is "+str(np.shape(label_arr)))
# Shorten times of loaded arrays
skip_to_time = 1000
reduced_time_list_of_array_trace_V_arr = []
for anArray in  trace_V_arr[:2]:
    reduced_time_list_of_array_trace_V_arr.append(anArray[:,skip_to_time:])
reduced_time_list_of_label_arr = []
for anArray in  label_arr[:2]:
    reduced_time_list_of_label_arr.append(anArray[skip_to_time:])
print(np.shape(reduced_time_list_of_array_trace_V_arr))


reduced_time_list_of_array_test_V_arr = []
for anArray in  test_V_arr[:2]:
    reduced_time_list_of_array_test_V_arr.append(anArray[:,skip_to_time:])
reduced_time_list_of_label_test_arr = []
for anArray in  label_test_arr[:2]:
    reduced_time_list_of_label_test_arr.append(anArray[skip_to_time:])

# -------------------------------------------------------------------

# PCA
pca = True  # do PCA on output or not

if pca:  # do PCA on the output
    pca_dim = 3
    pca_arr, PCA = anal.doPCA(reduced_time_list_of_array_trace_V_arr, k=pca_dim)

    X = np.hstack(pca_arr).T
else:
    X = np.hstack(reduced_time_list_of_array_trace_V_arr).T

# normalize
mini = np.min(X)
maxi = np.max(X)
X = anal.normalize(X, mini, maxi)

y = np.hstack(reduced_time_list_of_label_arr)

# train the SVM
print(np.shape(X))
print(np.shape(y))
clf = anal.learnSVM(X, y)

# --------------------------------------------------------------
# test SVM
if pca:
    test_data = anal.applyPCA(PCA, reduced_time_list_of_array_test_V_arr)
else:
    test_data = reduced_time_list_of_array_test_V_arr

test_data = anal.normalize(test_data, mini, maxi)

y_test = np.mean(reduced_time_list_of_label_test_arr, axis=1)

# check predictions on the test data
pred_arr = []
for i in range(len(test_data)):
    pred = clf.predict(test_data[i].T)
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

# ----------------------------------------------------------------------
# Plotting

# Plotting only works if pca_dim = 2 or 3
if pca and pca_dim == 2:
    # plot no training boundary
    title = 'PCA 2D'
    name = tr_prefix + 'PCA_2D.pdf'
    anal.plotPCA2D(pca_arr, title, name, num_train, skip=2)

    # plot the training boundary
    title = 'Training Boundary 2D'
    name = tr_prefix + 'boundary_AI.pdf'
    anal.plotSVM(clf, X, y, title, name)

    # plot the test data
    title = 'Testing Input with Noise ' + str(np.rint(100 * noise_test / sqrt(3))) + '%'
    name = tr_prefix + 'testing_AI.pdf'
    anal.plotSVM(clf, test_data, reduced_time_list_of_label_test_arr, title, name)

if pca and pca_dim == 3:
    title = 'PCA 3D'
    name = tr_prefix + 'PCA_3D.pdf'
    anal.plotPCA3D(pca_arr, N_AL, title, name, el=30, az=30, skip=1, start=0)