from __future__ import division

from brian2 import *

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal
from scipy import stats
import currents_storage as cs

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn import metrics

np.random.seed(125)

run_time = 120*ms
defaultclock.dt = .02*ms
test_I_arr_file_loaded = np.load('comparison_files/test_I_arr_file.npz')
I = np.multiply(np.power(10, 9), np.array(test_I_arr_file_loaded['I_arr']))
print("I is "+str(I[0][:6]))

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
n_waveforms = 10 # number of base odors to make
num_mixtures = 10 # number of inbetween mixtures to make given one pair of base odors

t_array = []
# for j in range(n_waveforms):
#     # Make multiple waveforms from one stimulated current by sampling
#     # every 250 ms, creating waveform length of 150 ms
#     tmp = TimedArray(I[7], dt = dt*ms)
#     t_array.append

# Creating base odors
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
noise_amp = 0.0 #0.1*np.sqrt(3) #max noise as a fraction of inp
noise_test = 0.0 #0.1*np.sqrt(3)

# Different spatial injections
sp_odors = 1
num_odors = int(sp_odors*n_waveforms)
num_train = 1
num_test = 1

I_arr = [] # Note: This goes completely unused
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

# --------------------------------------------------------------
# Create list of mixture pairs
mixture_pairs = []
for i in range(n_waveforms):
    for j in range(n_waveforms):
        if i>j:
            mixture_pairs.append((i,j))
print("Mixture pairs:")
print(mixture_pairs)

#---------------------------------------------------------------
# run the simulation and save to disk
# ex.createDataTD(run_params_train, I_arr, states, net, t_array=t_array)

# load in the data from disk
spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr, label_arr \
    = anal.load_data(tr_prefix, num_runs=n_waveforms)
X = np.hstack(trace_V_arr).T

#normalize training (base odor) data
mini = np.min(X) # lowest voltage of any neuron's voltages at any time for any odor
maxi = np.max(X) # highest voltage of any neuron's voltages at any time for any odor
normalized_Training_Voltages = anal.normalize(X, mini, maxi) # 0 if minimum value, 1 if maximum value.

point_Labels = np.hstack(label_arr)

# train the SVM
# clf = anal.learnSVM(normalized_Training_Voltages, point_Labels)

for mixture_pair in mixture_pairs[:2]:
    print("Testing this file.")
    print("Larger loop at " +str(mixture_pair))
    current_index_B = mixture_pair[0]
    current_index_A = mixture_pair[1]
    save_numbers = "(" + str(current_index_B) + "," + str(current_index_A) + ")_"
    # ex.mixtures2_TD(run_params_test, I_arr, states, net, mixture_pairs,
    #                 np_arrays_of_currents, defaultclock, current_index_B, current_index_A,
    #                 t_array=t_array, num_mixtures=num_mixtures)

for mixture_pair in mixture_pairs[:2]:
    print("Larger loop at " +str(mixture_pair))
    current_index_B = mixture_pair[0]
    current_index_A = mixture_pair[1]
    save_numbers = "(" + str(current_index_B) + "," + str(current_index_A) + ")_"

    spikes_t_test_arr, spikes_i_test_arr, I_test_arr, test_V_arr, test_t_arr, label_test_arr \
        = anal.load_data(te_prefix + save_numbers, num_runs=num_mixtures)
    print("Training current")
    print(I_arr)
    print("Testing current")
    print(I_test_arr)

    test_data = test_V_arr
    test_data = anal.normalize(test_data, mini, maxi)

    y_test = np.mean(label_test_arr, axis=1)

    pred_arr = []
    A_arr = []
    for i in range(len(test_data)):
        pred = clf.predict(test_data[i].T)
        total_pred = stats.mode(pred)[0]
        pred_arr.append(total_pred)
        A_arr.append(np.histogram(pred, bins=np.arange(n_waveforms + 1))[0])
    print("Shape of A_arr: "+str(np.shape(A_arr)))
    A_arr = np.array(A_arr) / np.sum(A_arr[0])  # alpha array

    np.savetxt(tr_prefix + 'alpha_histogram.txt', A_arr, fmt='%1.3f')
    expected = y_test
    predicted = np.array(pred_arr)
    print("Shape of A_arr:")
    print(np.shape(A_arr))
    odorAProportion = A_arr[:, current_index_A]
    odorBProportion = A_arr[:, current_index_B]
    odorOtherProportion = 1 - odorAProportion - odorBProportion


    def fsigmoid(x, a, b):
        return 1.0 / (1.0 + np.exp(-a * (x - b)))


    A = np.arange(num_mixtures) / num_mixtures

    # popt_x, pcov_x = curve_fit(fsigmoid, A, x)
    # popt_y, pcov_y = curve_fit(fsigmoid, A, y)

    # print(popt_x)
    # print(popt_y)
    # ----------------------------------------------------
    # Plotting
    plt.figure()
    plt.plot(A, odorAProportion, 'r.', label=r'$P(I_'+str(current_index_A)+')$')
    # plt.plot(A, fsigmoid(A, *popt_y), 'r-', linewidth=2)
    plt.plot(A, odorBProportion, 'b.', label=r'$P(I_'+str(current_index_B)+')$')
    # plt.plot(A, fsigmoid(A, *popt_x), 'b-', linewidth=2)
    plt.plot(A, odorOtherProportion, 'm.', label=r'1-$P(I_'+str(current_index_A)+')$-$P(I_'+str(current_index_B)+')$')
    plt.title('Classification of the Mixture of 2 Odors', fontsize=20)
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'Classification Probability of $I_'+str(current_index_A)+'$ and $I_'+str(current_index_B)+'$', fontsize=16)
    plt.legend()

    plt.savefig("mixtures_"+save_numbers+".pdf", bbox='tight')

plt.show()

