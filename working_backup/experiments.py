import numpy as np
import os.path
from struct import unpack
import sys

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle

from brian2.only import *
from sklearn.utils import shuffle as rshuffle


def createData(run_params, I_arr, states, net, t_array=None,**kwargs):
    '''
    t_array: TimeArray object, must specify a run_regularly function in
    run script

    kwargs:
        start_number: An number used to shift the number in the save files
        start_skip: The number of data points to skip when saving
        shift_odor_num: shifts the odor number that is saved to labels file
    '''
    start_number = kwargs.get('start_number',0)
    start_skip = kwargs.get('start_skip',0)
    shift_odor_num = kwargs.get('shift_odor_num',0)

    num_odors = run_params['num_odors']
    num_trials = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['inp']
    noise_amp = run_params['noise_amp']
    run_time = run_params['run_time']
    N_AL = run_params['N_AL']
    train = run_params['train']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']


    n = 0 #Counting index

    for j in range(num_odors):
        for k in range(num_trials):
            noise = noise_amp#*np.random.randn()
            net.restore()

            if k == 0 and train:
                G_AL.I_inj = I_arr[j]
            else:
                G_AL.I_inj = I_arr[j]+noise*inp*(2*np.random.random(N_AL)-1)*nA

            net.run(run_time, report = 'text')
            np.save(prefix+'spikes_t_'+str(n+start_number) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n+start_number) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n+start_number), G_AL.I_inj)
            np.save(prefix+'trace_V_'+str(n+start_number), trace_AL.V[:,start_skip:])
            np.save(prefix+'trace_t_'+str(n+start_number), trace_AL.t[start_skip:])

            np.save(prefix+'labels_'+str(n+start_number), np.ones(len(trace_AL.t[start_skip:]))*(j+shift_odor_num))
            n = n+1

def createDataTD(run_params, I_arr, states, net, t_array = None, **kwargs):
    '''
    t_array: TimeArray object, must specify a run_regularly function in
    run script

    kwargs:
        start_number: An number used to shift the number in the save files
        start_skip: The number of data points to skip when saving
        shift_odor_num: shifts the odor number that is saved to labels file
    '''
    start_number = kwargs.get('start_number',0)
    start_skip = kwargs.get('start_skip',0)
    shift_odor_num = kwargs.get('shift_odor_num',0)

    num_odors = run_params['num_odors']
    sp_odors = run_params.get('sp_odors', num_odors)
    num_trials = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['inp']
    noise_amp = run_params['noise_amp']
    run_time = run_params['run_time']
    N_AL = run_params['N_AL']
    train = run_params['train']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']
    trace_current = states['trace_current']
    net.restore()
    m_arrays = t_array
    G_run = G_AL.run_regularly('''I_inj = t_array(t + td*ms, i)''')
    G_noise = G_AL.run_regularly('''I_noise = inp*noise*(2*rand() - 1)''')
    net.add(G_run,G_noise)

    # @network_operation(dt = 10*ms)
    # def f(t):
    #     print(np.shape(noise))
    # net.add(f)
    net.store()

    n = 0 #Counting index
    for j in range(num_odors):
        for k in range(num_trials):
            t_array = m_arrays[int(np.floor(j/sp_odors))]
            noise = noise_amp
            net.restore()
            G_AL.scale = I_arr[j%sp_odors] # 0 or 1

            if k == 0 and train:
                noise = 0

            # need to double check out current is saved!!! Right now, no numbers are skipped
            net.run(run_time, report = 'text')
            np.save(prefix+'spikes_t_'+str(n+start_number) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n+start_number) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n+start_number), trace_current.I_inj*trace_current.scale + trace_current.I_noise)
            np.save(prefix+'trace_V_'+str(n+start_number), trace_AL.V[:,start_skip:])
            np.save(prefix+'trace_t_'+str(n+start_number), trace_AL.t[start_skip:])

            np.save(prefix+'labels_'+str(n+start_number), np.ones(len(trace_AL.t[start_skip:]))*(j+shift_odor_num))
            #np.save(prefix + 'trace_scale_'+str(n),trace_current.scale[:,start:])
            n = n+1

def mixtures2(run_params, mix_arr, states, net, start = 100):
    assert len(mix_arr) ==2, 'mix_arr must have length 2'

    num_odors = run_params['num_odors']
    num_trials = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['inp']
    noise_amp = run_params['noise_amp']
    run_time = run_params['run_time']
    N_AL = run_params['N_AL']
    train = run_params['train']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']

    #I_A = AI_1 + (1-A)I_2
    eta_arr = np.linspace(0, 1, num_trials)

    n = 0 #Counting index
    if train:
        for i in range(num_odors):
            net.restore()
            G_AL.I_inj = mix_arr[i]
            net.run(run_time, report = 'text')
            np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n), G_AL.I_inj)
            np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
            np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])

            np.save(prefix+'labels_'+str(n), np.ones(len(trace_AL.t[start:]))*i)
            n = n+1
    else:
        for k in range(num_trials):
            noise = noise_amp #*np.random.randn()
            net.restore()

            I = (1-eta_arr[k])*mix_arr[0]+eta_arr[k]*mix_arr[1]
            G_AL.I_inj = I+noise*inp*(2*np.random.random(N_AL)-1)*nA

            net.run(run_time, report = 'text')

            np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n), G_AL.I_inj)
            np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
            np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])

            lab = 0
            np.save(prefix+'labels_'+str(n), np.ones(len(trace_AL.t[start:]))*lab)
            n = n+1


def mixtures2_TD(run_params, I_arr, states, net, mixture_pairs, np_arrays_of_currents, defaultclock,
                 current_index_B, current_index_A,
                 t_array = None, **kwargs):
    '''
    t_array: TimeArray object, must specify a run_regularly function in
    run script

    kwargs:
        start_number: An number used to shift the number in the save files
        start_skip: The number of data points to skip when saving
        shift_odor_num: shifts the odor number that is saved to labels file
        num_mixtures: number of mixtures between each pair of odors to use
    '''

    start_number = kwargs.get('start_number',0)
    start_skip = kwargs.get('start_skip',0)
    shift_odor_num = kwargs.get('shift_odor_num',0)
    num_mixtures = kwargs.get('num_mixtures', 2)

    num_odors = run_params['num_odors']
    sp_odors = run_params.get('sp_odors', num_odors)
    num_trials = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['inp']
    noise_amp = run_params['noise_amp']
    run_time = run_params['run_time']
    N_AL = run_params['N_AL']
    train = run_params['train']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']
    trace_current = states['trace_current']
    net.restore()
    m_arrays = t_array
    G_run = G_AL.run_regularly('''I_inj = t_array(t + td*ms, i)''')
    G_noise = G_AL.run_regularly('''I_noise = inp*noise*(2*rand() - 1)''')
    net.add(G_run,G_noise)
    net.store()

    #I_A = AI_1 + (1-A)I_2
    alpha_arr = np.linspace(0, 1, num_mixtures)
    n = 0 #Counting index
    current_B = np_arrays_of_currents[current_index_B]
    current_A = np_arrays_of_currents[current_index_A]
    print("Working on using current indices "+str(current_index_B)+","+str(current_index_A))
    for k in range(num_mixtures):
        net.restore()

        noise = noise_amp #*np.random.randn()
        t_array = TimedArray(np.multiply((1-alpha_arr[k]),current_B) +
                             np.multiply(   alpha_arr[k] ,current_A),
                             dt=defaultclock.dt)

        net.run(run_time, report = 'text')
        save_numbers = "("+str(current_index_B)+","+str(current_index_A)+")_"
        np.save(prefix+save_numbers+'spikes_t_'+str(n), spikes_AL.t)
        np.save(prefix+save_numbers+'spikes_i_'+str(n),spikes_AL.i)
        np.save(prefix+save_numbers+'I_'+str(n), G_AL.I_inj)
        np.save(prefix+save_numbers+'trace_V_'+str(n), trace_AL.V[:,start_skip:])
        np.save(prefix+save_numbers+'trace_t_'+str(n), trace_AL.t[start_skip:])

        lab = 0
        np.save(prefix+save_numbers+'labels_'+str(n), np.ones(len(trace_AL.t[start_skip:]))*lab)
        n = n+1




def runMNIST(run_params, imgs, states, net):
    num_train = run_params['num_trials']
    prefix = run_params['prefix']
    inp = run_params['input_intensity']
    run_time = run_params['time_per_image']
    N_AL = run_params['N_AL']
    labels = run_params['labels']
    numbers_to_inc = run_params['numbers_to_inc']
    bin_thresh = run_params['bin_thresh']
    n_input = run_params['n_input']

    G_AL = states['G_AL']
    spikes_AL = states['spikes_AL']
    trace_AL = states['trace_AL']

    num_run = {}
    for i in numbers_to_inc:
        num_run[i] = 0

    start = 100
    #run the network
    n = 0
    for i in range(60000):
        y = labels[i][0]
        if y in numbers_to_inc and num_run[y] < num_train:
            net.restore()
            print('image: ' + str(n))
            print('label: '+ str(labels[i][0]))

            #right now creating binary image
            rates = np.where(imgs[i%60000,:,:] > bin_thresh, 1, 0)*inp

            linear = np.ravel(rates)
            padding = N_AL - n_input
            I = np.pad(linear, (0,padding), 'constant', constant_values=(0,0))
            I = I+rshuffle(I, random_state = 10)+rshuffle(I, random_state = 2)

            I = np.where(I > inp, inp, I)
            print(np.sum(I/inp))

            G_AL.I_inj = nA*I

            net.run(run_time*ms, report = 'text')

            np.save(prefix+'spikes_t_'+str(n) ,spikes_AL.t)
            np.save(prefix+'spikes_i_'+str(n) ,spikes_AL.i)
            np.save(prefix+'I_'+str(n), I)
            np.save(prefix+'trace_V_'+str(n), trace_AL.V[:,start:])
            np.save(prefix+'trace_t_'+str(n), trace_AL.t[start:])

            np.save(prefix+'labels_'+str(n), np.ones(len(trace_AL.t[start:]))*y)
            n = n+1
            num_run[y] = num_run[y]+1
        if all(value == num_train for value in num_run.values()):
            break


'''
returns balanced binary MNIST datasets for num_train objects per class
'''
def get_bin_MNIST(imgs, labels, num_train, bin_thresh):

    X_train = []
    y_train = []

    num_run = {}
    for i in range(10):
        num_run[i] = 0
    for i in range(60000):
        y = labels[i][0]
        if num_run[y] < num_train:
            rates = np.where(imgs[i%60000,:,:] > bin_thresh, 1, 0)
            linear = np.ravel(rates)
            num_run[y] = num_run[y]+1
            X_train.append(linear)
            y_train.append(y)

        if all(value == num_train for value in num_run.values()):
            break

    return [np.array(X_train), np.array(y_train)]
'''
Random constant current input
N: size of array
p: probability of index having input
I_inp: strength of current
'''
def get_rand_I(N, p, I_inp):
    I = np.random.random(N)
    I[I > p] = 0
    I[np.nonzero(I)] = I_inp
    return I


'''
MNIST helper function to load and unpack MNIST data.  To run you need to download
the MNIST dataset from http://yann.lecun.com/exdb/mnist/.

picklename: name of file to write/read data from
MNIST_data_path: path to the MNIST data folder
bTrain: if tr
'''
def get_labeled_data(picklename, MNIST_data_path, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))#, encoding='utf-8')
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"), -1)
    return data
