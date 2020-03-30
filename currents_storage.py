from brian2 import *
import os.path

def create_TD_currents(run_time, N_AL, I_input, defaultclock):
    def Modulation(t):
        return (t >= 0 * ms)
    def ModulationSteps(t, lower_threshold, upper_threshold):
        return np.multiply((t >= lower_threshold * ms).astype(int),(t < upper_threshold * ms).astype(int))
    def cosineCurrent(I_arr, A, f, properShapeAllOnes,t_recorded):
        fullArray = np.multiply(I_arr,
                              np.multiply(properShapeAllOnes,
                               A * np.cos(f * t_recorded) * Modulation(t_recorded)).transpose()) * nA
        return fullArray
    # I = normalize(I_input)
    I = I_input
    # List time-dependent currents here:
    num_samples = int(run_time / defaultclock.dt)+1  # 10,000 samples
    # print("num_samples is "+str(num_samples))
    t_recorded = np.arange(num_samples) * defaultclock.dt
    properShapeAllOnes = np.ones((np.shape(t_recorded)[0], N_AL)).transpose()

    f_multiplier = 10

    A = 0.1
    f = f_multiplier*2 * np.pi / 20.0 / ms
    # Rabinovich Brian2 practice file sine current
    TD_I_arr_1 =  cosineCurrent(I[0], A, f, properShapeAllOnes, t_recorded)
    # print("TD_I_arr_1 is (should be 0.015 nA or 15 pA): "+str(TD_I_arr_1))
    # Dimensions (2400,1000)
    # print("Shape of TD_I_arr_1: " + str(np.shape(TD_I_arr_1)))

    # Oscillating current (small amplitude)
    A2 = 0.1
    TD_I_arr_2 = cosineCurrent(I[1], A2, f, properShapeAllOnes, t_recorded)

    # Oscillating current (medium amplitude)
    A3 = 0.5
    TD_I_arr_3 = cosineCurrent(I[2], A3, f, properShapeAllOnes, t_recorded)

    # Oscillating current (Large amplitude)
    A4 = 2
    TD_I_arr_4 = cosineCurrent(I[3], A4, f, properShapeAllOnes, t_recorded)

    # Sigmoid current
    A5 = 0.1
    f5 = f_multiplier*0.1*2 * np.pi / 20.0 / ms
    TD_I_arr_5 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A5 * np.divide(1.0,1.0+np.exp(-f5 * t_recorded)) * Modulation(t_recorded)).transpose()) * nA

    # Alternating currents every dt
    switch_odor_list = np.random.random(np.shape(t_recorded))
    switch_odor = np.array([switch_odor_list for i in range(N_AL)])
    switch_odor[switch_odor<0.5] = 0
    switch_odor[switch_odor>0.5] = 1
    A6 = A5
    f6 = 0 / ms
    TD_I_arr_6 = (np.multiply(switch_odor.transpose(),
                             cosineCurrent(I[4], A6, f6, properShapeAllOnes, t_recorded) / nA) \
        + np.multiply(1-switch_odor.transpose(),
                             cosineCurrent(I[5], A6, f6, properShapeAllOnes, t_recorded) / nA) )*nA

    A7 = 0.1
    f7 = 0 / ms
    # Constant (static)
    TD_I_arr_7 = cosineCurrent(I[7], A7, f7, properShapeAllOnes, t_recorded)


    A8 = A6
    f8 = f6
    # Constant (static)
    TD_I_arr_8 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A8 * np.cos(f8 * t_recorded) * ModulationSteps(t_recorded,0,50)).transpose()) * nA \
    + np.multiply(I[5],
                              np.multiply(properShapeAllOnes,
                               A8 * np.cos(f8 * t_recorded) * ModulationSteps(t_recorded,50,100)).transpose()) * nA \
    + np.multiply(I[6],
                               np.multiply(properShapeAllOnes,
                                           A8 * np.cos(f8 * t_recorded) * ModulationSteps(t_recorded, 100,
                                                                                          150)).transpose()) * nA


    A9 = 1
    f9 = 0 / ms
    # Constant (static)
    TD_I_arr_9 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A9 * np.cos(f9 * t_recorded) * ModulationSteps(t_recorded,0,50)).transpose()) * nA

    A10 = 1
    f10 = 0 / ms
    # Constant (static)
    TD_I_arr_10 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A10 * np.cos(f10 * t_recorded) * (
                                              ModulationSteps(t_recorded,0,50) +
                                              ModulationSteps(t_recorded,350,400) +
                                              ModulationSteps(t_recorded, 700, 750) +
                                              ModulationSteps(t_recorded, 1000, 1050)
                                          )).transpose()) * nA

    A11 = 0
    f11 = 0 / ms
    # Constant (static)
    TD_I_arr_11 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A11 * np.cos(f11 * t_recorded) * ModulationSteps(t_recorded,0,50)).transpose()) * nA

    A12 = 0.1
    f12 = f_multiplier*0.2*np.pi/20.0/ms
    # Constant (static)
    TD_I_arr_12 = cosineCurrent(I[7], A12, 0 / ms, properShapeAllOnes, t_recorded) \
                  + cosineCurrent(I[7], A12, f12, properShapeAllOnes, t_recorded)

    A13 = 0.25
    f13 = f_multiplier*0.2 * np.pi / 20.0 / ms
    # Constant (static)
    TD_I_arr_13 = cosineCurrent(I[7], 0.1, 0 / ms, properShapeAllOnes, t_recorded) \
                  + cosineCurrent(I[7], A13, f13, properShapeAllOnes, t_recorded)

    A14 = 5
    f14 = f_multiplier*0.2 * np.pi / 20.0 / ms
    # Constant (static)
    TD_I_arr_14 = cosineCurrent(I[7], 0.1, 0 / ms, properShapeAllOnes, t_recorded) \
                  + cosineCurrent(I[7], A14, f14, properShapeAllOnes, t_recorded)


    A15 = 3
    f15 = 0 / ms
    # Constant (static)
    TD_I_arr_15 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A15 * np.cos(f15 * t_recorded) * (
                                              ModulationSteps(t_recorded,0,50) +
                                              ModulationSteps(t_recorded,350,400) +
                                              ModulationSteps(t_recorded, 700, 750) +
                                              ModulationSteps(t_recorded, 1000, 1050)
                                          )).transpose()) * nA

    A16 = 5
    f16 = 0 / ms
    # Constant (static)
    TD_I_arr_16 = np.multiply(I[4],
                              np.multiply(properShapeAllOnes,
                               A16 * np.cos(f16 * t_recorded) * (
                                              ModulationSteps(t_recorded,0,50) +
                                              ModulationSteps(t_recorded,350,400) +
                                              ModulationSteps(t_recorded, 700, 750) +
                                              ModulationSteps(t_recorded, 1000, 1050)
                                          )).transpose()) * nA



    TD_I_arr_list=[]
    # TD_I_arr_list.append(TD_I_arr_1)
    # TD_I_arr_list.append(TD_I_arr_2)
    # TD_I_arr_list.append(TD_I_arr_3)
    # TD_I_arr_list.append(TD_I_arr_4)
    # TD_I_arr_list.append(TD_I_arr_5)
    # TD_I_arr_list.append(TD_I_arr_6)
    # TD_I_arr_list.append(TD_I_arr_7)
    # TD_I_arr_list.append(TD_I_arr_8)
    # TD_I_arr_list.append(TD_I_arr_9)
    # TD_I_arr_list.append(TD_I_arr_10)
    # TD_I_arr_list.append(TD_I_arr_11)
    # TD_I_arr_list.append(TD_I_arr_12)
    # TD_I_arr_list.append(TD_I_arr_13)
    # TD_I_arr_list.append(TD_I_arr_14)
    # TD_I_arr_list.append(TD_I_arr_15)
    # TD_I_arr_list.append(TD_I_arr_16)

    # to make sine perturbed versions
    # for aCurrent in I:
    #     A_unperturbed = 3 # causes amplitude of base current to be 0.45 nA
    #     f_unperturbed = 0 / ms # causes base current to be time-independent
    #     A_perturbed = 2.0 # causes perturbations to have amplitude 0.3 nA
    #     f_perturbed = (2.0 * np.pi) * 0.03 / ms # Sine perturbations have frequency 30 Hz
    #     # Base currents perturbed by sine waves
    #     TwoD_version_of_current = cosineCurrent(aCurrent, A_unperturbed, f_unperturbed, properShapeAllOnes, t_recorded) + \
    #                               cosineCurrent(aCurrent, A_perturbed, f_perturbed, properShapeAllOnes, t_recorded)
    #     TD_I_arr_list.append(TwoD_version_of_current)
    #     print(TwoD_version_of_current)

    # to make multiple perturbed versions of same static current
    odor_index = 0
    number_of_perturbations = 10
    for i in range(number_of_perturbations):
        A_unperturbed = 3 # causes amplitude of base current to be 0.45 nA
        f_unperturbed = 0 / ms # causes base current to be time-independent
        A_perturbed = 3.0 # causes perturbations to have amplitude 0.45 nA
        f_perturbed = (2.0 * np.pi) * 0.006 * i / ms # Sine perturbations have frequency 30 Hz for i=5
        # Base currents perturbed by sine waves
        TwoD_version_of_current = cosineCurrent(I[odor_index], A_unperturbed, f_unperturbed, properShapeAllOnes, t_recorded) + \
                                  cosineCurrent(I[odor_index], A_perturbed, f_perturbed, properShapeAllOnes, t_recorded)
        TD_I_arr_list.append(TwoD_version_of_current)
        # print(TwoD_version_of_current)

    return np.multiply(1,TD_I_arr_list)*amp

def add_noise(arrayCurrent_list, num_odors, num_trials, N_AL, run_time, defaultclock, noise,prefix,inp):
    num_samples = int(run_time / defaultclock.dt)  # 10,000 samples
    t_recorded = np.arange(num_samples) * defaultclock.dt
    properShapeAllOnes = np.ones((np.shape(t_recorded)[0], N_AL))
    currentWithNoise_list = []
    random_arr = np.zeros(N_AL)
    if os.path.exists(prefix+"random_multiplier.txt"):
        random_arr = np.loadtxt(prefix + "random_multiplier.txt")
        if len(random_arr)>0:
            print("Adding noise (first 6 of first list in random_arr):\n"+str(random_arr[0][:6]))
        else:
            print("No random noise to load.")
    else:
        print("No random noise to load.")

    print("Inp and noise amplitude: " + str((inp, noise)))
    for i in range(num_odors):
        for j in range(num_trials):
            n = i*num_trials + j
            random_arr = np.zeros(N_AL)
            properShapeRandom = inp*noise*np.multiply(properShapeAllOnes, 2*random_arr-1)
            currentWithNoise_list.append(arrayCurrent_list[i] + properShapeRandom*nA)
    return np.array(currentWithNoise_list)
    print("Noise is: "+str(noise))

def normalize(an_array):
    max = np.max(an_array)
    min = np.min(an_array)
    return np.divide(max - an_array, max - min)

