Sine perturbation on top of static current
uses
        A_unperturbed = 3 # causes amplitude of base current to be 0.45 nA
        f_unperturbed = 0 / ms # causes base current to be time-independent
        A_perturbed = 0.7 # causes perturbations to have amplitude 0.105 nA
        f_perturbed = (2.0 * np.pi) * 0.03 / ms # Sine perturbations have frequency 30 Hz