import numpy as np



def yellin(max_gap, scale, n = 5, trial = 10000):
    """
    Returns the confidence level of exclusion

    Parameters
    ----------
    max_gap : np.array, len = n
        maximum interval of [0...(n-1)] events
    scale : int/float
        expected total number of events
    n : int
        largest number of event interval to use for the Yellin method
    trial : int
        number of Monte carlo realization

    Returns
    -------
    The confidence level of exclusion that scale is too large for the
    given max_gap
    """

    # generate poisson realizations of MC
    N = np.random.poisson(scale, size = trial)

    # calculate the intervals of the trials
    x_gap = np.ones((trial, n))*scale
    for i in range(trial):
        event = np.random.rand(N[i])
        event = np.sort(event)
        event = np.insert(event, 0, 0)
        event = np.insert(event, len(event), 1)

        for j in range(n):
            if j + 1 >= len(event):
                break
            x_gap[i, j] = (np.max(event[j+1:] - event[:-(j+1)]))*scale

    # calculate C_n for all trials
    c = np.zeros((trial, n))
    for i in range(trial):
        c[i,:] = np.sum(x_gap<x_gap[i], axis = 0)/trial

    # calculate C_max of the trials
    c_max_mc = np.max(c,axis = 1)

    # calculate C_max of the input experiment
    c_max = np.max(np.sum(x_gap<max_gap, axis = 0)/trial)
    return np.sum(c_max_mc<c_max)/trial
