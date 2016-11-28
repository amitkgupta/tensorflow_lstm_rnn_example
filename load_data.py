# DUMMY DATA
def load_data(path_to_data=None, create_dummy_data=False):
    if not create_dummy_data:
        raise NotImplementedError

    return load_dummy_data()

def load_dummy_data():
    import random
    from random import shuffle

    # MAKE IT DETERMINISTIC TO PROVE TF SAVE/RESTORE WORKS
    from random_seed import RANDOM_SEED
    random.seed(RANDOM_SEED)

    NUM_CLASSES = 16
    total_observations = 2**(NUM_CLASSES-1)

    import numpy as np

    observations = np.array([map(lambda ch: [int(ch)], '{0:b}'.format(n).zfill(NUM_CLASSES-1)) for n in range(total_observations)])
    shuffle(observations)

    one_hots = np.zeros((len(observations), NUM_CLASSES), dtype=np.int32)
    for idx, obs in enumerate(observations):
        one_hots[idx][sum(observations[idx])] = 1

    return observations, one_hots

X_all, y_all = load_data(create_dummy_data=True)

# Test/train split, roughly 50/50
num_training_observations = len(X_all)/2
X_train = X_all[:num_training_observations]
y_train = y_all[:num_training_observations]
X_test = X_all[num_training_observations:]
y_test = y_all[num_training_observations:]
