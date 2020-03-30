import numpy as np
"""
This file includes utility functions needed for working with triplets and the triplet loss.
"""

def create_triplets(data: np.ndarray, num_triplets: int = 300) -> (np.ndarray, np.ndarray):
    """
    Creates triplets from the provided data. To do this, an anchor is randomly selected and then a positive and negativ
    sample for that anchor are selected.
    The function returns an array of size (num_triplets, 3)

    :param data: the data from which the triplets should be formed. needs to be an array of measurements
    :param num_triplets: number of triplets to be created
    :return: array of triplets (num_triplets, 3), label_of_triplets (num_triplets, 3)
    """
    #Create Triplets for train
    triplets = np.zeros((0,3,64,49))
    triplet_labels = []

    for i in range(num_triplets):
        anchor_index = int(np.random.random()*len(data))
        anchor = data[anchor_index]
        found_pos = False
        found_neg = False
        while not found_pos:
            pos_index = int(np.random.random()*len(data))
            pos = data[pos_index]
            if pos.label == anchor.label and pos_index != anchor_index:
                found_pos = True

        while not found_neg:
            neg = data[int(np.random.random()*len(data))]
            if neg.label != anchor.label:
                found_neg = True

        #create data
        anchor_data = np.expand_dims(anchor.get_data()[:64,:], axis=0)
        pos_data = np.expand_dims(pos.get_data()[:64,:], axis=0)
        neg_data = np.expand_dims(neg.get_data()[:64,:], axis=0)
        triplet = np.expand_dims(np.vstack((np.vstack((anchor_data, pos_data)),neg_data)), axis=0)
        triplets = np.vstack((triplets,triplet))
        triplet_labels.append((anchor.label, pos.label, neg.label))

    print(triplets.shape)
    return triplets, triplet_labels


def getInputBatchFromTriplets(train_triplets: np.ndarray, val_triplets: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    This function takes in an array of triplets and flattens them so they can easily be used in a model for training

    It also normalizes both the train and the validation batch
    :param train_triplets: train triplet array: shape (num_train_triplets, 3)
    :param val_triplets: val triplet array: shape (num_val_triplets, 3)
    :return: train_batch and val_batch both with shape (3*num_xxx_triplets, 1)
    """
    def flatten_triplets(triplets):
        N = len(triplets)
        T = 3
        W = triplets.shape[2]
        C = triplets.shape[3]
        input_batch = np.empty([N * T, W, C])

        for i in range(N):
            for t in range(T):
                input_batch[i * T + t] = triplets[i][t][:][:]

        return input_batch

    train_batch = flatten_triplets(train_triplets)
    val_batch = flatten_triplets(val_triplets)

    # normalize
    mn = np.min([np.min(train_batch), np.min(val_batch)])
    mx = np.max([np.max(train_batch), np.max(val_batch)])

    train_batch = (train_batch - mn) / (mx - mn)
    val_batch = (val_batch - mn) / (mx - mn)

    return train_batch, val_batch