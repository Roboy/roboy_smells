import numpy as np

def create_triplets(data, num_triplets=300):
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


def getInputBatchFromTriplets(train_triplets, val_triplets):
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