import numpy as np

def pick_rand_sample(z_enc, batch_size):
    
    while True:
        batch_idxs = np.random.randint(0, len(z_enc), batch_size)
        z_batch = z_enc[batch_idxs]
        yield z_batch