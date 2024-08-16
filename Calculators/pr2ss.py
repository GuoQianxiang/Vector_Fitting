import numpy as np
from scipy.sparse import diags


def pr2ss(SER):
    """
    Convert a SER object to state-space representation.

    Parameters:
        SER: dict containing the following keys:
            - R: (Nc, Nc, N) array
            - poles: (Nc, Nc) array
            - D: (Nc,) array

    Returns:
        SER with updated A, B, C matrices.
    """

    R = SER['R']
    poles = SER['poles']
    Nc = len(SER['D'])
    N = R.shape[2]

    # Initialize C, A, B matrices
    C = np.zeros((Nc, Nc * N))
    A = np.zeros((Nc * N, 1))
    B = np.zeros((Nc * N, Nc))

    poles = np.diag(np.diag(poles))  # Make sure poles is a diagonal matrix

    for m in range(N):
        Rdum = R[:, :, m]
        for n in range(Nc):
            ind = n * N + m
            C[:, ind] = Rdum[:, n]

    for n in range(Nc):
        A[n * N:(n + 1) * N] = poles

        B[n * N:(n + 1) * N, n] = np.ones((N,))

    # Create sparse diagonal matrix for A
    A_sparse = diags(A.flatten())

    # Update SER dictionary
    SER['A'] = A_sparse
    SER['B'] = B
    SER['C'] = C

    return SER