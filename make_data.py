'''
Makes mock ensemble contact data for a 2D snake (spiral) and a 2D hairpin
'''
import numpy as np

np.random.seed(42)


def kth_diag_indices(a, k):
    """
    From
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


bead_diameter = 2
contact_distance = bead_diameter * 2
neighbor_cutoff = 2
gamma = 10
num_copies = 50
snake_side_length = 7


def zero_diagonals(a, n):
    b = a.copy()
    for i in range(-n, n+1):
        inds = kth_diag_indices(b, i)
        b[inds] = 0
    return b


top = np.vstack((np.arange(snake_side_length),
                 np.zeros(snake_side_length)))
right = np.array([snake_side_length - 1, 1])
left = np.array([0, 3])

s_shape_indices = np.hstack((top, right[:, None],
                             top[:, ::-1] + np.array([0, 2])[:, None],
                             left[:, None], top + np.array([0, 4])[:, None])).T
s_shape_indices = s_shape_indices.astype(int)

hairpin_shape_indices = np.zeros(((snake_side_length * 3 + 1) // 2, 2))
hairpin_shape_indices[:, 0] = np.arange((snake_side_length * 3 + 1) / 2)
hairpin_shape_indices = np.vstack((hairpin_shape_indices, [(snake_side_length * 3 + 1) / 2 - 1, 1],
                                   hairpin_shape_indices[::-1, :] + np.array([0, 2])[None, :]))
hairpin_shape_indices = hairpin_shape_indices.astype(int)

s_shape_indices *= bead_diameter
hairpin_shape_indices *= bead_diameter

n_beads = len(s_shape_indices)


def calculate_mock_counts(x):
    distance_matrix = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    distances = distance_matrix[np.triu_indices_from(distance_matrix, neighbor_cutoff)]

    return (num_copies / (1 + np.exp(-gamma * (dc - distances)))).astype(int)


mock_counts_hairpin = calculate_mock_counts(hairpin_shape_indices)
mock_counts_s = calculate_mock_counts(s_shape_indices)
summed_mock_counts = mock_counts_hairpin + mock_counts_s

noisy_counts = np.random.poisson(summed_mock_counts)


contact_matrix = np.zeros((n_beads, n_beads)).astype(int)
contact_matrix[np.triu_indices(n_beads, neighbor_cutoff)] = noisy_counts
contact_matrix[np.tril_indices(n_beads, -neighbor_cutoff)] = contact_matrix.T[np.tril_indices(n_beads, -neighbor_cutoff)]

with open('data.txt', 'w') as opf:
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            opf.write('{}\t{}\t{}\n'.format(i, j, contact_matrix[i, j]))
