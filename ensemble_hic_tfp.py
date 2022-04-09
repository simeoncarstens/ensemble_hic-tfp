'''
Bayesian inference of a multi-state 3D conformations using Tensorflow 
Probability

This script illustrates model definition and HMC sampling via Tensorflow 
Probability on an inference problem occuring in chromosome structure
determination from population-based contact data (Carstens et al., PNAS
2020). Find the paper at https://www.pnas.org/content/117/14/7824.abstract.

We consider two chains of 23 spherical particles each. The two chains take
on two different conformations: a hairpin shape which looks as follows

**********
*
**********

and an "S" shape which looks like, well, an "S":

*******
*
*******
      *
*******

We now consider a smooth approximation to a contact function: if two
particles are closer than a distance of d_c units, then we get a "1"
and if the distance is smaller, then we measure a count "0". The com-
policating issue now is that what we measure are not the contacts for
each chain separately, but only something proportional to the sum.
That means that if in both chains particle 1 and 17 are "in contact"
(closer than d_c), we will measure (modulo the global proportionality
constant) a value of approximately "2". If the contact is formed in
only one of the chains, we will measure "1" . If the contact is formed
in neither chain, we will measure "0".

The inference problem now is to infer the two conformational states
from this "averaged" data.
'''
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_v2_behavior()

tfd = tfp.distributions

# load in some test data
# data = np.loadtxt('hairpin_s_fwm_poisson.txt')
data = np.loadtxt('data.txt')
# set number of spheres (specific to the above dataset)
n_beads = 23
# contact distance below which a contact is formed
d_c = 4
gamma = 50
neighbor_distance = 2
collision_distance = neighbor_distance
bead_markersize = 25
n_samples = 5000


def forward_model(pairs, X):
    '''
    The forward model is simple: for a given pair of spherical particles,
    calculate the distances in both chains, pass them through a shifted
    sigmoidal function (which approximates the contact function)
    and sum the result over the two chains, yielding a result in
    the interval (0, 2). Then multiply this by the (for now known)
    proportionality constant of ~50.
    '''
    # Tensorflow currently can't do fancy indexing à la
    # partners1 = X[:,pairs[:,0]]
    Xdim = len(X.shape)
    partners1 = tf.gather(X, pairs[:, 0], axis=Xdim - 2)
    partners2 = tf.gather(X, pairs[:, 1], axis=Xdim - 2)
    distances = tf.linalg.norm(partners1 - partners2, axis=Xdim - 1)
    smoothed_contacts = tf.math.sigmoid(d_c - distances)

    # Tensors don't have a .sum() method, instead, tf.math.reduce_sum is used
    return gamma * tf.math.reduce_sum(smoothed_contacts, axis=Xdim - 3)


def log_likelihood(pairs, counts, X):
    '''
    We describe deviations of the measured data from the ideal, back-
    calculated data using Poisson distributions with the rates given by the
    back-calculated data. The log-likelihoods for all data points are then
    summed up to yield the likelihood of the complete dataset.
    '''
    rates = forward_model(pairs, X)
    # batching at work: this really defines many Poisson distributions,
    # each with a different rate, and evaluates them at different values
    likelihoods = tfd.Poisson(rate=rates).log_prob(counts)

    return tf.math.reduce_sum(likelihoods, axis=1)


def log_backbone_prior(X):
    '''
    We have some prior information: we know that the spherical particles form
    chains. We thus imput the distance between one sphere and its next neighbor
    in the chain to a certain distance using a Normal distribution with the
    mean at that distance.
    Again, all log-probabilities are summed up to give the total prior probability.
    '''
    neighbor_distance_sigma = 0.1
    # again batching: this actually defines several normal distributions
    subsequents = tfd.Normal(np.float32(neighbor_distance),
                             neighbor_distance_sigma)
    neighboring_distances = tf.linalg.norm(
        X[..., 1:, :] - X[..., :-1, :], axis=-1)
    log_probs = subsequents.log_prob(neighboring_distances)
    return tf.math.reduce_sum(log_probs, axis=(1, 2))


def log_volume_exclusion_prior(X):
    force_constant = 100
    distances = tf.linalg.norm(
        X[:, :, None, ...] - X[:, :, :, None, ...], axis=-1)
    # build a mask in order to get upper triangular distance matrix,
    # but without the diagonal
    ones = tf.ones_like(distances)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    mask_b = tf.linalg.band_part(ones, 0, 0)
    # this is the final mask
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)
    # subtract collision distance from entries that pass the mask
    dist_diffs = tf.ragged.boolean_mask(distances, mask) - collision_distance
    # build another mask that, this time, lets only negative differences
    # bead distance and collision distance pass
    collision_mask = tf.cast(dist_diffs < 0, dtype=tf.bool)
    # these are now the distance differences that indicate a collision
    collision_diffs = tf.ragged.boolean_mask(dist_diffs, collision_mask)
    return -tf.math.reduce_sum(force_constant * collision_diffs ** 4, axis=(1, 2, 3))


# pairs is a list of shere pairs for which a contact was measured
pairs = data[:, :2].astype(int)
# counts are the measured count values (proportionality constant *
# (some value in (0, 2)
data_counts = data[:, 2].astype(np.float64)


def unnormalized_log_posterior(X):
    '''
    The unnormalized log-posterior is just the sum of the total log-likelihood
    and the total log-prior probability
    '''
    return log_backbone_prior(X) + log_volume_exclusion_prior(X) + log_likelihood(pairs, data_counts, X)


# inverse temperatures (betas) for Replica Exchange
inverse_temperatures = 0.5 ** tf.range(5, dtype=np.float32)

initial_state = tf.random.uniform(
    shape=(2, n_beads, 3), minval=-3, maxval=3)


# the Replica Exchange kernel takes a factory function that, given
# a target log-probability, returns another (single-chain) sampling kernel
def make_kernel_fn(target_log_prob_fn):
    # we want to sample the posterior distribution with Hamiltonian Monte Carlo
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_posterior,
        step_size=0.025,
        num_leapfrog_steps=20
    )

    # ...actually, let's adapt stepsizes automatically such that the acceptance
    # rates are around 50%.
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=int(1000),
        target_accept_prob=0.5
    )

    return adaptive_kernel


# after each sampling step, this function determines what, in addition to the
# samples, is being returned to the user.
# We return
# - a boolean matrix of shape (n_samples, n_replicas - 1) containing True for replica pairs for which a swap was proposed,
# - a similarly shaped boolean matrix containing True for replica pairs which were swapped successfully,
# - a boolean matrix (n_samples, n_replicas) with True where a HMC step was accepted
def trace_swaps(unused_state, results):
    return (results.is_swap_proposed_adjacent,
            results.is_swap_accepted_adjacent,
            results.post_swap_replica_results.inner_results.is_accepted)


remc_kernel = tfp.mcmc.ReplicaExchangeMC(
    # the next two lines make only the likelihood tempered with beta
    target_log_prob_fn=None,
    untempered_log_prob_fn=log_backbone_prior,
    tempered_log_prob_fn=lambda X: log_likelihood(pairs, data_counts, X),
    inverse_temperatures=inverse_temperatures,
    make_kernel_fn=make_kernel_fn)


# Decorating that function with tf.function builds a static compute graph,
# which is much faster. In this case. though, it doesn't seem to work.
@tf.function
def run_chain(initial_state, num_results=5000):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_state,
        kernel=remc_kernel,
        num_burnin_steps=500,
        trace_fn=trace_swaps)


samples, (is_swap_proposed_adjacent, is_swap_accepted_adjacent, hmc_accepted) = (
    run_chain(initial_state, n_samples)
)

# (...).is_accepted is a tensor containing boolean values for whether a
# HMC move was accepted or not. This has to be transformed to a numpy
# array before we can calculate the mean to obtain the average acceptance
# rate. Same holds for [...]_accepted_[...] tensors.
print("RE acceptance rates:", is_swap_accepted_adjacent.numpy().sum(
    0) / is_swap_proposed_adjacent.numpy().sum(0))
print("HMC acceptance rates:", hmc_accepted.numpy().mean(0))

samples = samples.numpy()

# we visualize the data and results
last_structures = samples[-1]
fig = plt.figure()


def plot_contact_matrix(ax, counts):
    contact_matrix = np.zeros((23, 23))
    for (i, j), c in zip(pairs, counts):
        contact_matrix[i, j] = c
    contact_matrix += contact_matrix.T
    ax.matshow(np.log(contact_matrix + 1e-3))
    ax.set_title('contact data')
    ax.set_xticks(())
    ax.set_yticks(())


# this plots the raw data, meaning, the contact matrix
ax1 = fig.add_subplot(321)
plot_contact_matrix(ax1, data_counts)

# this plots, for the last sample, the back-calculated, idealized contact
# matrix. This should look somewhat like the previous plot
ax2 = fig.add_subplot(322)
mock_data = forward_model(pairs, last_structures)
plot_contact_matrix(ax2, mock_data)

# this plots the back-calculated contact matrix for only the first state
ax3 = fig.add_subplot(323)
mock_data = forward_model(pairs, last_structures[None, 0])
plot_contact_matrix(ax3, mock_data)

# this plots the back-calculated contact matrix for only the second state.
# Both added together yield the top right (combined back-calculated) matrix
ax4 = fig.add_subplot(324)
mock_data = forward_model(pairs, last_structures[1:2])
plot_contact_matrix(ax4, mock_data)

# 3D visualization of the conformation of the first chain in the last HMC
# sample. After rotating properly, you should see either a hairpin or an S shape
plot_args = dict(marker="o", markersize=bead_markersize, alpha=0.75)
ax5 = fig.add_subplot(325, projection="3d")
ax5.plot(*last_structures[0].T, **plot_args)
ax5.axis("off")
ax5.set_title('3D visualization\nof 1st state')

# 3D visualization of the conformation of the first chain in the last HMC
# sample. After rotating properly, you should see either a hairpin or an S shape
ax6 = fig.add_subplot(326, projection="3d")
ax6.plot(*last_structures[1].T, **plot_args)
ax6.axis("off")
ax6.set_title('3D visualization\nof 2nd state')

# If we're unlucky, we might also end up in a local minimum: two half-S shapes added
# together also reproduce the input data

fig.tight_layout()
plt.show()
