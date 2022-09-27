import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

if True:
    plt.style.use('dark_background')
plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 2,
    'lines.markersize': 5,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.figsize': (6.4, 6.4)
})

# PREPARE DATA

from indl.misc.sigfuncs import sigmoid
from functools import partial

N_LATENTS = 4
N_SENSORS = 32
N_CLASSES = 5
FS = 25
DURATION = 2  # seconds
n_timesteps = int(DURATION * FS)
tvec = np.arange(n_timesteps) / FS


def make_signal_generator(n_latents=N_LATENTS, n_sensors=N_SENSORS,
                          n_classes=N_CLASSES, n_timesteps=n_timesteps,
                          seed=99):
    np.random.seed(seed)
    t = np.arange(n_timesteps) / FS
    f_sig = partial(sigmoid, B=5, x_offset=1.0)
    class_amps = np.random.uniform(low=-1.0, high=1.0, size=(N_CLASSES, N_LATENTS))
    class_amps /= np.sum(np.abs(class_amps), axis=1, keepdims=True)
    f_sig = partial(sigmoid, B=5, x_offset=1.0)
    mix_mat = np.random.randn(N_SENSORS, N_LATENTS)
    mix_mat /= np.sum(np.abs(mix_mat), axis=1, keepdims=True)

    def draw_sample(class_ix):
        latent_mods = class_amps[class_ix, :, None] * f_sig(t)[None, :]
        latent_protos = 0.5 * np.ones((N_LATENTS, 1)) + 0.1 * np.random.randn(N_LATENTS, n_timesteps)
        latent_protos /= np.std(latent_protos, axis=1, keepdims=True)
        latent_class_dat = latent_mods * latent_protos  # (N_LATENTS, n_timesteps)
        sensor_class_dat = mix_mat @ latent_class_dat

        sig = sensor_class_dat.T
        return sig

    return draw_sample


sig_gen = make_signal_generator()
for class_ix in range(N_CLASSES):
    plt.subplot(2, 3, class_ix + 1)
    plt.plot(tvec, sig_gen(class_ix))
plt.show()

N_TRIALS = 10000
BATCH_SIZE = 8

sig_gen = make_signal_generator()
Y = np.random.randint(0, high=N_CLASSES, size=N_TRIALS)
X = np.stack([sig_gen(_) for _ in Y])
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(BATCH_SIZE, drop_remainder=True)
dataset.element_spec

# MODIFY DATASET
ae_dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32),
                                       (tf.zeros(0, dtype=tf.float32), tf.cast(x, tf.float32))))
print(ae_dataset.element_spec)

# Variational autoencoder using tensorflow-probability
tfb = tfp.bijectors


def make_mvn_prior(ndim, trainable=False, offdiag=False):
    if not trainable:
        if offdiag:
            # With covariances
            # Note: Diag must be > 0, upper triangular must be 0, and lower triangular may be != 0.
            prior = tfd.MultivariateNormalTriL(
                loc=tf.zeros(ndim),
                scale_tril=tf.eye(ndim)
            )
        else:
            if True:  # kl_exact needs same dist types for prior and latent.
                prior = tfd.MultivariateNormalDiag(loc=tf.zeros(ndim), scale_diag=tf.ones(ndim))
            else:
                prior = tfd.Independent(tfd.Normal(loc=tf.zeros(ndim), scale=1),
                                        reinterpreted_batch_ndims=1)
    else:
        # Note, in TransformedVariable, the initial value should be that AFTER transform
        if offdiag:
            prior = tfd.MultivariateNormalTriL(
                loc=tf.Variable(tf.random.normal([ndim], stddev=0.1, dtype=tf.float32),
                                name="prior_loc"),
                scale_tril=tfp.util.TransformedVariable(
                    tf.random.normal([ndim, ndim], mean=1.0, stddev=0.1, dtype=tf.float32),
                    tfb.FillScaleTriL(), name="prior_scale")
            )
        else:
            scale_shift = np.log(np.exp(1) - 1).astype(np.float32)
            prior = tfd.MultivariateNormalDiag(
                loc=tf.Variable(tf.random.normal([ndim], stddev=0.1, dtype=tf.float32),
                                name="prior_loc"),
                scale_diag=tfp.util.TransformedVariable(
                    tf.random.normal([ndim], mean=1.0, stddev=0.1, dtype=tf.float32),
                    bijector=tfb.Chain([tfb.Shift(1e-5), tfb.Softplus(), tfb.Shift(scale_shift)]),
                    name="prior_scale"
                )
            )
    return prior


def make_mvn_dist_fn(_x_, ndim, offdiag=False):
    _loc = tfkl.Dense(ndim)(_x_)
    if offdiag:
        _scale = tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(ndim) - ndim)(_x_)

        _scale2=tf.convert_to_tensor(_scale) #################
        print(f'_scale:{_scale}')
        print(f'type _scale:{type(_scale)}')


        #_scale = tfp.bijectors.FillScaleTriL()(_scale)  # softplus(x+0.5413) + 1e-5 --> lower tri mat
        _scale = tfp.bijectors.FillScaleTriL()(_scale2)  # softplus(x+0.5413) + 1e-5 --> lower tri mat

        #b=tfp.bijectors.FillScaleTriL()
        #_scale=b.forward(_scale)

        #_scale = tfp.bijectors.FillScaleTriL.forward(_scale)  # softplus(x+0.5413) + 1e-5 --> lower tri mat
        make_dist_fn = lambda t: tfd.MultivariateNormalTriL(loc=t[0], scale_tril=t[1])
    else:
        _scale = tfkl.Dense(ndim)(_x_)
        _scale = tf.math.softplus(_scale + np.log(np.exp(1) - 1)) + 1e-5
        if True:  # Match type with prior
            make_dist_fn = lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=t[1])
        else:
            make_dist_fn = lambda t: tfd.Independent(tfd.Normal(loc=t[0], scale=t[1]))
    return make_dist_fn, [_loc, _scale]


#########################

LATENT_DIM = 3

vae_model = None
encoder_model = None
K.clear_session()
K.set_floatx('float32')
tf.random.set_seed(42)

kl_beta = K.variable(value=0.0)
kl_beta._trainable = False  # It isn't trained. We set it explicitly with the callback.


def kl_beta_update(epoch_ix, N_epochs, M_cycles=4, R_increasing=0.5):
    T = N_epochs // M_cycles
    tau = (epoch_ix % T) / T
    new_beta_value = tf.minimum(1.0, tau / R_increasing)
    # new_beta_value = new_beta_value * BATCH_SIZE  #  / N_TRIALS
    K.set_value(kl_beta, new_beta_value)


# Add the following to the list of model.fit callbacks.
# (we may overwrite this later after we've defined our N_EPOCHS constant)
kl_beta_cb = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch, log: kl_beta_update(epoch, N_epochs=30))

LATENT_OFFDIAG = True  # Set to False to restrict latents to being independent (diagonal covariance)
N_HIDDEN = 16
NUM_SAMPLES = 1

# Create the prior which will be used to calculate KL-divergence of latent dists:
prior = make_mvn_prior(LATENT_DIM, trainable=False, offdiag=LATENT_OFFDIAG)

_input = tfkl.Input(shape=(n_timesteps, N_SENSORS))
_x = tfkl.Bidirectional(tfkl.LSTM(N_HIDDEN), merge_mode="sum")(_input)
make_dist_fn, params = make_mvn_dist_fn(_x, LATENT_DIM, offdiag=LATENT_OFFDIAG)
q_z = tfpl.DistributionLambda(
    name="q_z",
    make_distribution_fn=make_dist_fn,
    convert_to_tensor_fn=lambda s: s.sample(NUM_SAMPLES),
    activity_regularizer=tfpl.KLDivergenceRegularizer(
        prior, use_exact_kl=True, weight=kl_beta)
)(params)

# encoder_model = tf.keras.Model(inputs=_input, outputs=q_z)
# encoder_model.summary()
# tf.keras.utils.plot_model(encoder_model)

# DECODER
OUTPUT_OFFDIAG = False

# tfkl.RepeatVector doesn't work on distributions, so we do it manually by broadcast-add to zeros.
_y = q_z[..., tf.newaxis, :] + tf.zeros([n_timesteps, 1])
_y = tf.reshape(_y, [-1, n_timesteps, LATENT_DIM])  # RNN requires ndim=3: Collapse samp + batch dims
_y = tfkl.LSTM(N_HIDDEN, return_sequences=True)(_y)
_y = tf.reshape(_y, [NUM_SAMPLES, -1, n_timesteps, N_HIDDEN])  # Restore sample dim

# GRU output parameterizes distribution
make_dist_fn, params = make_mvn_dist_fn(_y, N_SENSORS, offdiag=OUTPUT_OFFDIAG)

p_full = tfpl.DistributionLambda(
    make_distribution_fn=make_dist_fn,
    name="p_full"
    # No KL regularizer on output distribution. Will calculate p(recon|inputs) as model loss.
)(params)

# MODEL
vae_model = tf.keras.Model(inputs=_input, outputs=[q_z, p_full])
vae_model.summary()
# tf.keras.utils.plot_model(vae_model)

# KERAS TRAINING
N_EPOCHS = 30
optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae_model.compile(optimizer=optim,
                  loss=[lambda x_true, enc_out: tfd.kl_divergence(enc_out, prior),
                        lambda x_true, model_out: -model_out.log_prob(x_true)],
                  loss_weights=[0.0, 1.0])
history = vae_model.fit(ae_dataset,
                        epochs=N_EPOCHS,
                        callbacks=[kl_beta_cb],
                        verbose=1)
##############
plt.figure(figsize=(8, 8))

trial_ix = 10
test_X = X[trial_ix, :, :].astype(np.float32)[None, ...]
recon_X = np.mean(vae_model(test_X)[1].mean().numpy(), axis=0)

for chan_ix, chan_idx in enumerate([0, 1, 2, 15, 16, 17, 28, 29, 30]):
    plt.subplot(3, 3, chan_ix + 1)
    plt.plot(tvec, test_X[0, :, chan_idx], label="Input")
    plt.plot(tvec, recon_X[0, :, chan_idx], label="Recon")
    plt.title(f"Sensor {chan_idx}")
    if chan_ix == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# Visualize Latents

# Scatter plot per class
plot_range = [-2.0, 2.0]
fig = plt.figure(figsize=[8, 6], tight_layout=True)
axes = fig.subplots(2, 2)

for class_id in np.unique(Y):
    b_class = Y == class_id
    lat_dist = vae_model(X[b_class].astype(np.float32))[0]
    lat_dist = lat_dist.mean().numpy()
    for pair_ix, dim_pair in enumerate([[0, 1], [0, 2], [1, 2]]):
        row_ix = pair_ix // 2
        col_ix = pair_ix % 2
        axes[row_ix, col_ix].scatter(lat_dist[:, dim_pair[0]], lat_dist[:, dim_pair[1]])
        axes[row_ix, col_ix].set_xlim(plot_range)
        axes[row_ix, col_ix].set_ylim(plot_range)
        axes[row_ix, col_ix].set_xlabel(f"ax {dim_pair[0]}")
        axes[row_ix, col_ix].set_ylabel(f"ax {dim_pair[1]}")

gen_model = tf.keras.Sequential([
    tfkl.Input(shape=(LATENT_DIM,)),
    tfkl.RepeatVector(n_timesteps),
    tfkl.LSTM(N_HIDDEN, return_sequences=True),
    tfkl.Dense(N_SENSORS)
])
gen_model.layers[-2].set_weights(vae_model.get_layer(name='lstm_1').get_weights())
gen_model.layers[-1].set_weights(vae_model.get_layer(name='dense_2').get_weights())
gen_model.summary()

N_GRID = 5
dim_idx = [0, 1]
sensor_ix = 0
samp_amp = 2.0

d0 = np.linspace(-samp_amp, samp_amp, N_GRID)
d1 = np.linspace(-samp_amp, samp_amp, N_GRID)
v0, v1 = np.meshgrid(d0, d1)

fig = plt.figure()
fig.suptitle(f"Sensor {sensor_ix} gen from latent dims {dim_idx}")
for pl_ix in range(N_GRID ** 2):
    latent = np.zeros((1, LATENT_DIM), dtype=np.float32)
    latent[0, dim_idx[0]] = v0.flatten()[pl_ix]
    latent[0, dim_idx[1]] = v1.flatten()[pl_ix]
    gen_data = gen_model(latent).numpy()[0]

    plt.subplot(N_GRID, N_GRID, pl_ix + 1)
    plt.plot(gen_data[:, sensor_ix])
    plt.axis('off')
    plt.ylim([-1, 1])

    # if (pl_ix + 1) % N_GRID == 1:
    #    plt.ylabel(f"{latent[0, dim_idx[1]]:.1f}")
    if pl_ix < N_GRID:
        plt.title(f"{latent[0, dim_idx[0]]:.1f}")

mus = []
sigmas = []
for class_id in np.unique(Y):
    b_class = Y == class_id
    tmp = np.mean(X[b_class], axis=0, keepdims=True)
    lat_dist = vae_model(tmp.astype(np.float32))[0]
    mus.append(lat_dist.mean().numpy())
    sigma = np.eye(4)
    np.fill_diagonal(sigma, lat_dist.variance())
    sigmas.append(sigma)
mus = np.squeeze(np.array(mus))
sigmas = np.squeeze(np.array(sigmas))

plot_range = [-2, 2]
grid_steps = 20
xx, yy = np.mgrid[plot_range[0]:plot_range[1]:(1 / grid_steps),
         plot_range[0]:plot_range[1]:(1 / grid_steps)]
positions = np.dstack((xx, yy))

from scipy.stats import multivariate_normal


def bivariate_contour(class_id=0, dim0=0, dim1=1):
    test_mus = [mus[class_id][dim0], mus[class_id][dim1]]
    test_sigmas = [[sigmas[class_id][dim0, dim0], 0], [0, sigmas[class_id][dim1, dim1]]]
    rv = multivariate_normal(test_mus, test_sigmas)
    return rv.pdf(positions)


fig = plt.figure(figsize=[8, 6])
for pair_ix, dim_pair in enumerate([[0, 1], [0, 2], [1, 2]]):
    ax = fig.add_subplot(2, 2, pair_ix + 1)
    zzs = np.zeros((positions.shape[:-1]))
    for class_id in range(5):
        dist = bivariate_contour(class_id=class_id, dim0=dim_pair[0], dim1=dim_pair[1])
        zzs += dist / np.max(dist)
    ax.contourf(xx, yy, zzs)
    ax.set_xlabel(f"Dim{dim_pair[0]}")
    ax.set_ylabel(f"Dim{dim_pair[1]}")
plt.tight_layout()
