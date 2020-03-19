import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from utils import PlotGmm, DataGenerator

# Generate Data
n=2000
dg = DataGenerator(n)
x, z = dg.generate_data()

# Plot
plot_gmm = PlotGmm()
plot_gmm.plot_data(x, z, dg.mu, dg.cov, dg.pi)


# Model
tfd = tfp.distributions
tfb = tfp.bijectors


class GmmLayer(tf.keras.layers.Layer):
    def __init__(self, k=2, *args, **kwargs):
        super(GmmLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        print(input_shape)
        initnorm = tf.random_normal_initializer()
        inituni = tf.random_uniform_initializer(0, 1)

        self.k = k

        self.covchol = tfp.util.TransformedVariable(
            tf.eye(input_shape[-1], batch_shape=(self.k,)),
            # inituni(shape=(self.k, 2, 2)),
            tfb.FillScaleTriL(),
            name='covchol', dtype=tf.float32
        )
        self.covchol_val = self.covchol.variables
        self.mu = tf.Variable(initnorm(shape=(self.k, input_shape[-1])), name='mu', dtype=tf.float32)
        self.pi = tf.Variable(initnorm(shape=(self.k,)), name='pi', dtype=tf.float32)

        self.bimix_gauss = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.pi),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=self.mu, 
                scale_tril=self.covchol
            )
        )


    def call(self, inputs):
        return self.bimix_gauss.log_prob(inputs)

    def get_params(self): 
        covchol_tensor = tf.convert_to_tensor(self.covchol)
        pi = tf.math.softmax(self.pi)
        return [t.numpy() for t in (self.mu, covchol_tensor, pi)]

class UpdatePlotCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):
    m = self.model.get_layer(name='gmm')
    mu, covchol, pi = m.get_params()
    cov = covchol @ covchol.transpose(0,2,1)
    plot_gmm.update_plot(mu, cov, pi, epoch)
    print(f'Training: {logs}')


k = 2
model = tf.keras.Sequential(GmmLayer(k, name='gmm'))

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.02),
              loss=lambda y_true, y_pred: -y_pred)
              
y = np.ones(x.shape[0])
model.fit(x, y,
          batch_size=100,
          epochs=100,
          verbose=0,
          steps_per_epoch=int(len(x)/100),
          callbacks=[UpdatePlotCallback()]
)


plt.show(block=True)


