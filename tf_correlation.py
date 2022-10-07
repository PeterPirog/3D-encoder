import tensorflow as tf
import tensorflow_probability as tfp

z=tf.random.normal(shape=(1000, 4))
n,m=tf.shape(z)



corr_matrix = tfp.stats.correlation(z)
corr_matrix2=tf.linalg.band_part( corr_matrix, num_lower=0, num_upper=-1)
corr_matrix2=tf.linalg.set_diag(corr_matrix2, tf.zeros([m]))
corr_matrix2=tf.math.abs(corr_matrix2)
n_nonzero=tf.math.count_nonzero(corr_matrix2,dtype=tf.dtypes.float32)
sum=tf.reduce_sum(corr_matrix2)

print (corr_matrix)
print (corr_matrix2)
print(sum/n_nonzero)



cov_matrix = tfp.stats.covariance(z)
print (corr_matrix)
cov_matrix=tf.linalg.band_part(cov_matrix, num_lower=0, num_upper=-1) #get upper part
cov_matrix=tf.linalg.set_diag(cov_matrix, tf.zeros([m])) # zeroing diagonal
cov_matrix=tf.math.abs(cov_matrix) #get abs values of covariance
n_nonzero=tf.math.count_nonzero(cov_matrix,dtype=tf.dtypes.float32)
sum=tf.reduce_sum(cov_matrix)
loss=sum/n_nonzero

print (cov_matrix)
print(loss)