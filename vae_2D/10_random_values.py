import tensorflow as tf
# https://medium.com/the-owl/creating-a-tf-dataset-using-a-data-generator-5e5564609e64
# https://jackd.github.io/posts/deterministic-tf-part-2/

if __name__ == "__main__":
    dataset = tf.data.Dataset.random(seed=4).take(10)
    dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([5, 2]))

   # print(dataset )
    for item in dataset:
        pass
        #print(item)

    def gen():
        out=tf.random.normal(
            shape=(1,10), #first value is
            mean=0.0,
            stddev=1.0,
            dtype=tf.dtypes.float32,
            seed=None,
            name=None
        )
        return out


    dataset = tf.data.Dataset.from_generator(generator=gen,
                                             output_types=tf.float32).batch(3)

    print('\n',dataset)
    for item in dataset:
        print(item)
