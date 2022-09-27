
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
# Cube size
N=64

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    def randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin) * np.random.rand(n) + vmin


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    # Import libraries

    #from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt

    # Define Data

    x = np.arange(0, 20, 0.2)
    y = np.sin(x)
    z = np.cos(x)

    # Create Figure

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Create Plot

    ax.scatter3D(x, y, z)

    # Show plot

    plt.show()



