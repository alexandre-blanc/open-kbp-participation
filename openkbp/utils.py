import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

def visualize_3d_image(X):
    assert(len(X.shape) == 3)
    assert(X.shape[0] == X.shape[1])
    assert(X.shape[0] == X.shape[2])

    Nx = X.shape[0]

    fig, ax = plt.subplots(1,3, num='anim')
    im0 = ax[0].imshow(X[0,:,:])
    ax[0].set_title('x')
    ax[0].axis('off')
   
    im1 = ax[1].imshow(X[:,0,:])
    ax[1].set_title('y')
    ax[1].axis('off')
   
    im2 = ax[2].imshow(X[:,:,0])
    ax[2].set_title('z')
    ax[2].axis('off')

    def animate(i):

        im0.set_data(X[i,:,:])
        im1.set_data(X[:,i,:])
        im2.set_data(X[:,:,i])
        plt.suptitle('i = {}'.format(i))
        return im0, im1, im2
    
    anim = animation.FuncAnimation(fig, animate, frames=Nx, interval=20, repeat=False)
    rc('animation', html='jshtml')
    plt.close()
    
    return anim