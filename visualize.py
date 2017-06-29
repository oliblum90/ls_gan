import os
import matplotlib.pylab as plt
import inspect, re
import numpy as np
import cv2
import scipy.misc


def out(p):
    """
    function prints variable name and value. good for debugging
    """
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bout\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            print m.group(1), ":  ", p

            
def show_training_status(fname, epoch, d_loss, g_loss, d_test_loss):
    print "EPOCH: {}".format(epoch)
    print "--------------------------------------------------------"
    print "disc_loss: \t {} \t {}".format(d_loss, d_test_loss)
    print "gen_loss: \t {}".format(g_loss)
    print "--------------------------------------------------------"
    
    try:
        history = np.load(fname)
    except:
        history = {"d_loss"      : np.array([]),
                   "d_test_loss" : np.array([]),
                   "g_loss"      : np.array([])}
        history = np.array(history)
        
    history[()]["d_loss"] = np.append(history[()]["d_loss"], d_loss)
    history[()]["d_test_loss"] = np.append(history[()]["d_test_loss"], d_test_loss)
    history[()]["g_loss"] = np.append(history[()]["g_loss"], g_loss)
    
    np.save(fname, history)
    
            
def disp(img_list, title_list = None, fname = None, un_norm = False):
    """
    display a list of images
    """

    plt.figure()

    for idx, img in enumerate(img_list):

        plt.subplot(1, len(img_list), idx+1)
        if title_list is not None:
            plt.title(title_list[idx])
            
        if un_norm:
            img_show = img * 127.5 + 127.5
        else:
            img_show = img.copy()
            
        plt.imshow(img_show.astype(np.uint8), vmax = img.max(), vmin = img.min())
        plt.axis("off")
    
    if fname is not None:
        plt.savefig(fname)
        
    plt.show()
    
def disp_array(imgs, shape, fname = None, title = None):
    
    plt.figure()
    if title is not None:
        plt.title(title)
    row = shape[0]
    col = shape[1]
    for idx, img in enumerate(imgs):
        if idx + 1 > row * col:
            break
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        plt.axis("off")
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    
def make_gif(imgs, gif_fname, fps = 10):
    
    # make temp img dir
    os.mkdir("tmp")
    
    # save images
    fnames = []
    for idx, sample in enumerate(imgs):
        plt.figure()
        plt.imshow(sample.astype(np.uint8))
        plt.axis("off")
        fname = os.path.join("tmp", str(idx) + ".png")
        fnames.append(fname)
        plt.savefig(fname)
    
    # make gif
    import imageio
    images = []
    for filename in fnames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_fname, images, fps = fps)
    
    # delete images
    import shutil
    shutil.rmtree('tmp')
    
def save_g_imgs(fname, imgs, imgs_per_row = 10):
    imgs = imgs * 127.5 + 127.5
    n_rows = len(imgs) / imgs_per_row
    rows = []
    for i in range(n_rows):
        lower_idx = i * 10
        upper_idx = lower_idx + 10
        rows.append(np.concatenate(imgs[lower_idx : upper_idx], axis = 1))
    img = np.concatenate(rows)
    scipy.misc.imsave(fname, img)
    
    
    