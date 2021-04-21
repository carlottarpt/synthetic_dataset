# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.io
import os
import json
import matplotlib.pyplot as plt
from shapely.geometry import Point as pt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize
import scipy.optimize
from scipy.ndimage import gaussian_filter
import cv2
import imageio
import pandas as pd
from PIL import Image
from itertools import product
from skimage.draw import circle, circle_perimeter
from os.path import isfile, join

def activation_rise(m, t, b):
    return m * t + b


def activation_decay(t, intensity, delay, decay):
    return intensity * np.exp(-decay * (t - delay))


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def convert_frames_to_video(pathIn,pathOut,fps=1):

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = (500, 500, 1)
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def convert_frames_to_gif(pathIn,pathOut,fps=5):
    files = os.listdir(pathIn)
    path_save = pathOut + "video.gif"
    # assume that your images that you
    # want to make the GIF are under the my_images folder
    images_path = [os.path.join(pathIn, file) for file in files]

    # for sorting the file names properly
    #print(images_path[0][5:4])
    sort_nicely(images_path)
    #images_path.sort(key=lambda x: int(x[9:]))
    # fps are the frames per second
    images = []
    for img_p in images_path:

        img = imageio.imread(img_p)
        images.append(img)


    imageio.mimsave(path_save, images, fps=fps)

######################################## dataset #############################################

class Dataset:
    def __init__(self, image_size, background_color, n_sequences, n_frames, seed=0):
        self.image_size = image_size
        self.image = None
        self.seed = seed
        self.background_color = background_color
        self.n_sequences = n_sequences
        self.n_frames = n_frames        # frames per sequence

    def create_image(self):
        """ Returns image array filled with zeros of size imag_size set in constructor """
        self.image = np.ones(self.image_size) * self.background_color
        return self.image

    def create_sequences(self, neurons, neuron_coords, neuron_act): #, neuron_act_outer):
        sequence_set = []
        for n in range(self.n_sequences):
            image_set = []
            t = 0       # time step inside sequence
            for f in range(self.n_frames):
                self.image = self.create_image()
                self.image = self.plot_neurons(self.image, neurons, neuron_coords, neuron_act, t)
                #self.image = self.plot_neurons(self.image, neurons, neuron_coords, neuron_act, neuron_act_outer, t)
                self.image = self.apply_noise(self.image)
                image_set.append(self.image)
                t = t + 1

            sequence_set.append(image_set)

        return sequence_set

    # def plot_neurons(self, image, neurons, neuron_coords, neuron_act, neuron_act_outer, t):
    #
    #     for neuron_coord, neuron_a, neuron_a_o in zip(neuron_coords, neuron_act, neuron_act_outer):
    #         image_neurons = self.rasterize(neuron_coord, image, neuron_a, neuron_a_o, t)
    #
    #     return image_neurons

    def plot_neurons(self, image, neurons, neuron_coords, neuron_act, t):

        for neuron_coord, neuron_a in zip(neuron_coords, neuron_act):
            image_neurons = self.rasterize(neuron_coord, image, neuron_a, t)

        return image_neurons

    def rasterize(self, neuron, raster, intensity_curve, t):

        minx, miny, maxx, maxy = neuron.bounds
        # center_x = maxx - minx
        # center_y = maxy - miny
        # r = center_x - minx
        # r_outer = r * 1.3
        # coords_outer = points_in_circle(center_x, center_y, r_outer)
        minx, miny = np.floor(minx), np.floor(miny)
        maxx, maxy = np.ceil(maxx), np.ceil(maxy)
        for x in np.arange(minx, maxx):
            for y in np.arange(miny, maxy):
                if neuron.contains(pt(x, y)) and 0 < int(y) < raster.shape[0] and \
                        0 < int(x) < raster.shape[1]:
                    raster[int(y), int(x)] = max(raster[int(y), int(x)], intensity_curve[t])

        return raster

    def apply_noise(self, image):

        image = image / 255
        noise = np.random.normal(0, 0.06, (500, 500))
        image = gaussian_filter(image, 3)
        image = image + noise

        return image


class Neuron:
    def __init__(self, size_param=(2, 3), rise_length=4, decay_param=(0.2, 1.5),
                 intensity_mean=60, intensity_max=(160,200), active_value=(0.1), seed=0):

        self.active_value = np.random.normal(active_value, 0.00005)

        self.intensity_mean = np.random.normal(intensity_mean, 10)
        self.intensity_max = np.random.randint(intensity_max[0], intensity_max[1])
        self.intensity = np.random.randint(0, 255)
        self.size = np.random.uniform(size_param[0], size_param[1])
        self.rise = rise_length
        self.decay_length = np.random.uniform(decay_param[0], decay_param[1])


    def get_activation_function(self, n_frames, save_path, n_neurons, save, noise=True):



        nb_activations = int(n_frames * self.active_value)

        if nb_activations < 0:
            nb_activations = 0
            self.intensity_mean = 10

        intensities = np.full(n_frames + 151, self.intensity_mean)
        spikes = []

        spike_times = self.generate_spike_moments(n_frames, nb_activations)
        for n in range(nb_activations):
            spikes.append(self.create_single_activation())

        for spike, spike_t in zip(spikes, spike_times):
            i = 0
            for elem in spike:
                intensities[spike_t + i] = elem
                i += 1

        if len(intensities) > n_frames:
            intensities = intensities[:n_frames]
        else:
            intensities = np.pad(intensities, (0, n_frames- len(intensities)), "constant",
                   constant_values=self.intensity_mean)
        if noise:
            noisy = np.random.normal(0, 5, n_frames)
            intensities = intensities + noisy
        save_path_neuron_act = save_path + "/traces/"
        if not os.path.exists(save_path_neuron_act):
            os.makedirs(save_path_neuron_act)
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(np.arange(0, n_frames), intensities)

        if save:
            fig.savefig(save_path_neuron_act + f"/neuron_{n_neurons}.png")
        plt.close(fig)
        return intensities

    def generate_spike_moments(self, n_frames, nb_activations):
        time_sequence = np.arange(0, n_frames-1)
        return np.random.choice(time_sequence, nb_activations)


    def create_single_activation(self):
        activation=[]
        activation_intensity = self.intensity_max - np.random.randint(0, 50)

        m = activation_intensity / (self.rise - 1)
        rise = self.activation_rise(m, np.arange(self.rise))
        decay_length = np.random.randint(100, 130)
        decay = self.activation_decay(np.arange(decay_length), activation_intensity,
                                      self.decay_length, 0,
                                      self.intensity_mean)
        for i in rise:
            activation.append(i)
        for i in decay[1:len(decay)]:
            activation.append(i)

        return activation

    def activation_rise(self, m, t, b=None):
        """ returns one single rise """
        if b is None:
            b = self.intensity_mean
        return m * t + b

    def activation_decay(self, t, intensity, decay, delay, offset):
        """ returns one single decay """
        return intensity * np.exp(-decay * (t - delay)) / np.exp(-decay * (t[0] - delay)) + offset

    def smoothListGaussian(self, list, degree=5):
        window = degree * 2 - 1
        weight = np.array([1.0] * window)
        weightGauss = []

        for i in range(window):
            i = i - degree + 1
            frac = i / float(window)
            gauss = 1 / (np.exp((4 * (frac)) ** 2))
            weightGauss.append(gauss)
        weight = np.array(weightGauss) * weight
        smoothed = [0.0] * (len(list) - window)
        for i in range(len(smoothed)):
            smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)
        return smoothed

    def get_neuron_shape(self, edge_smooth_factor=18):
        """ returns polygon shapely object """
        # Generate the polygon
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        r = np.random.poisson(3, 100)

        r = np.pad(r, (edge_smooth_factor - 1, edge_smooth_factor), mode='wrap')

        r = self.smoothListGaussian(r, degree=edge_smooth_factor)
        r = r / np.mean(r) * self.size
        coords = zip(np.cos(theta) * r, np.sin(theta) * r)

        neuron_shape = Polygon(coords)

        return neuron_shape

    def get_neuron_coord(self, edge_smooth_factor=18):
        #returns coords of neuron shape and position in 2D image

        # create neuron shape
        neuron = self.get_neuron_shape(edge_smooth_factor)

        # get polygon exterior shape coord
        xy = np.array(neuron.exterior.coords.xy)

        # apply vector to shape to change position in image
        # question: should this really be gaussian distributed?
        vec_x = np.random.normal(250, 150)
        vec_y = np.random.normal(250, 150)
        scale = np.random.normal(1.8, 0.1)
        x_list = xy[0]
        y_list = xy[1]
        xy_polar = cart2pol(x_list, y_list)
        xy_polar[0] *= scale
        x_list, y_list = pol2cart(*xy_polar)
        x_new = [x * scale + vec_x for x in x_list]
        y_new = [y * scale + vec_y for y in y_list]
        xy_new = [x_new, y_new]

        neuron_coord = Polygon(np.array(xy_new).T)

        return neuron_coord


def add_background( img, snr=0.2, rotate_angle=0):
    high_pass_mat = scipy.io.loadmat('/home/user/Documents/github-projects/'
                                     'MovieAnalysis/preprocessing/'
                                     'residuals/filtered_highpass_1.mat')
    high_pass = high_pass_mat['inputFiltered']
    high_pass_n = (high_pass - np.min(high_pass)) / (np.max(high_pass) - np.min(high_pass))
    high_pass_n = scipy.ndimage.rotate(high_pass_n, rotate_angle)
    low_pass_mat = scipy.io.loadmat('/home/user/Documents/github-projects/'
                                     'MovieAnalysis/preprocessing/'
                                     'residuals/filtered_lowpass_1.mat')
    low_pass = low_pass_mat['inputFiltered']
    low_pass_n = (low_pass - np.min(low_pass)) / (np.max(low_pass) - np.min(low_pass))
    low_pass_n = scipy.ndimage.rotate(low_pass_n, rotate_angle)
    img = img* (1-snr)+ (low_pass_n -np.mean(low_pass_n))* snr + \
          (high_pass_n - np.mean(high_pass_n))
    return img

def extract_labels(neuron_coordinates, neuron_activities, imgs, param, save_path):

    map = np.zeros((500, 500))
    coord_all=[]
    for neuron_coord_single in neuron_coordinates:
        coord_single = list(neuron_coord_single.exterior.coords)
        for (x, y) in coord_single:
            if x >= 500 or y >= 500:
                pass
            else:
                map[int(x), int(y)] = 1

        coord_all.append(coord_single)
    fig, ax = plt.subplots(figsize=(8, 8))
    # fig.add_subplot(rows, columns, i)
    plt.imshow(map)
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)

    fig.savefig(save_path + f"/coords_map.png")
    plt.close(fig)
    map = map.tolist()
    neuron_act = [x.tolist() for x in neuron_activities]

    dict = {"neuron_coordinates": coord_all,
            "neuron_activities": neuron_act,
            "images": imgs,
            "parameters": param,
            "map": map}
    #df = pd.DataFrame.from_dict(dict).to_json(save_path + '/data.json')
    with open(save_path + '/data.json', 'w') as fp:
        json.dump(dict, fp)

if __name__ == '__main__':
    # saving directory
    sd = "/home/user/Documents/synthetic_datasets/"
    save = True
    # set parameters dataset
    n_datasets = 50
    n_sequences = 1
    sequence_length = 500
    image_size = (500, 500)

    # background parameters:
    background_snr = float(np.random.choice([0.1, 0.2, 0.3]))
    background_rotation = int(np.random.choice([0, 90, 180, 270]))
    background_color = int(np.random.choice([20, 40, 30]))

    random_seed = np.random.seed()

    # set parameters neurons
    nb_neurons = int(np.random.choice([150, 170, 200, 250]))
    neuron_size = (2,3)
    edge_smooth_factor = int(np.random.choice([16, 18, 20]))
    rise_length = int(np.random.choice([3, 6, 10]))
    decay_param = (0.05, 0.1, 0.2)
    intensity_mean = int(np.random.choice([20, 30, 40]))
    intensity_max = (140, 190)
    active_value = float(np.random.choice([0.005, 0.01, 0.02]))
    noise = True

    # create neurons
    neurons = []
    neuron_coord = []
    neuron_act = []
    i=4

    for i in range(n_datasets):

        save_path = sd + f"sequence_{i}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        params = {"image_size": image_size,
                  "background_snr": background_snr,
                  "background_rotaion": background_rotation,
                  "background_color": background_color,
                  "nb_neurons": nb_neurons,
                  "neuron_size": neuron_size,
                  "rise_length": rise_length,
                  "decay_param": decay_param,
                  "intensity_mean": intensity_mean,
                  "intensity_max": intensity_max,
                  "active_value": active_value,
                  "noise_trace": noise}

        for n in range(nb_neurons):
            neuron = Neuron(size_param=neuron_size, rise_length=rise_length, decay_param=decay_param,
                            intensity_mean=intensity_mean, intensity_max=intensity_max, active_value=active_value)
            neurons.append(neuron)
            neuron_coord.append(neuron.get_neuron_coord(edge_smooth_factor=edge_smooth_factor))
            neuron_act.append(neuron.get_activation_function(sequence_length, save_path, n, save=save, noise=noise))
            #neuron_act_outer.append(neuron.get_activation_function(sequence_length, background_color)[1])

        # create dataset
        dataset = Dataset(image_size=image_size, background_color=background_color, n_sequences=n_sequences, n_frames=sequence_length, seed=random_seed)
        sequence_set = dataset.create_sequences(neurons, neuron_coord, neuron_act)
        image_set = sequence_set[0]

        # plot and save images
        columns = sequence_length
        rows = 1
        imgs = []

        save_path_imgs = save_path + "/images"
        if not os.path.exists(save_path_imgs):
            os.makedirs(save_path_imgs)

        for j in range(columns):
            img = image_set[j]
            img = add_background(img, snr=background_snr, rotate_angle=background_rotation)
            imgs.append(img.tolist())
            fig, ax = plt.subplots(figsize=(8, 8))
            # fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
            fig.axes[0].get_xaxis().set_visible(False)
            fig.axes[0].get_yaxis().set_visible(False)

            if save:
                fig.savefig(save_path_imgs + f"/frame{j}.jpg")
            plt.close(fig)
        save_path_video = save_path + "/video/"
        if not os.path.exists(save_path_video):
            os.makedirs(save_path_video)
        extract_labels(neuron_coord, neuron_act, imgs, params, save_path)
        convert_frames_to_gif(save_path + "/images/", save_path + "/video/")
        i += 1
        plt.close('all')