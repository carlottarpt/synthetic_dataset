# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.io
import os
import json
from scipy.stats import skewnorm
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
import glob
from PIL import Image
from itertools import product
from skimage.draw import circle, circle_perimeter
from os.path import isfile, join
import h5py

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
    # assume that your images that youcreate_single_activation
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
    def __init__(self, image_size, background_color, background_snr,  n_sequences, n_frames, seed=0):
        self.image_size = image_size
        self.image = None
        self.seed = seed
        self.background_color = background_color
        self.n_sequences = n_sequences
        self.n_frames = n_frames        # frames per sequence

        self.raster_mask = np.zeros((500, 500))
        self.sin = 0.05 * np.sin(np.arange(0, n_frames/0.01, 0.01))
    def create_image(self):
        """ Returns image array filled with zeros of size imag_size set in constructor """
        self.image = np.ones(self.image_size) * self.background_color
        return self.image

    def create_sequences(self, neuron_coords, neuron_act,  neuron_coords_dark=None): #, neuron_act_outer):
        sequence_set = []
        for n in range(self.n_sequences):
            image_set = []
            t = 0       # time step inside sequence
            for f in range(self.n_frames):

                print(f"Processing frame {f} of {self.n_frames}")
                self.image = self.create_image()
                self.image = self.plot_neurons(self.image, neuron_coords, neuron_act, t)
                #self.image = self.plot_neurons(self.image, neurons, neuron_coords, neuron_act, neuron_act_outer, t)

                if neuron_coords_dark is not None:
                    self.image = self.plot_dark_neurons(self.image, neuron_coords_dark, t)

                self.image = self.apply_noise(self.image)
                self.image = add_background(self.image, snr=background_snr, rotate_angle=background_rotation)
                self.add_drift(t)
                image_set.append(self.image)
                t = t + 1


            sequence_set.append(image_set)

        return sequence_set

    def add_drift(self, t):

        self.image = self.image + self.sin[t]

    def plot_neurons(self, image, neuron_coords, neuron_act, t):

        for neuron_coord, neuron_a in zip(neuron_coords, neuron_act):
            image_neurons = self.rasterize(neuron_coord, image, neuron_a, t)

        return image_neurons

    def rasterize(self, neuron, raster, intensity_curve, t, dark=False):

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

                    #if self.raster_mask[int(x), int(y)] == 0:
                    raster[int(y), int(x)] = intensity_curve[t]
                    # else:
                    #     if raster[int(y), int(x)] <= intensity_curve[t]:
                    #         raster[int(y), int(x)] = intensity_curve[t]
                    #
                    # self.raster_mask[int(x), int(y)] = 1
                    # print(self.raster_mask)

        return raster

    def apply_noise(self, image):

        image = image / 255
        noise = np.random.normal(0, 0.07, (500, 500))
        image1 = gaussian_filter(image, 5)
        image2 = gaussian_filter(image, 10)
        image = 0.9*image1 + 0.1*image2 + noise
        return image


class Neuron:
    def __init__(self, size_param=(2, 3), rise_length=4, decay_param=(0.2, 1.5),
                 intensity_mean=60, intensity_max=(160,200), active_value=(0.1), seed=0):

        self.active_value = np.random.normal(active_value, 0.00005)

        self.intensity_mean = intensity_mean
        self.intensity_max = np.random.randint(intensity_max[0], intensity_max[1])
        self.intensity = np.random.randint(0, 255)
        self.size = np.random.uniform(size_param[0], size_param[1])
        self.rise = rise_length
        self.decay_length = np.random.uniform(decay_param[0], decay_param[1])

        self.activation_events = None

    def get_activation_function(self, n_frames, save_path, n_neurons, save, noise=True):



        nb_activations = int(n_frames * self.active_value)
        nb_activations_neg = int(n_frames * self.active_value * 0.01 )
        print(nb_activations, nb_activations_neg)

        # if nb_activations < 0:
        #     nb_activations = 0
        #
        # if nb_activations_neg < 0:
        #     nb_activations_neg = 0
        intensities = np.full(n_frames + 151, self.intensity_mean)
        spikes = []
        spikes_neg = []
        spike_times = self.generate_spike_moments(n_frames, nb_activations)
        print("spike times:", spike_times)
        spike_times_neg = self.generate_spike_moments(n_frames, nb_activations_neg)
        # for ground truth
        self.activation_events = np.array(spike_times) + self.rise
        self.activation_events = [x for x in self.activation_events if x <= n_frames]


        for i in range(nb_activations):
            spikes.append(self.create_single_activation())
        for i in range(nb_activations_neg):
            spikes_neg.append(self.create_single_activation(sign_pos=False))

        for spike, spike_t in zip(spikes, spike_times):
            i = 0
            for elem in spike:
                intensities[spike_t + i] = elem
                i += 1

        for spike, spike_t in zip(spikes_neg, spike_times_neg):
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
            noisy = np.random.normal(0, 10, n_frames)
            intensities = intensities + noisy
        save_path_neuron_act = save_path + "/traces/"
        if not os.path.exists(save_path_neuron_act):
            os.makedirs(save_path_neuron_act)

        #plot calcium traces
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(np.arange(0, n_frames), intensities)

        if save:
            fig.savefig(save_path_neuron_act + f"/neuron_{n_neurons}.png")
        plt.show()
        plt.close(fig)
        return intensities

    def generate_spike_moments(self, n_frames, nb_activations):
        time_sequence = np.arange(0, n_frames-1)
        return np.random.choice(time_sequence, int(nb_activations))

    def get_neuron_events(self):
        return self.activation_events

    def draw_skewnorm(self, skewness=1, boundaries=(-0.75, 1)):
        r = np.inf
        while (r < boundaries[0]) or (r > boundaries[1]):
            r = skewnorm.rvs(skewness, size=1)
        r = (r - boundaries[0]) / (boundaries[1] - boundaries[0]) * 400
        return r[0]


    def create_single_activation(self, sign_pos=True):
        my_list = [True] * 50 + [False] * 50
        hold_on_top = np.random.choice(my_list)

        activation=[]
        activation_intensity = self.draw_skewnorm(5)
        if sign_pos:
            m = activation_intensity / (self.rise - 1)
        else:
            activation_intensity /= 5
            activation_intensity = - activation_intensity
            if activation_intensity < 0:
                activation_intensity = 0
            m = activation_intensity / (self.rise - 1)
        rise = self.activation_rise(m, np.arange(self.rise))
        decay_length = np.random.randint(35, 85)
        decay = self.activation_decay(np.arange(decay_length), activation_intensity,
                                      self.decay_length, 0,
                                      self.intensity_mean)
        for i in rise:
            activation.append(i)

        if sign_pos:
            if hold_on_top:
                hold_on_top_length = int(np.random.uniform(4, 8))
                for i in range(hold_on_top_length):
                    activation.append(activation_intensity)
        else:
            hold_on_top_length = int(np.random.uniform(20,45))
            for i in range(hold_on_top_length):
                activation.append(activation_intensity)
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
        vec_x = np.random.uniform(0, 500)
        vec_y = np.random.uniform(0, 500)

        # vec_x = np.random.normal(250, 150)
        # vec_y = np.random.normal(250, 150)

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

    background_imgs = glob.glob('/home/user/Documents/utils/filtered/*.jpg')

    background_smoothed_index = np.random.randint(0, 99)
    background_smoothed_path = background_imgs[background_smoothed_index]
    background_smoothed = np.array( cv2.imread(background_smoothed_path))
    background_smoothed = cv2.cvtColor(background_smoothed, cv2.COLOR_BGR2GRAY) / 255
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
    img = img* (1-snr) + (low_pass_n -np.mean(low_pass_n))* snr + \
          (high_pass_n - np.mean(high_pass_n))
    #img = img * (0.8) + (background_smoothed - np.mean(background_smoothed)) * 0.2

    # try contrast
    #img = img * 0.8
    return img

def extract_labels(neuron_coordinates, neuron_activities, neuron_events,imgs, param, save_path):

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


    map = map.tolist()
    neuron_act = [x.tolist() for x in neuron_activities]
    neuron_act_new = []
    for act in neuron_act:
        act_n = [int(x) for x in act]
        neuron_act_new.append(act_n)

    neuron_evt_new = []
    for evt in neuron_events:
        evt_n = [int(x) for x in evt]
        neuron_evt_new.append(evt_n)

   
    dict = {"neuron_coordinates": coord_all,
            "neuron_activities": neuron_act_new,
            "neuron_events": neuron_evt_new,
            "parameters": param,
            "map": map}
    with open(save_path + '/data.json', 'w') as fp:
        json.dump(dict, fp)

if __name__ == '__main__':

    random_seed = np.random.seed(16)
    # saving directory
    sd = "/home/user/Documents/synthetic_datasets/"
    save = True
    # set parameters dataset/home/user/Documents/synthetic_datasets/sequence_6/video/extracted/resultsPCAICA/new
    n_datasets = 1
    n_sequences = 1
    sequence_length = 800
    image_size = (500, 500)

    # background parameters:
    background_snr = float(np.random.choice([ 0.3,0.25]))
    background_rotation = int(np.random.choice([0, 90, 180, 270]))
    background_color = 50



    # set parameters neurons
    nb_neurons = 600
    neuron_size = (1.5,2.5)
    edge_smooth_factor = int(np.random.choice([16, 18, 20]))
    rise_length = int(np.random.choice([2,4]))
    # the smaller, the slower the decay
    decay_param = (0.05, 0.7)
    intensity_mean = 50
    #intensity_mean = int(np.random.choice([20, 30, 40]))
    #intensity_max = (190, 300)
    #active_value = float(np.random.choice([ 0.01, 0.02]))
    active_value = 0.2



    noise = True

    # create neurons

    neuron_coord = []

    neuron_act = []
    neuron_events = []


    for i in range(n_datasets):

        save_path = sd + f"sequence_11"
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
                  "active_value": active_value,
                  "noise_trace": noise}

        # create neurons
        for n in range(nb_neurons):
            neuron = Neuron(size_param=neuron_size, rise_length=rise_length, decay_param=decay_param,
                            intensity_mean=intensity_mean, active_value=active_value)

            neuron_coord.append(neuron.get_neuron_coord(edge_smooth_factor=edge_smooth_factor))
            print(f"Get activation function of neuron {n} of {nb_neurons}")
            neuron_act.append(neuron.get_activation_function(sequence_length, save_path, n, save=save, noise=noise))
            neuron_events.append(neuron.get_neuron_events())

        # show neuron ROI map
        map = np.zeros((500, 500))
        coord_all = []
        for neuron_coord_single in neuron_coord:
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
        if save:

            map = map * 255
            im = Image.fromarray(map)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(save_path + "/coord_map.jpeg")


        # create dataset
        dataset = Dataset(image_size=image_size, background_color=background_color, background_snr=background_snr, n_sequences=n_sequences, n_frames=sequence_length, seed=random_seed)
        sequence_set = dataset.create_sequences( neuron_coord, neuron_act)
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
            imgs.append(img.tolist())
            if save:
                #no rescaling since I strated to do it too late and now it fucks up my parameters.
                #img = (img - np.min(image_set)) / (np.max(image_set)- np.min(image_set))

                img = img * 255
                im = Image.fromarray(img)
                if im.mode != 'RGB':

                    im = im.convert('RGB')
                im.save(save_path_imgs + f"/frame{j}.jpeg")

            # fig, ax = plt.subplots(figsize=(8, 8))
            # plt.imshow(img, cmap="gray", vmin=0, vmax=255)
            # fig.axes[0].get_xaxis().set_visible(False)
            # fig.axes[0].get_yaxis().set_visible(False)
            # plt.show()
            # plt.close(fig)

        save_path_video = save_path + "/video/"
        if not os.path.exists(save_path_video):
            os.makedirs(save_path_video)
        print(f"Extracting and saving labels as .json file")
        extract_labels(neuron_coord, neuron_act, neuron_events, imgs, params, save_path)
        print("Converting frames to .h5 movie")
        hf = h5py.File(save_path_video + 'movie.h5', 'w')
        hf.create_dataset('1', data=np.array(image_set))
        hf.close()
        print("Converting frames to gif")
        convert_frames_to_gif(save_path + "/images/", save_path + "/video/")
        i += 1
        print("Successfully created data set!")
        print("Saved to:\n" + save_path)
        plt.close('all')