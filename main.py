# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from shapely.geometry import Point as pt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize
import scipy.optimize
from scipy.ndimage import gaussian_filter
from itertools import product
from skimage.draw import circle, circle_perimeter

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

def points_in_circle(x, y, radius):
    for x, y in product(range(int(radius) + 1), repeat=2):
        if x**2 + y**2 <= radius**2:
            yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))

def points_circle(center_x, center_y, radius):
    print(circle(center_x, center_y, radius))
    print(circle_perimeter( center_x, center_y, radius, shape=(500,500)))

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

    def create_sequences(self, neurons, neuron_coords, neuron_act, neuron_act_outer):
        sequence_set = []
        for n in range(self.n_sequences):
            image_set = []
            t = 0       # time step inside sequence
            for f in range(self.n_frames):
                self.image = self.create_image()
                self.image = self.plot_neurons(self.image, neurons, neuron_coords, neuron_act, neuron_act_outer, t)
                self.image = self.apply_noise(self.image)
                image_set.append(self.image)
                t = t + 1

            sequence_set.append(image_set)

        return sequence_set

    def plot_neurons(self, image, neurons, neuron_coords, neuron_act, neuron_act_outer, t):

        for neuron_coord, neuron_a, neuron_a_o in zip(neuron_coords, neuron_act, neuron_act_outer):
            image_neurons = self.rasterize(neuron_coord, image, neuron_a, neuron_a_o, t)

        return image_neurons

    def rasterize(self, neuron, raster, intensity_curve, intensity_curve_outer, t):

        minx, miny, maxx, maxy = neuron.bounds
        center_x = maxx - minx
        center_y = maxy - miny
        r = center_x - minx
        r_outer = r * 1.3
        coords_outer = points_in_circle(center_x, center_y, r_outer)
        minx, miny = np.floor(minx), np.floor(miny)
        maxx, maxy = np.ceil(maxx), np.ceil(maxy)
        for x in np.arange(minx, maxx):
            for y in np.arange(miny, maxy):
                if neuron.contains(pt(x, y)) and 0 < int(y) < raster.shape[0] and \
                        0 < int(x) < raster.shape[1]:
                    raster[int(y), int(x)] = intensity_curve[t]

        X = int(r_outer)  # R is the radius
        for x in range(-X, X + 1):
            Y = int((r_outer * r_outer - x * x) ** 0.5)  # bound for y given x
            for y in range(-Y, Y + 1):
                raster[int(y), int(x)] = intensity_curve_outer[t]

        return raster

    def apply_noise(self, image):

        image = image / 255
        noise = np.random.normal(0, 0.06, (500, 500))
        image = gaussian_filter(image, 3)
        image = image + noise

        return image


class Neuron:
    def __init__(self, seed=0):

        self.active_value = np.random.normal(0.2, 0.05) # between inactive and 50 % of the sequence active

        self.intensity_mean = np.random.normal(60, 20)
        self.intensity_max = np.random.randint(160, 200)
        self.intensity_min = np.random.randint(0, 100)
        self.intensity = np.random.randint(0, 255)
        self.size = np.random.randint(2, 3)
        self.rise = 3
        self.decay_length = np.random.uniform(1, 2)


    def get_activation_function(self, n_frames, background_color):


        intensities = np.full(n_frames + 20, self.intensity_mean)
        nb_activations = int(n_frames * self.active_value)
        intensities_outer = np.full(n_frames + 20, background_color)

        # get single activations and spike moments
        spikes = []

        spike_times = self.generate_spike_moments(n_frames, nb_activations)
        for n in range(nb_activations):
            spikes.append(self.create_single_activation())

        for spike, spike_t in zip(spikes, spike_times):
            i = 0
            for elem in spike:
                intensities[spike_t + i] = elem
                i += 1
            intensities_outer[spike_t + self.rise] = self.intensity_max

        if len(intensities) > n_frames:
            intensities = intensities[:n_frames]
        else:
            intensities = np.pad(intensities, (0, n_frames- len(intensities)), "constant",
                   constant_values=self.intensity_mean)

        if len(intensities_outer) > n_frames:
            intensities_outer = intensities_outer[:n_frames]
        else:
            intensities_outer = np.pad(intensities_outer, (0, n_frames- len(intensities_outer)), "constant",
                   constant_values=background_color)

        #print(intensities)
        plt.plot(np.arange(0, n_frames), intensities_outer)
        plt.show()
        return intensities, intensities_outer

    def generate_spike_moments(self, n_frames, nb_activations):
        time_sequence = np.arange(0, n_frames-1)
        return np.random.choice(time_sequence, nb_activations)


    def create_single_activation(self):
        activation=[]
        activation_intensity = self.intensity_max - np.random.randint(0, 50)

        m = activation_intensity / (self.rise - 1)
        rise = self.activation_rise(m, np.arange(self.rise))
        decay_length = np.random.randint(6, 12)
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

    def get_neuron_coord(self):
        #returns coords of neuron shape and position in 2D image

        # create neuron shape
        neuron = self.get_neuron_shape()

        # get polygon exterior shape coord
        xy = np.array(neuron.exterior.coords.xy)

        # apply vector to shape to change position in image
        # question: should this really be gaussian distributed?
        vec_x = np.random.normal(250, 110)
        vec_y = np.random.normal(250, 110)
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


def add_background( img, snr=0.2):
    high_pass_mat = scipy.io.loadmat('/home/user/Documents/github-projects/'
                                     'MovieAnalysis/preprocessing/'
                                     'residuals/filtered_highpass_1.mat')
    high_pass = high_pass_mat['inputFiltered']
    high_pass_n = (high_pass - np.min(high_pass)) / (np.max(high_pass) - np.min(high_pass))
    low_pass_mat = scipy.io.loadmat('/home/user/Documents/github-projects/'
                                     'MovieAnalysis/preprocessing/'
                                     'residuals/filtered_lowpass_1.mat')
    low_pass = low_pass_mat['inputFiltered']
    low_pass_n = (low_pass - np.min(low_pass)) / (np.max(low_pass) - np.min(low_pass))

    img = img* (1-snr)+ (low_pass_n -np.mean(low_pass_n))* snr + \
          (high_pass_n - np.mean(high_pass_n))
    return img

if __name__ == '__main__':

    # set parameters dataset
    n_sequences = 1
    sequence_length = 100
    background_color = 60
    image_size = (500, 500)
    random_seed = np.random.seed()

    # set parameters neurons
    nb_neurons = 150

    print(list(points_in_circle(4, 4, 3)))

    # create neurons
    neurons = []
    neuron_coord = []
    neuron_act = []
    neuron_act_outer = []

    for n in range(nb_neurons):
        neuron = Neuron()
        neurons.append(neuron)
        neuron_coord.append(neuron.get_neuron_coord())
        neuron_act.append(neuron.get_activation_function(sequence_length, background_color)[0])
        neuron_act_outer.append(neuron.get_activation_function(sequence_length, background_color)[1])

    # create dataset
    dataset = Dataset(image_size=image_size, background_color=background_color, n_sequences=n_sequences, n_frames=sequence_length, seed=random_seed)
    sequence_set = dataset.create_sequences(neurons, neuron_coord, neuron_act, neuron_act_outer)
    image_set = sequence_set[0]

    # plot and save images
    columns = sequence_length
    rows = 1
    save = True
    for i in range(columns):
        img = image_set[i]
        img = add_background(img)
        fig, ax = plt.subplots(figsize=(8, 8))
        # fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        if save:
            fig.savefig(f"image_{i}.png")
        plt.show()

