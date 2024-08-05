import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import numba

@numba.njit
def get_item_z(z, w, h, x, y):
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0
    return z[x][y]

@numba.njit
def get_pair_energy(z, w, h, val, x, y):
    return -np.cos(val - get_item_z(z, w, h, x-1, y)) - np.cos(val - get_item_z(z, w, h, x+1, y)) - np.cos(val - get_item_z(z, w, h, x, y-1)) - np.cos(val - get_item_z(z, w, h, x, y+1))

@numba.njit
def get_tot_energy(z, w, h):
    sum = 0.0
    for x in range(w):
        for y in range(h):
            sum += get_pair_energy(z, w, h, z[x][y], x, y)
    sum /= 2.0
    return sum

@numba.njit
def metropolis_goto(z, w, h, delta, max_epoch=10000, T=1):
    old_energyt = get_tot_energy(z, w, h)
    for epoch in range(max_epoch):
        for x in range(w):
            for y in range(h):
                new_z = np.random.uniform(z[x][y] - delta, z[x][y] + delta)
                if new_z < 0:
                    new_z += 2 * np.pi
                if new_z >= 2 * np.pi:
                    new_z -= 2 * np.pi
                old_energy = get_pair_energy(z, w, h, z[x][y], x, y)
                new_energy = get_pair_energy(z, w, h, new_z, x, y)
                if new_energy < old_energy or np.random.uniform(0, 1) < np.exp((-new_energy + old_energy) / T):
                    z[x][y] = new_z
        new_energyt = get_tot_energy(z, w, h)
        if np.abs(-(new_energyt - old_energyt)) / np.abs(old_energyt) < 1e-4:
            break
        old_energyt = new_energyt
        # print(f'EPOCH: {epoch}, ENERGY: {new_energyt}')
        # print(new_energyt)

class lattice():
    def __init__(self, width=128, height=128, delta=np.pi/2):
        self.w = width
        self.h = height
        self.z = np.random.rand(self.w, self.h) * 2 * np.pi
        self.delta = delta

    def metropolis_goto(self, max_epoch=int(1e4), T=int(1)):
        metropolis_goto(self.z, self.w, self.h, self.delta, max_epoch, T)

    def show(self):
        plt.figure()
        plt.imshow(self.z)
        plt.show()

z = lattice()
# z.metropolis_goto(max_epoch=100, T=0.1)
# z.show()
to_pil = transforms.ToPILImage()
num = 0
for i in range(500):
    print(f"low:{i}")
    z = lattice()
    z.metropolis_goto(max_epoch=100, T=0.1)
    tensor = z.z
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
    tensor = torch.from_numpy(tensor)
    image = to_pil(tensor)
    image.save(f"./diffusion_val/1eminus1/{num}_low.png")
    num += 1

num = 0
for i in range(50):
    print(f"high:{i}")
    z = lattice()
    z.metropolis_goto(max_epoch=100, T=100)
    tensor = z.z
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
    tensor = torch.from_numpy(tensor)
    image = to_pil(tensor)
    image.save(f"./database_2binary/val/high/{num}_high.png")
    num += 1