import numpy as np
from PIL import Image
import os

class XYModelWolff:
    def __init__(self, L, J=1.0, T=0.9):
        self.L = L  # Lattice size
        self.J = J  # Interaction strength
        self.beta = 1 / T  # Inverse temperature
        self.spins = np.random.uniform(0, 2 * np.pi, (L, L))  # Random initial spins

    def cluster_flip(self):
        visited = np.zeros((self.L, self.L), dtype=bool)
        stack = []

        # Pick a random starting spin
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)
        start_angle = self.spins[i, j]
        cluster_angle = np.random.uniform(0, 2 * np.pi)

        stack.append((i, j))
        visited[i, j] = True

        while stack:
            x, y = stack.pop()
            old_angle = self.spins[x, y]
            new_angle = 2 * cluster_angle - old_angle - np.pi
            while new_angle < 0:
                new_angle += 2 * np.pi
            while new_angle >= 2 * np.pi:
                new_angle -= 2 * np.pi
            self.spins[x, y] = new_angle

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = (x + dx) % self.L, (y + dy) % self.L

                if not visited[nx, ny]:
                    neighbor_angle = self.spins[nx, ny]
                    p_target = 1 - np.exp(min(0.0, 2 * self.beta * self.J * np.cos(neighbor_angle - cluster_angle) * np.cos(new_angle - cluster_angle)))
                    if np.random.rand() < p_target:
                        stack.append((nx, ny))
                        visited[nx, ny] = True

    def run(self, steps=1000):
        for step in range(steps):
            if step % 100 == 0:
                print(f'Steps:{step} Energy:{self.get_energy()}')
            self.cluster_flip()

    def get_energy(self):
        # Calculate the total energy of the system
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = (i + dx) % self.L, (j + dy) % self.L
                    energy -= self.J * np.cos(self.spins[i, j] - self.spins[nx, ny])
        return energy / (2 * self.L * self.L)  # Normalize energy by the number of lattice sites

    def save_spins_as_image(self, step, save_dir):
        # Normalize spins to the range [0, 255] for grayscale image
        normalized_spins = (self.spins) / (2 * np.pi) * 255
        normalized_spins = normalized_spins.astype(np.uint8)
        
        # Create a PIL image
        img = Image.fromarray(normalized_spins, 'L')
        
        # Save image to the specified directory
        img.save(os.path.join(save_dir, f'spin_{step:04d}.png'))

# Example usage:
L = 128  # Lattice size
model = XYModelWolff(L, J=1.0, T=0.5)
model.run(steps=2500)

# Create directory to save images
save_dir = 'xy_model_images'
os.makedirs(save_dir, exist_ok=True)

# Run the model and save 300 images
for step in range(300):
    model.run(steps=50)  # Adjust the number of steps per image as needed
    model.save_spins_as_image(step, save_dir)
