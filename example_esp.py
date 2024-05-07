import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def check_symmetric_and_traceless(Q):
    """
    Check if the 3x3 matrix Q is symmetric and traceless.

    Parameters:
    Q : numpy.ndarray
        The 3x3 matrix to be checked.

    Returns:
    tuple of bool
        Returns (is_symmetric, is_traceless), where each is a boolean indicating the property.
    """
    # Check symmetry: Q should be equal to its transpose
    is_symmetric = np.allclose(Q, Q.T)

    # Check traceless: Trace of Q should be zero
    is_traceless = np.isclose(np.trace(Q), 0)

    return (is_symmetric, is_traceless)


def make_traceless(Q):
    """
    Adjust the diagonal elements of a 3x3 matrix Q to make it traceless.

    Parameters:
    Q : numpy.ndarray
        The 3x3 matrix to be adjusted.

    Returns:
    numpy.ndarray
        The adjusted 3x3 matrix that is traceless.
    """
    trace_Q = np.trace(Q)
    correction = trace_Q / 3
    Q_traceless = Q.copy()
    np.fill_diagonal(Q_traceless, Q_traceless.diagonal() - correction)

    return Q_traceless


# set colormap
plt.set_cmap('bwr')


def plot_3d(grid, esp, atom_info=None):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(*grid.T, c=esp, vmin=-0.015, vmax=0.015)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-10, 10)
    # label axes
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # plot atoms on xy plane
    if atom_info is not None:
        xyz, atom_types = atom_info
        ax1.scatter(*xyz.T, c='black', s=10)
        # ax1.plot(*xyz.T, c='black')

        for i, _ in enumerate(xyz):
            if atom_types[i] != 'H':
                an = atom_types[i] if atom_types[i] != 'C' else '.'
                ax1.text(_[0], _[1], -10, an, zdir='z')
                ax1.text(_[0], 10, _[2], an, zdir='y')
                ax1.text(-10, _[1], _[2], an, zdir='x')

    plt.show()


testdata = Path("/home/boittier/Documents/phd/pythonProject/data/val-ala_0_0")

# read xyz file
xyz = np.genfromtxt(testdata / "val-ala_0_0.xyz",
                    skip_header=2, usecols=(1, 2, 3))
atom_types = np.genfromtxt(testdata / "val-ala_0_0.xyz",
                           skip_header=2, usecols=(0), dtype=str)

grid = np.genfromtxt(testdata / "grid.dat")
esp = np.genfromtxt(testdata / "grid_esp.dat")

# read npz file
npz = np.load(testdata / "multipoles.npz")
print(npz.files)

monopoles = npz['monopoles']
dipoles = npz['dipoles']
quadrupoles = npz['quadrupoles']
print("Monopoles, dipoles, quadrupoles:")
print(monopoles.shape, dipoles.shape, quadrupoles.shape)

calc_esp = np.zeros(esp.shape)
print("Calculated ESP:")
print(calc_esp.shape)

for i in range(monopoles.shape[0]):
    for grid_point in range(grid.shape[0]):
        r = grid[grid_point] - xyz[i]
        r_norm = np.linalg.norm(r) * 1.8897259886
        calc_esp[grid_point] += monopoles[i] / r_norm

plt.scatter(esp, calc_esp)
rmse = np.sqrt(np.mean((esp - calc_esp) ** 2))
plt.xlabel("ESP")
plt.ylabel("Calculated ESP")
plt.title("Monopole\n"
          "ESP vs Calculated ESP (RMSE = {:.4f})".format(rmse * 627.509))
plt.plot([-0.15, 0.15], [-0.15, 0.15], color='red')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plot_3d(grid, esp)
plot_3d(grid, calc_esp)
plot_3d(grid, esp - calc_esp)

#  add on dipole contribution
for i in range(dipoles.shape[0]):
    for grid_point in range(grid.shape[0]):
        r = grid[grid_point] - xyz[i]
        r_norm = np.linalg.norm(r) * 1.8897259886
        calc_esp[grid_point] += np.dot(dipoles[i], r) / r_norm ** 3

plt.scatter(esp, calc_esp)
rmse = np.sqrt(np.mean((esp - calc_esp) ** 2))
plt.xlabel("ESP")
plt.ylabel("Calculated ESP")
plt.title("Dipole\n"
          "ESP vs Calculated ESP (RMSE = {:.4f})".format(rmse * 627.509))
plt.plot([-0.15, 0.15], [-0.15, 0.15], color='red')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plot_3d(grid, esp)
plot_3d(grid, calc_esp)
plot_3d(grid, esp - calc_esp)

# add on quadrupole contribution
for i in range(quadrupoles.shape[0]):
    quadrupoles[i] = make_traceless(quadrupoles[i])
    # print(check_symmetric_and_traceless(quadrupoles[i]))
    for grid_point in range(grid.shape[0]):
        r = grid[grid_point] - xyz[i]
        r_norm = np.linalg.norm(r) * 1.8897259886
        v = np.dot(r, np.dot(quadrupoles[i], r))
        # print(v)
        calc_esp[grid_point] += v / (2 * r_norm ** 5)

plt.scatter(esp, calc_esp)
rmse = np.sqrt(np.mean((esp - calc_esp) ** 2))
plt.xlabel("ESP")
plt.ylabel("Calculated ESP")
plt.title("Quadrupole\n"
          "ESP vs Calculated ESP (RMSE = {:.4f})".format(rmse * 627.509))
plt.plot([-0.15, 0.15], [-0.15, 0.15], color='red')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

plot_3d(grid, esp)
plot_3d(grid, calc_esp)
plot_3d(grid, esp - calc_esp)
