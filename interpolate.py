import numpy as np
import pyvista as pv
import scipy
from scipy import linalg
from numpy import random
from scipy.spatial.transform import Rotation as R
from stiefel_exp.Stiefel_Exp_Log import Stiefel_Exp, Stiefel_Log


def interpolate(basis_0, basis_1, alpha):
    delta, _ = Stiefel_Log(basis_0, basis_1, 1e-3)
    basis_2 = Stiefel_Exp(basis_0, alpha * delta)
    return basis_2

def is_orthonormal(matrix):
    return np.allclose(matrix.T @ matrix, np.eye(matrix.shape[1]))


def main():
    basis_0 = np.eye(3)
    rotation = R.from_rotvec(np.array([0, 5*np.pi/4, 0])).as_matrix()
    basis_1 = basis_0 @ rotation
    basis_2 = interpolate(basis_0, basis_1, 0.5)

    assert is_orthonormal(basis_0), 'basis_0 is not orthonormal'
    assert is_orthonormal(basis_1), 'basis_1 is not orthonormal'
    assert is_orthonormal(basis_2), 'basis_2 is not orthonormal'

    plot = pv.Plotter()
    plot.add_arrows(np.zeros((3, 3)), basis_0.T, mag=1, color='r')
    plot.add_arrows(np.zeros((3, 3)), basis_2.T, mag=1, color='g')
    plot.add_arrows(np.zeros((3, 3)), basis_1.T, mag=1, color='b')
    plot.show()

if __name__ == "__main__":
    main()


