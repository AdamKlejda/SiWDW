#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    values, eigvec = np.linalg.eig(A)
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec.T)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    
    values, eigvec = np.linalg.eig(A)
    K = eigvec
    K_inv = np.linalg.inv(K)
    L = np.diag(values)

    print("Macierz L:")
    print(L)
    print("Macier K:")
    print(K)
    print("Macierz K_inv:")
    print(K_inv)

    A_new = K.dot(L).dot(K_inv)
    print("Macierz A:")
    print(A)
    print("Macierz A_new:")
    print(A_new)
    # W 2 macierzy występuje drobna rozbieżnośc spowodowana wartościami zmiennoprzecinkowymi
    if np.array_equal(A,A_new):
        print("Macierze są równe")
    else:
        print("Macierze sa różne")
    pass


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    values, eigvec = np.linalg.eig(A)
    eigvec = eigvec.T
    attractors = {}
    
    if np.allclose(eigvec[0],eigvec[1]):
        attractors = {
            'red': eigvec[0],
            'orange': eigvec[0]*-1,
        }
    else:
        attractors = {
            'red': eigvec[0],
            'orange': eigvec[0]*-1,
            'green': eigvec[1],
            'blue': eigvec[1] * -1,
        }

    for c, a in attractors.items():
        plt.quiver(0,0,a[0],a[1],color=c, width=0.006, scale_units='xy', angles='xy', scale=1,
            zorder=4,headwidth=5, headlength=5)
        i = 0 if c in ['red','orange'] else 1
        plt.text(a[0]*1, a[1]*1,  "{0}".format(np.around(values[i],2)) if c in ['red','green'] else "",color=c, zorder=10)
        

    for vec in vectors:
        temp_vec = vec.copy()
        vec = vec/np.linalg.norm(vec)

        for _ in range(500):
            temp_vec = A.dot(temp_vec)
            temp_vec = temp_vec/np.linalg.norm(temp_vec)
        
        color = "black"

        for c,a in attractors.items():
            distance = np.mean(np.abs(temp_vec - a))
            if 0.1 > distance:
                color = c
                break
        
        plt.quiver(0.0, 0.0, vec[0], vec[1], width=0.005, color=color, scale_units='xy', angles='xy',
                   scale=1, zorder=6)

    plt.grid()
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.margins(0.05)
    plt.show()

    pass


def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    print("-------------------------------")
    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)
    print("-------------------------------")

    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)
    print("-------------------------------")

    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)
    print("-------------------------------")

    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
