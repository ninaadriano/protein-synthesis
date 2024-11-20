import numpy as np
import matplotlib.pyplot as plt
import random
import control
import pandas as pd
import seaborn as sns

def simple_transcription_rates(prod_rate, deg_rate, mrna_count):
    return [prod_rate, deg_rate*mrna_count]

def simple_translation_rates(prod_rate, deg_rate, mrna_count, protein_count):
    return [prod_rate*mrna_count, deg_rate*protein_count]

def self_regulating_transcription(prod_rate, deg_rate, K, h, mrna_count, protein_count):
    return [(2*prod_rate*K)/(K + protein_count**h), deg_rate*mrna_count]

def unmatured_protein_translation(prod_rate, deg_rate, tau_p, mrna_count, unmatured_protein_count, protein_count):
    return [prod_rate*mrna_count, (1/tau_p)*unmatured_protein_count, deg_rate*protein_count]

def gillespie_step(num_species, i, rates, r, T, X, t, x):
    tau = -(1 / np.sum(rates)) * np.log(np.random.rand())
    t = t + tau
    T[i] = t
    reac = np.sum(np.cumsum(rates / np.sum(rates)) < np.random.rand())
    if num_species == 1:
        x = x + r[reac]
        X[i] = x
    else:
        x = x + r[:, reac]
        X[:, i] = x

    return t, x

def plot_species(num_species, T, X):

    plt.rcParams["figure.figsize"] = (8, 6)
    if num_species == 1:
        plt.plot(T, X)
        plt.title('mRNA transcription', size=16)
        plt.xlabel('time (min)', size=14)
        plt.ylabel('mRNA copy number', size=14)
    elif num_species == 2:
        plt.plot(T, X[0], label='mRNA')
        plt.plot(T, X[1], label='protein')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title('Gene expression', size=16)
        plt.xlabel('time (min)', size=14)
        plt.ylabel('Species copy number', size=14)
    elif num_species == 3:
        plt.plot(T, X[0], label='mRNA')
        plt.plot(T, X[1], label='unmatured protein', color='g')
        plt.plot(T, X[2], label='protein', color='orange')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title('Gene expression', size=16)
        plt.xlabel('time (min)', size=14)
        plt.ylabel('Species copy number', size=16)


def simple_transcription(lambd, beta, N):
    r = np.array([1, -1])
    T = np.zeros(N)
    X = np.zeros(N)

    theoretical_mrna_mean = lambd / beta
    theoretical_mrna_variance = theoretical_mrna_mean

    t = 0
    x = [0]

    for i in range(N):
        rates = simple_transcription_rates(lambd, beta, x[0])
        t, x = gillespie_step(1, i, rates, r, T, X, t, x)

    return T, X, theoretical_mrna_mean, theoretical_mrna_variance


def simple_expression(lambd, beta, alpha, gamma, N):
    r = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
    T = np.zeros(N)
    X = np.zeros((2, N))

    t = 0
    x = [0, 0]

    for i in range(N):
        rates = [*simple_transcription_rates(lambd, beta, x[0]), *simple_translation_rates(alpha, gamma, x[0], x[1])]
        t, x = gillespie_step(2, i, rates, r, T, X, t, x)

    theoretical_mrna_mean = lambd / beta
    theoretical_protein_mean = theoretical_mrna_mean * alpha / gamma

    A = np.array([[-beta, 0], [alpha, -gamma]])
    D = np.array([[lambd + beta * theoretical_mrna_mean, 0],
                  [0, alpha * theoretical_mrna_mean + gamma * theoretical_protein_mean]])
    S = control.lyap(A, D)

    return T, X, theoretical_mrna_mean, theoretical_protein_mean, S


def self_regulating_expression(lambd, beta, alpha, gamma, K, h, N):
    r = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
    T = np.zeros(N)
    X = np.zeros((2, N))

    t = 0
    x = [0, 0]

    for i in range(N):
        rates = [*self_regulating_transcription(lambd, beta, K, h, x[0], x[1]),
                 *simple_translation_rates(alpha, gamma, x[0], x[1])]
        t, x = gillespie_step(2, i, rates, r, T, X, t, x)

    theoretical_protein_mean = ((2 * lambd * K * alpha) / (beta * gamma)) ** (1 / (h + 1))
    theoretical_mrna_mean = (gamma / alpha) * theoretical_protein_mean

    A = np.array([[-beta, -(2 * lambd * K * h * (theoretical_protein_mean ** (h - 1))) / (
                (K + theoretical_protein_mean ** h) ** 2)], [alpha, -gamma]])
    D = np.array([[(2 * lambd * K) / (K + theoretical_protein_mean ** h) + beta * theoretical_mrna_mean, 0],
                  [0, alpha * theoretical_mrna_mean + gamma * theoretical_protein_mean]])
    S = control.lyap(A, D)

    return T, X, theoretical_mrna_mean, theoretical_protein_mean, S


def unmatured_expression(tx_type, lambd, beta, alpha, gamma, tau_p, N, K=0, h=0):
    r = np.array([[1, -1, 0, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]])
    T = np.zeros(N)
    X = np.zeros((3, N))

    t = 0
    x = [0, 0, 0]

    for i in range(N):
        tx_rates = lambda tx_type: {'simple': simple_transcription_rates(lambd, beta, x[0]),
                                    'regulated': self_regulating_transcription(lambd, beta, K, h, x[0], x[2])}.get(
            tx_type, )
        rates = [*tx_rates(tx_type), *unmatured_protein_translation(alpha, gamma, tau_p, x[0], x[1], x[2])]
        t, x = gillespie_step(3, i, rates, r, T, X, t, x)

    return T, X