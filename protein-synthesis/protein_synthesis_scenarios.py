import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utility_functions import *



def print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, mRNA, protein):
    print(f'Theoretical results:')
    print(f'Mean mRNA number = {theoretical_mrna_mean}')
    print(f'Mean protein number = {theoretical_protein_mean}')
    print('Covariance matrix:')
    print(S)
    print('-------------------')
    print('Simulation results:')
    print(f'Mean mRNA number = {np.mean(mRNA[-99000:])}')
    print(f'Mean protein number = {np.mean(protein[-99000:])}')
    print('Covariance matrix:')
    print(np.cov(mRNA, protein))

def plot_heatmap(data, title):
    df = pd.DataFrame(data)
    pivoted = df.pivot("K", "h", title)
    sns.heatmap(pivoted)
    plt.title(f'Variation of {title} with K and h')
    plt.show()

def transcription_only():
    lambd_array = np.arange(0.5, 2.5, 0.1)
    beta_array = np.arange(0.01, 0.5, 0.03)
    N = 5000

    T, X, theoretical_mrna_mean, theoretical_mrna_variance = simple_transcription(1, 0.01, 50000)
    plot_species(1, T, X)
    plt.show()
    print(f'When lambda = 1 and beta = 0.01:')
    print(f'Theoretical mean = {theoretical_mrna_mean}')
    print(f'Simulation mean = {np.mean(X[-49000:])}')
    print(f'Theoretical variance = {theoretical_mrna_variance}')
    print(f'Simulation variance = {np.var(X[-49000:])}')

    means_lambd, variances_lambd, means_beta, variances_beta = [], [], [], []
    for lambd in lambd_array:
        for beta in beta_array:
            T, X, theoretical_mrna_mean, theoretical_mrna_variance = simple_transcription(lambd, beta, N)
            if np.round(lambd, 1) == 1.0:
                means_beta.append(np.mean(X[-4500:]))
                variances_beta.append(np.var(X[-4500:]))
            elif beta == 0.01:
                means_lambd.append(np.mean(X[-4500:]))
                variances_lambd.append(np.var(X[-4500:]))

    plt.rcParams["figure.figsize"] = (15, 6)
    plt.subplot(2, 2, 1)
    plt.plot(lambd_array[:-1], means_lambd)
    plt.title('Change of mean with lambda; beta = 0.01')
    plt.subplot(2, 2, 2)
    plt.plot(beta_array, means_beta)
    plt.title('Change of mean with beta; lambda = 1')
    plt.subplot(2, 2, 3)
    plt.plot(lambd_array[:-1], variances_lambd)
    plt.title('Change of variance with lambda; beta = 0.01')
    plt.subplot(2, 2, 4)
    plt.plot(beta_array, variances_beta)
    plt.title('Change of variance with beta; lambda = 1')
    plt.tight_layout()
    plt.show()

def simple_transcription_and_simple_translation():
    lambd = np.linspace(0.4, 3.3, 5)
    beta = np.linspace(0.2, 0.33, 5)
    alpha = np.linspace(0.4, 3.2, 5)
    gamma = np.linspace(0.007, 0.03, 5)
    N = 100000

    for l, b, a, g in zip(lambd, beta, alpha, gamma):
        T, X, theoretical_mrna_mean, theoretical_protein_mean, S = simple_expression(l, b, a, g, N)
        plot_species(2, T, X)
        plt.show()
        print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, X[0], X[1])

def simple_expression_mrna_copy_number_around_2():
    lambd, beta, alpha, gamma, N = 0.8, 0.4, 0.8, 0.08, 100000
    T, X, theoretical_mrna_mean, theoretical_protein_mean, S = simple_expression(lambd, beta, alpha, gamma, N)
    plot_species(2, T, X)
    plt.show()
    print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, X[0], X[1])

def simple_expression_mrna_copy_number_around_2_high_transcription_low_translation_rate():
    lambd, beta, alpha, gamma, N = 2.8, 1.4, 0.7, 0.07, 100000
    T, X, theoretical_mrna_mean, theoretical_protein_mean, S = simple_expression(lambd, beta, alpha, gamma, N)
    plot_species(2, T, X)
    plt.show()
    print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, X[0], X[1])

def self_regulation_transcription_and_simple_translation_h_and_k_trials():
    lambd, beta, alpha, gamma, N = 0.8, 0.4, 0.8, 0.08, 100000
    K_array, h_array = np.arange(10, 100, 20), np.arange(1, 6, 1)

    for K in K_array:
        for h in h_array:
            T, X, theoretical_mrna_mean, theoretical_protein_mean, S = self_regulating_expression(lambd, beta, alpha, gamma, K, h, N)
            plot_species(2, T, X)
            plt.show()
            print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, X[0], X[1])

def keep_h_at_2_and_increasing_k():
    lambd, beta, alpha, gamma, h, N = 0.8, 0.4, 0.8, 0.08, 2, 100000
    K_array = np.arange(60, 160, 5)

    for K in K_array:
        T, X, theoretical_mrna_mean, theoretical_protein_mean, S = self_regulating_expression(lambd, beta, alpha, gamma, K, h, N)
        plot_species(2, T, X)
        plt.show()
        print_simulation_results(theoretical_mrna_mean, theoretical_protein_mean, S, X[0], X[1])

def self_regulating_transcription_and_simple_translation():
    lambd, beta, alpha, gamma, N = 0.8, 0.4, 0.8, 0.08, 100000
    K_array, h_array = np.arange(10, 100, 20), np.arange(1, 6, 1)
    K_vals, h_vals, d_m_mean, d_p_mean, d_m_var, d_covar, d_p_var = [], [], [], [], [], [], []

    for K in K_array:
        for h in h_array:
            T, X, theoretical_mrna_mean, theoretical_protein_mean, S = self_regulating_expression(lambd, beta, alpha, gamma, K, h, N)
            mRNA, protein = X[0], X[1]
            simulated_covariance_matrix = np.cov(mRNA, protein)
            K_vals.append(K)
            h_vals.append(h)
            d_m_mean.append(abs(theoretical_mrna_mean - np.mean(mRNA[-99000:])))
            d_p_mean.append(abs(theoretical_protein_mean - np.mean(protein[-99000:])))
            d_S = abs(S - simulated_covariance_matrix)
            d_m_var.append(d_S[0][0])
            d_covar.append(d_S[0][1])
            d_p_var.append(d_S[1][1])

    plt.rcParams["figure.figsize"] = (5, 4)
    for data, title in zip([d_m_mean, d_p_mean, d_m_var, d_covar, d_p_var],
                           ['mRNA mean difference', 'protein mean difference', 'mRNA variance difference', 'covariance difference', 'protein variance difference']):
        plot_heatmap({'K': K_vals, 'h': h_vals, title: data}, title)

def simple_transcription_and_unmatured_protein_translation():
    lambd, beta, alpha, gamma, tau_p, N = 0.5, 0.2, 1.1, 0.011, 1, 100000
    T, X = unmatured_expression('simple', lambd, beta, alpha, gamma, tau_p, N)
    plot_species(3, T, X)
    plt.show()
    print('Simulation results:')
    print(f'Mean mRNA number = {np.mean(X[0][-99000:])}')
    print(f'Mean unmatured protein number = {np.mean(X[1][-99000:])}')
    print(f'Mean protein number = {np.mean(X[2][-99000:])}')
    print('Covariance matrix:')
    print(np.cov(X[0], X[2]))

def self_regulating_and_unmatured_protein_translation():
    lambd, beta, alpha, gamma, K, h, N = 0.8, 0.4, 0.8, 0.08, 100, 2, 100000
    tau_p_array = np.geomspace(1, 100, num=60)
    protein_variances = []

    for tau_p in tau_p_array:
        T, X = unmatured_expression('regulated', lambd, beta, alpha, gamma, tau_p, N, K, h)
        mRNA, protein = X[0], X[2]
        protein_variances.append(np.cov(mRNA, protein)[1][1])
        if round(tau_p, 1) in [1.0, 100.0]:
            plot_species(3, T, X)
            plt.show()
            print(f'Simulation results, tau = {tau_p}:')
            print(f'Mean mRNA number = {np.mean(mRNA[-99000:])}')
            print(f'Mean unmatured protein number = {np.mean(X[1][-99000:])}')
            print(f'Mean protein number = {np.mean(protein[-99000:])}')
            print('Covariance matrix:')
            print(np.cov(mRNA, protein))

    plt.plot(tau_p_array, protein_variances)
    plt.grid(b=True, which='both', color='0.65', linestyle='-')
    plt.xlabel('tau', size=15)
    plt.ylabel('Protein variance', size=15)
    plt.title('Dependency of protein variance on tau', size=16)
    plt.show()