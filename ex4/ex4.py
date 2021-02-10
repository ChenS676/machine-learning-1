import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Union, Optional

def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1], color='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 7)
    plt.ylim(0, 7)

def draw_2d_gaussian(mu: np.ndarray, sigma: np.ndarray, plt_std: float = 2, *args, **kwargs) -> None:
    (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(sigma)
    phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

    plt.scatter(mu[0:1], mu[1:2], marker="x", *args, **kwargs)

    a = plt_std * np.sqrt(largest_eigval)
    b = plt_std * np.sqrt(smallest_eigval)

    ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi, num=200))
    ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi, num=200))

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
    plt.plot(mu[0] + r_ellipse[:, 0], mu[1] + r_ellipse[:, 1], *args, **kwargs)


def plot_ev(mu, eig_vec_1, eig_vec_2):
    arrow_1_end = mu + eig_vec_1
    arrow_1_x = [mu[0], arrow_1_end[0]]
    arrow_1_y = [mu[1], arrow_1_end[1]]

    arrow_2_end = mu + eig_vec_2
    arrow_2_x = [mu[0], arrow_2_end[0]]
    arrow_2_y = [mu[1], arrow_2_end[1]]

    plt.plot(mu[0], mu[1], 'xr')
    plt.plot((mu + eig_vec_1)[0], (mu + eig_vec_1)[1], 'xr')
    plt.plot(arrow_1_x, arrow_1_y, 'red')
    plt.plot(arrow_2_x, arrow_2_y, 'red')


########################### Probabilistic PCA with Expectation Maximization (7 Points) ##########################
# def e_step(W, mu, X, sigma_quad):
#     """
#     Computes/Samples the Latent vectors in matrix Z given transformation matrix W and data X.
#     :param W: Transformation matrix W (shape: [DxM], where D is data dimension, M is latent Dimension)
#     :param X: Data matrix containing the data (shape: [NxD])
#     :param sigma_quad: sigma^2, the variance of the likelihood (already in quadratic form) (shape: float)
#     :return: returns mu_z, the mean of the posterior for each sample x (shape: [NxM])
#              returns z_samples, the latent variables (shape: [MxN])
#              returns var_z, the covariance of the posterior (shape: [MxM])
#     """
#     # TODO: Implement the e-step for PPCA
#     # Hint: np.linalg.solve is useful. You could also use np.linalg.inv. But np.linalg.solve is in general prefered
#
#     N, D = X.shape
#     D, M = W.shape
#     # # compute mean of z -> NxM
#     mu_z = np.array([np.linalg.solve(W.T @ W + sigma_quad * np.eye(M, M), W.T @ i) for i in X-mu])
#     # # compute covariance of z -> MxM
#     var_z = np.linalg.solve(sigma_quad* (W.T @ W + sigma_quad*np.eye(M, M)), np.ones((M, M)))
#     # # sample z for each mean (mu_z is a Matrix (NxM), containg a mean for each data x_i)
#     try:
#         z_samples = np.array(
#             [np.random.multivariate_normal(mu_z[i, :], var_z) for i in range(N)]).T
#     except:
#         z_samples = np.array([np.random.multivariate_normal(mu_z[i, :], var_z+1e-4*np.random.rand(M, M)) for i in range(N)]).T
#     return mu_z, z_samples, var_z
#
# ############################################# M-Step in Probabilistic PCA (4Points) ##############################
# def m_step(z_samples, X):
#     """
#     Computes the variance and the transformation matrix W given the latent vectors in z_samples and the data
#     in matrix X.
#     :param Z: The latent variable vectors stored in z_samples (shape: [NxM])
#     :param X: Data matrix containg the data (shape: [NxD])
#     :return: returns the variance sigma_quad and the transformation matrix W (shape: [DxM])
#     """
#     #############################################################################################################
#     # TODO: Implement the m-step for PPCA
#     # Hint: np.linalg.solve is useful. You could also use np.linalg.inv. But np.linalg.solve is in general prefered
#
#     # # create feature matrix Z
#     M, N = z_samples.shape
#     N, D = X.shape
#     Z = np.concatenate((np.ones((N, 1)), z_samples.T), axis=1)
#     # # Calculate W_tilde (Dx(M+1)) containg the mean of the likelihood and the projection matrix W
#     R, R = Z.shape
#     try:
#         W_tilde = np.linalg.solve(Z.T@Z, Z.T @ X)
#     except:
#         print("W_tilde is a singulalr matrix, a flexible solution will be performed.")
#         W_tilde = np.linalg.solve(Z.T @ Z + 10e-4 * np.random.rand(R, R), Z.T @ X)
#     mu = W_tilde[0]
#     W = W_tilde[1:].T
#     # # Perform the predictions y in matrix Y
#     Y = np.array([W@z_samples[:, i]+mu for i in range(N)])
#     # # calculate variance sigma_quad scalar
#     sigma_quad = np.linalg.norm(Y - X, 2) / (Y-X).size
#     return sigma_quad, mu, W
#
#
#
# def do_ppca(X: np.ndarray, n_principle_comps: int, num_iters: int = 50):
#     np.random.seed(0)
#     W = np.random.normal(size=(X.shape[1], n_principle_comps))
#     mu_X = np.mean(X, axis=0)
#     mu = mu_X.copy()
#     sigma_quad = 1
#     for i in range(num_iters):
#         mu_z, z_samples, var_z = e_step(W, mu, X, sigma_quad)
#         sigma_quad, mu, W = m_step(z_samples, X)
#     return W, z_samples, var_z, sigma_quad, mu


# np.random.seed(0)
#
# x = np.random.uniform(1, 5, size=(120, 1))
# y = x + 1 + np.random.normal(0, 0.7, size=x.shape)
#
# X = np.concatenate((x, y), axis = 1)
# plot_data(X)
#
#
# plt.figure(figsize=(6, 6))
# plot_data(X)
#
# W, z_samples, var_z, sigma_quad, mu = do_ppca(X, n_principle_comps=1)
#
# x_tilde = np.dot(W, z_samples).T + mu                         # reproject to high-dim space
# # x_tilde = z_samples.T @ W.T + mu
# C = np.dot(W, W.T) + sigma_quad*np.eye(W.shape[0])      # covariance of p(x) (reconstructed)
#
# v, U = np.linalg.eig(np.cov(X.T))
# mu_X = np.mean(X, axis=0)
# plot_ev(mu_X, 2*np.sqrt(v[0])*U[:, 0], 2*np.sqrt(v[1])*U[:, 1])
#
# draw_2d_gaussian(mu_X, C)
#
# plt.plot(x_tilde[:, 0], x_tilde[:, 1], 'o', color='orange', alpha=0.2)   # reprojected data points
# plt.show()

#
# ###################################################### auf 2 ##################################################
# from sklearn.datasets import load_digits
#
# digits = load_digits()
# targets = digits.target
#
# # get the images for digit 3 only
# digits_3_indx = np.where(targets == 3)[0]
# digit_3_data = digits.data[digits_3_indx]       # shape: (183, 64)  -> (8 x 8)
# digit_3_targets = digits.target[digits_3_indx]       # only needed to verify that we load digit 3
#
# mu_X_im = np.mean(digit_3_data, axis=0)
#
# #Plot the original digit 3 images
# # plt.figure()
# # fig, axes = plt.subplots(10, 10, figsize=(20, 10))
# # for i, ax in enumerate(axes.flat):
# #      ax.imshow(digit_3_data[i].reshape(8, 8))
#
# # let's perform ppca on the data
# n_principle_comps = 10
# W_im, z_samples_im, var_z_im, sigma_quad_im, mu_im = do_ppca(digit_3_data, n_principle_comps=n_principle_comps)
# x_tilde_im = np.dot(W_im, z_samples_im).T + mu_im
#
# considered_im = digit_3_data[15]
# considered_im_x_tilde = x_tilde_im[15, :]
#
# plt.figure()
# plt.subplot(121)
# plt.title('Original')
# plt.imshow(considered_im.reshape(8, 8))
#
# plt.subplot(122)
# plt.title('Reprojection')
# plt.imshow(considered_im_x_tilde.reshape(8,8))
# plt.show()
#
# # Sample some vectors z
# z = np.random.normal(size=(5, n_principle_comps))
#
# # Project back to D-dim space
# y = np.dot(W_im, z.T).T + mu_im
#
# # Sample noise
# eps = np.random.normal(scale=sigma_quad_im, size=y.shape)
# # Get image
# x = y + eps
#
# plt.figure('Sampled Image')
# fig, axes = plt.subplots(1, 5, figsize=(20, 10))
# for i, ax in enumerate(axes.flat):
#      ax.imshow(x[i].reshape(8, 8))
#


############################## Feature-Based Support Vector Machine (Hinge Loss) (5 Points) ###########
import scipy.optimize as opt

train_data = dict(np.load("two_moons.npz", allow_pickle=True))
train_samples = train_data["samples"]
train_labels = train_data["labels"]
# we need to change the labels for class 0 to -1 to account for the different labels used by an SVM
train_labels[train_labels == 0] = -1

test_data = dict(np.load("two_moons_test.npz", allow_pickle=True))
test_samples = test_data["samples"]
test_labels = test_data["labels"]
# we need to change the labels for class 0 to -1 to account for the different labels used by an SVM
test_labels[test_labels == 0] = -1

plt.figure()
plt.title("Train Data")
plt.scatter(x=train_samples[train_labels == -1, 0], y=train_samples[train_labels == -1, 1], label="c=-1", c="blue")
plt.scatter(x=train_samples[train_labels == 1, 0], y=train_samples[train_labels == 1, 1], label="c=1", c="orange")
plt.legend()

plt.figure()
plt.title("Test Data")
plt.scatter(x=test_samples[test_labels == -1, 0], y=test_samples[test_labels == -1, 1], label="c=-1", c="blue")
plt.scatter(x=test_samples[test_labels == 1, 0], y=test_samples[test_labels == 1, 1], label="c=1", c="orange")
plt.legend()

def cubic_feature_fn(samples: np.ndarray) -> np.ndarray:
    """
    :param x: Batch of 2D data vectors [x, y] [N x dim]
    :return cubic features: [x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x*y, x, y, 1]
    """
    x = samples[..., 0]
    y = samples[..., 1]
    return np.stack([x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x*y, x, y, np.ones(x.shape[0])], axis=-1)

# ######################################### Exercise 2.1) Hinge Loss Objective (2 Points) ###########################
def objective_svm(weights: np.ndarray, features: np.ndarray, labels: np.ndarray, slack_regularizer: float) -> float:
    """
    objective for svm training with hinge loss
    :param weights: current weights to evaluate (shape: [feature_dim])
    :param features: features of training samples (shape:[N x feature_dim])
    :param labels: class labels corresponding to train samples (shape: [N])
    :param slack_regularizer: Factor to weight the violation of margin with (C in slides)
    :returns svm (hinge) objective (scalar)
    """
    ### TODO ###############################
    target = 1- labels * (features @ weights)
    target[target<0] = 0
    loss = np.linalg.norm(weights, 2)**2 + slack_regularizer*np.sum(target)
    return loss

def d_objective_svm(weights: np.ndarray, features: np.ndarray, labels: np.ndarray,
                    slack_regularizer: float) -> np.ndarray:
    """
    gradient of objective for svm training with hinge loss
    :param weights: current weights to evaluate (shape: [feature_dim])
    :param features: features of training samples (shape: [N x feature_dim])
    :param labels: class labels corresponding to train samples (shape: [N])
    :param slack_regularizer: Factor to weight the violation of margin with (C in slides)
    :returns gradient of svm objective (shape: [feature_dim])
    """
    ### TODO ###############################
    N, feature_dim = features.shape
    index = 1 - labels * (features @ weights)
    target = np.array([-1 *features[i,:]*l for i, l in enumerate(labels)])
    grad = 2 *weights + slack_regularizer*np.sum(target[index > 0, :], axis=0)
    return grad

feature_fn = cubic_feature_fn
C = 1000.0

# optimization

train_features = feature_fn(train_samples)
w = np.ones(train_features.shape[-1])
objective_svm(w, train_features, train_labels, C)
d_objective_svm(w, train_features, train_labels, C)
# For detail see: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
res = opt.minimize(
    # pass objective
    fun=lambda w: objective_svm(w, train_features, train_labels, C),
    # pass initial point
    x0=np.ones(train_features.shape[-1]),
    # pass function to evaluate gradient (in scipy.opt "jac" for jacobian)
    jac=lambda w: d_objective_svm(w, train_features, train_labels, C),
    # specify method to use
    method="l-bfgs-b")

print(res)
w_svm = res.x

# evaluation
test_predictions = feature_fn(test_samples) @ w_svm
train_predictions = feature_fn(train_samples) @ w_svm

predicted_train_labels = np.ones(train_predictions.shape)
predicted_train_labels[train_predictions < 0] = -1
print("Train Accuracy: ", np.count_nonzero(predicted_train_labels == train_labels) / len(train_labels))

predicted_test_labels = np.ones(test_predictions.shape)
predicted_test_labels[test_predictions < 0] = -1
print("Test Accuracy: ", np.count_nonzero(predicted_test_labels == test_labels) / len(test_labels))

# plot train, contour, decision boundary and margins
plt.figure()
plt.title("Max Margin Solution")
plt_range = np.arange(-1.5, 1.5, 0.01)
plt_grid = np.stack(np.meshgrid(plt_range, plt_range), axis=-1)
flat_plt_grid = np.reshape(plt_grid, [-1, 2])
plt_grid_shape = plt_grid.shape[:2]

pred_grid = np.reshape(feature_fn(flat_plt_grid) @ w_svm, plt_grid_shape)

plt.contour(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=[-1.0, 0.0, 1.0], colors=["blue", "black", "orange"])
plt.contour(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=[-1, 0, 1], colors=('blue', 'black', 'orange'),
             linestyles=('-',), linewidths=(2,))
plt.contourf(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=10)

plt.colorbar()

s0 =plt.scatter(x=train_samples[train_labels == -1, 0], y=train_samples[train_labels == -1, 1], label="c=-1", c="blue")
s1 =plt.scatter(x=train_samples[train_labels == 1, 0], y=train_samples[train_labels == 1, 1], label="c=1", c="orange")
plt.legend()

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()


#%%
###################################### 3.) Kernelized Support Vector Machine (8 Points) ##############################

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Union, Optional


import scipy.optimize as opt

def solve_qp(Q: np.ndarray, q: np.ndarray,
             G:np.ndarray, h: Union[np.ndarray, float],
             A:np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
    """
    solves quadratic problem: min_x  0.5x^T Q x + q.^T x s.t. Gx <= h and Ax = b
      in the following 'dim' refers to the dimensionality of the optimization variable x
    :param Q: matrix of the quadratic term of the objective, (shape [dim, dim])
    :param q: vector for the linear term of the objective, (shape [dim])
    :param G: factor for lhs of the inequality constraint (shape [dim], or [dim, dim])
    :param h: rhs of the inequality constraint (shape [dim], or scalar)
    :param A: factor for lhs of the equality constraint (shape [dim], or [dim, dim])
    :param b: rhs of the equality constraint (shape [dim], or scalar)
    :return: optimal x (shape [dim])
    """
    x = cp.Variable(q.shape[0])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), constraints=[G @ x <= h, A @ x == b])
    prob.solve()
    return x.value


def get_gaussian_kernel_matrix(x: np.ndarray, sigma: float, y: Optional[np.ndarray] = None) -> np.ndarray:
    """ Computes Kernel matrix K(x,y) between two sets of data points x, y  for a Gaussian Kernel with bandwidth sigma.
    If y is not given it is assumed to be equal to x, i.e. K(x,x) is computed
    :param x: matrix containing first set of points (shape: [N, data_dim])
    :param sigma: bandwidth of gaussian kernel
    :param y: matrix containing second set of points (shape: [M, data_dim])
    :return: kernel matrix K(x,y) (shape [M, N])
    """
    map=lambda err: np.exp(-1 * err.T  @ err/ (2 * sigma))
    N, data_dim = x.shape
    gauss_kernal_mat = np.zeros((N, N))
    if y is None:
        y = x
    M, data_dim = y.shape
    for i in range(N):
        for j in range(M):
            gauss_kernal_mat[i, j] = map(x[i]-y[j])
    return gauss_kernal_mat

def fit_svm(samples: np.ndarray, labels: np.ndarray, sigma: float) -> np.ndarray:
    """
    fits an svm (with Gaussian Kernel)
    :param samples: samples to fit the SVM to (shape: [N, data_dim])
    :param labels: class labels corresponding to samples (shape: [N])
    :param sigma: bandwidth of gaussian kernel
    :return: "alpha" values, weight for each datapoint in the dual formulation of SVM (shape [N])
    """
    ### TODO ######################
    N, data_dim = samples.shape
    Q = np.zeros((N, N))
    gauss_kernal_mat = get_gaussian_kernel_matrix(samples, sigma)
    for i in range(N):
        for j in range(N):
            Q[i, j] = labels[i]*labels[j]*gauss_kernal_mat[i, j]

    q = -1 * np.ones(N)
    A = labels
    h = np.zeros(N)
    G = -1 * np.eye(100)
    b = np.array([0])

    s = solve_qp(Q, q, G, h, A, b)
    epsilon = 1e-7
    s[np.abs(s)<epsilon] = 0
    return s


def predict_svm(samples_query: np.ndarray, samples_train: np.ndarray, labels_train: np.ndarray,
                alphas: np.ndarray, sigma: float) -> np.ndarray:
    """
    predict labels for query samples given training data and weights
    :param samples_query: samples to query (i.e., predict labels) (shape: [N_query, data_dim])
    :param samples_train: samples that where used to train svm (shape: [N_train, data_dim])
    :param labels_train: labels corresponding to samples that where used to train svm (shape: [N_train])
    :param alphas: alphas computed by training procedure (shape: [N_train])
    :param sigma: bandwidth of gaussian kernel
    :return: predicted labels for query points (shape: [N_query])
    """
    N, data_dim = train_samples.shape
    k = get_gaussian_kernel_matrix(samples_query, sigma, samples_train)
    alphas[alphas < 0] = 0
    b = train_labels - np.sum(np.array([labels_train * alphas * k[:, i] for i in range(N)]), axis=1)
    f_x = np.sum([labels_train * alphas * k[i, :] for i in range(N)], axis=1) + b
    ###############################
    return f_x

sigma = 0.3

# train
alphas = fit_svm(train_samples, train_labels, sigma)

# evaluate
train_predictions = predict_svm(train_samples, train_samples, train_labels, alphas, sigma)
test_predictions = predict_svm(test_samples, train_samples, train_labels, alphas, sigma)

predicted_train_labels = np.ones(train_predictions.shape)
predicted_train_labels[train_predictions < 0] = -1
print("Train Accuracy: ", np.count_nonzero(predicted_train_labels == train_labels) / len(train_labels))

predicted_test_labels = np.ones(test_predictions.shape)
predicted_test_labels[test_predictions < 0] = -1
print("Test Accuracy: ", np.count_nonzero(predicted_test_labels == test_labels) / len(test_labels))

# plot train, contour, decision boundary and margins
plt.figure()
plt_range = np.arange(-1.5, 2.5, 0.01)
plt_grid = np.stack(np.meshgrid(plt_range, plt_range), axis=-1)
flat_plt_grid = np.reshape(plt_grid, [-1, 2])
plt_grid_shape = plt_grid.shape[:2]

pred_grid = np.reshape(predict_svm(flat_plt_grid, train_samples, train_labels, alphas, sigma), plt_grid_shape)
plt.contour(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=[-1, 0, 1], colors=('blue', 'black', 'orange'),
             linestyles=('-',), linewidths=(2,))
plt.contourf(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=10)

plt.colorbar()

plt.scatter(x=train_samples[train_labels == -1, 0], y=train_samples[train_labels == -1, 1], label="c=-1", c="blue")
plt.scatter(x=train_samples[train_labels == 1, 0], y=train_samples[train_labels == 1, 1], label="c=1", c="orange")
plt.legend()

# plot margin, decision boundary and support vectors
plt.figure()
plt.contour(plt_grid[..., 0], plt_grid[..., 1], pred_grid, levels=[-1, 0, 1], colors=('blue', 'black', 'orange'),
             linestyles=('-',), linewidths=(2,))

# squeeze alpha values into interval [0, 1] for plotting
alphas_plt = np.clip(alphas / np.max(alphas), a_min=0.0, a_max=1.0)
for label, color in zip([-1, 1], ["blue", "orange"]):
    color_rgb = colors.to_rgb(color)
    samples = train_samples[train_labels == label]
    color_rgba = np.zeros((len(samples), 4))
    color_rgba[:, :3] = color_rgb
    color_rgba[:, 3] = alphas_plt[train_labels == label]
    plt.scatter(x=samples[:, 0], y=samples[:, 1], c=color_rgba)


plt.xlim(-1.5, 2.5)
plt.show()