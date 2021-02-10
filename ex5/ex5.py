import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_plot = np.load('x_plot.npy')
y_plot = np.load('y_plot.npy')

# the data noise is 1
sigma_y = 1

# hyperparameters
n_features = 5    # number of radial basis functions we want to use
lamb = 1e-3       # lambda regularization parameter
# the means of the Radial basis functions
features_means = np.linspace(np.min(x_plot), np.max(x_plot), n_features)

def rbf_features(x: np.ndarray, means: np.ndarray, sigma:float) -> np.ndarray:
    """
    :param x: input parameter (shape: [N, d])
    :param means: means of each rbf function (shape: [k, d] (k=num features))
    :param sigma: bandwidth parameter. We use the same for all rbfs here
    :return : returns the radial basis features including the bias value 1 (shape: [N, k+1])
    """
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if len(means.shape) == 1:
        means = means.reshape((-1, 1))

    (N, D), (K, _) = x.shape, means.shape
    # TODO Implement the normalized rbf features
    features = np.zeros((N, K+1))
    for i in range(K):
        features[:, i] = np.exp(-(np.linalg.norm(x - means[i], 2, axis=1)**2) / 2 *sigma) /x[:, 0]
    # for j in range(K):
    #     for i in range(N):
    #         features[i, j] = np.exp(-np.linalg.norm(x[i] - means[j], 2)/2*sigma)/x[i]
    features[:, -1] = np.ones(N)
    return features

feat_plot = plt.figure("Features")
feat_sigma = 0.6
y_featuers = rbf_features(x_plot, features_means, sigma=feat_sigma)
plt.plot(x_plot, y_featuers[:, :-1])
plt.show()


def posterior_distr(X: np.ndarray, y: np.ndarray, lamb:float, means: np.ndarray, sigma_feat:float):
    """
    :param x: input training data (shape: [N, d])
    :param y: output training data (shape: [N, 1])
    :param lamb: regularization factor (scalar)
    :param means: means of each rbf feature (shape: [k, d])
    :param sigma_feat: bandwidth of the features (scalar)
    :return : returns the posterior mean (shape: [k+1, 1])
                      the posterior covariance (shape: [k+1, k+1])
    """
    N, D = X.shape
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))
    ############################################################
    # TODO Implement the posterior distribution
    sigma_y = np.linalg.norm(y - y.mean()) ** 2 / N
    Phi = rbf_features(X, means, sigma_feat)
    post_mean = np.linalg.solve(Phi.T @ Phi + lamb * sigma_y * np.ones((Phi.T @ Phi).shape), Phi.T @ y)
    post_cov = sigma_y * (Phi.T @ Phi + sigma_y*lamb*np.ones((Phi.T @ Phi).shape))
    ############################################################
    return post_mean, post_cov

def predictive_distr(x: np.ndarray, y: np.ndarray, X: np.ndarray, lamb:float, means: np.ndarray, sigma_feat:float):
    """"
    :param x: input data (shape: [N, d])
    :param y: output training data (shape: [N, 1])
    :param X: input training data (shape: [N, d])
    :param means: means of each rbf feature (shape: [k, d])
    :param sigma_feat: bandwidth of the features (scalar)
    :return : returns the mean (shape: [N, d])
                      the variance (shape: [N])
                      of the predictive distribution
    """
    ############################################################
    # TODO Implement the predictive distribution
    N, D = x.shape
    sigma_y = np.linalg.norm(y-y.mean())**2/N
    Phi_X = rbf_features(X, means, sigma_feat)
    mean_x = np.zeros((N, D))
    var_x = np.zeros(N)
    for i in range(N):
        phi_x = rbf_features(x[i], means, sigma_feat)
        var_x[i] = sigma_y * (phi_x @ (np.linalg.solve((Phi_X.T @ Phi_X + lamb * sigma_y * np.ones((Phi_X.T@Phi_X).shape)), phi_x.T) + 1))
        mean_x[i] = phi_x @ np.linalg.solve((Phi_X.T@Phi_X + lamb * sigma_y * np.ones((Phi_X.T@ Phi_X).shape)), Phi_X.T@y)
    ############################################################
    return mean_x, var_x

def pred_lin_regr( weights: np.ndarray, input_features: np.ndarray):
    """
    :param x: input data (shape: [N, d])
    :param weights: weights for linear regression (shape: [k+1, 1])
    :param input_features: applied features on data to predict on (shape: [N, k+1])
    :return : returns the predictions to the inputs
    """
    return input_features @ weights


# first get the predictive distribution
pred_mean, pred_var = predictive_distr(x_plot, y_train, x_train, lamb=lamb,
                                       means=features_means, sigma_feat=feat_sigma)

# plot the predictive distribution together with the 95%intervall
plt.figure('Predictve Distr')
plt.plot(x_plot, pred_mean, 'b')
plt.fill_between(np.squeeze(x_plot), np.squeeze(pred_mean)-2*np.sqrt(pred_var),
                 np.squeeze(pred_mean)+2*np.sqrt(pred_var), alpha=0.2, color='blue')
plt.plot(x_train, y_train, 'or')
plt.plot(x_plot, y_plot, 'black')
plt.show()
# Calculate the posterior distribution for the weights now
post_mean, post_cov = posterior_distr(x_train, y_train, lamb=lamb, means=features_means,
                                      sigma_feat=feat_sigma)
# sample 10 different models and plot them:
weights = np.random.multivariate_normal(mean=np.squeeze(post_mean), cov=post_cov, size=(10))
example_funcs = np.zeros((weights.shape[0], y_plot.shape[0]))
for i in range(weights.shape[0]):
    example_funcs[i] = pred_lin_regr(weights[i, :], rbf_features(x_plot, features_means, sigma=feat_sigma))
    plt.plot(x_plot, example_funcs[i], 'red', alpha=0.4)
    plt.show()
