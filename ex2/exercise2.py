import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
from typing import Callable, Tuple
from sklearn.utils import shuffle
import random

warnings.filterwarnings('ignore')

# def minimize(f: Callable , df: Callable, x0: np.ndarray, lr: float, num_iters: int) -> \
#         Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
#     """
#     :param f: objective function
#     :param df: gradient of objective function
#     :param x0: start point, shape [dimension]
#     :param lr: learning rate
#     :param num_iters: maximum number of iterations
#     :return argmin, min, values of x for all interations, value of f(x) for all iterations
#     """
#     # initialize
#     x = np.zeros([num_iters + 1] + list(x0.shape))
#     f_x = np.zeros((num_iters + 1, 120))
#     x[0] = x0 # weights
#     f_x[0] = f(x0)
#     for i in range(num_iters):
#         # update using gradient descent rule
#         grad = df(x[i])
#         x[i + 1] = x[i] - lr * grad
#         f_x[i + 1] = f(x[i + 1])
#     return x[i+1], f_x[i+1], x[:i+1], f_x[:i+1] # logging info for visualization
# def affine_features(x: np.ndarray) -> np.ndarray:
#     """
#     implements affine feature function
#     :param x: inputs
#     :return inputs with additional bias dimension
#     """
#     return np.concatenate([x, np.ones((x.shape[0], 1))], axis=-1)
#
# def generate_one_hot_encoding(y: np.ndarray, num_classes: int) -> np.ndarray:
#     """
#     :param y: vector containing classes as numbers, shape: [N]
#     :param num_classes: number of classes
#     :return a matrix containing the labels in an one-hot encoding, shape: [N x K]
#     """
#     y_oh = np.zeros([y.shape[0], num_classes])
#
#     # can be done more efficiently using numpy with
#     # y_oh[np.arange(y.size), y] = 1.0
#     # but I decided to used the for loop for clarity
#
#     for i in range(y.shape[0]):
#         y_oh[i, y[i]] = 1.0
#
#     return y_oh
#
# def softmax(x: np.ndarray) -> np.ndarray:
#     """softmax function
#     :param x: inputs, shape: [N x K]
#     :return softmax(x), shape [N x K]
#     """
#     a = np.max(x, axis=-1, keepdims=True)
#     log_normalizer = a + np.log(np.sum(np.exp(x - a), axis=-1, keepdims=True))
#     return np.exp(x - log_normalizer)
#
# def categorical_nll(predictions: np.ndarray, labels: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
#     """
#     cross entropy loss function
#     :param predictions: class labels predicted by the classifier, shape: [N x K]
#     :param labels: true class labels, shape: [N x K]
#     :param epsilon: small offset to avoid numerical instabilities (i.e log(0))
#     :return negative log-likelihood of the labels given the predictions, shape: [N]
#     """
#     return - np.sum(labels * np.log(predictions + epsilon), -1)
# # objective
# def objective_cat(flat_weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
#     """
#     :param flat_weights: weights of the classifier (as flattened vector), shape: [feature_dim * K]
#     :param features: samples to evaluate objective on, shape: [N x feature_dim]
#     :param labels: labels corresponding to samples, shape: [N]
#     :return cross entropy loss of the classifier given the samples
#     """
#     num_features = features.shape[-1]
#     num_classes = labels.shape[-1]
#     weights = np.reshape(flat_weights, [num_features, num_classes])
#     #---------------------------------------------------------------
#     # TODO
#     predicted_labels = softmax(features@weights)
#     return categorical_nll(predicted_labels, labels)
#     # ---------------------------------------------------------------
# # derivative
# def d_objective_cat(flat_weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
#     """
#     :param flat_weights: weights of the classifier (as flattened vector), shape: [feature_dim * K]
#     :param features: samples to evaluate objective on, shape: [N x feature_dim]
#     :param labels: labels corresponding to samples, shape: [N]
#     :return gradient of cross entropy loss of the classifier given the samples, shape: [feature_dim * K]
#     """
#     feature_dim = features.shape[-1]
#     num_classes = labels.shape[-1]
#     weights = np.reshape(flat_weights, [feature_dim, num_classes])
#     #---------------------------------------------------------------
#     # TODO, do not forget to flatten the gradient before returning!
#     grad_w = (features.T@(labels*(softmax(features@weights)-1)))/features.shape[0]
#     #---------------------------------------------------------------
#     return grad_w.flatten()
# # optimization
#
# # %%
# data = np.load("iris_data.npz")
# train_samples = data["train_features"]
# train_labels = data["train_labels"]
# test_samples = data["test_features"]
# test_labels = data["test_labels"]
#
#
# train_features = affine_features(train_samples)
# test_features = affine_features(test_samples)
# oh_train_labels = generate_one_hot_encoding(train_labels, 3)
# oh_test_labels = generate_one_hot_encoding(test_labels, 3)
# w0_flat = np.zeros(5 * 3) # 4 features + bias, 3 classes
# w_opt_flat, loss_opt, x_history, f_x_history = minimize(lambda w: objective_cat(w, train_features, oh_train_labels),
#             lambda w: d_objective_cat(w, train_features, oh_train_labels),
#             w0_flat, 1e-1, 1000)
#
# w_opt = np.reshape(w_opt_flat, [5, 3])

# plotting and evaluation
# print("Final Loss:", loss_opt)
# plt.figure()
# plt.plot(f_x_history)
# plt.xlabel("iteration")
# plt.ylabel("negative categorical log-likelihood")
#
# train_pred = softmax(train_features @ w_opt)
# train_acc = np.count_nonzero(np.argmax(train_pred, axis=-1) == np.argmax(oh_train_labels, axis=-1))
# train_acc /= train_labels.shape[0]
# test_pred = softmax(test_features @ w_opt)
# test_acc = np.count_nonzero(np.argmax(test_pred, axis=-1) == np.argmax(oh_test_labels, axis=-1))
# test_acc /= test_labels.shape[0]
# print("Train Accuracy:", train_acc, "Test Accuracy:", test_acc)

def get_k_nearest(k: int, query_point: np.ndarray, x_data: np.ndarray, y_data: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    :param k: number of nearest neigbours to return
    :param query_point: point to evaluate, shape [dimension]

    :param x_data: x values of the data [N x input_dimension]
    :param y_data: y values of the data [N x target_dimension]
    :return k-nearest x values [k x input_dimension], k-nearest y values [k x target_dimension]
    """
    # ---------------------------------------------------------------
    # TODO
    # dist_list = np.linalg.norm(x_data - query_point, axis=1)
    dist_list = [np.inner(x_data[i], query_point)/(np.linalg.norm(x_data[i])*np.linalg.norm(query_point)) for i in range(x_data.shape[0])]
    sorted_dist = np.argsort(dist_list)[np.arange(-1, -1-k, -1)]
    x_data_k = x_data[sorted_dist]
    y_data_k = y_data[sorted_dist]
    # ---------------------------------------------------------------
    return x_data_k, y_data_k

def majority_vote(y: np.ndarray) -> int:
    """
    :param y: k nearest targets [K]
    :return the number x which occours most often in y.
    """
    # ---------------------------------------------------------------
    # TODO
    return np.unique(y)[0]
    # ---------------------------------------------------------------

# 接下来对KNN算法的思想总结一下：就是在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，
# 找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类，其算法的描述为：
#
# 1）计算测试数据与各个训练数据之间的距离；
#
# 2）按照距离的递增关系进行排序；
#
# 3）选取距离最小的K个点；
#
# 4）确定前K个点所在类别的出现频率；
#
# 5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。

# k = 5
# predictions = np.zeros(test_features.shape[0])
# for i in range(test_features.shape[0]):
#     _, nearest_y = get_k_nearest(k, test_features[i], train_features, train_labels)
#     predictions[i] = majority_vote(nearest_y)
#
# print("Accuracy: ", np.count_nonzero(predictions == test_labels) / test_labels.shape[0])

def plot_error_curves(MSE_val: np.ndarray, MSE_train: np.ndarray, x_axis, m_star_idx: int):
    plt.yscale('log')
    plt.plot(x_axis, np.mean(MSE_val, axis=0), color='blue', alpha=1, label="mean MSE validation")
    plt.plot(x_axis, np.mean(MSE_train, axis=0), color='orange', alpha=1, label="mean MSE train")
    plt.plot(x_axis[m_star_idx], np.min(np.mean(MSE_val, axis=0)), "x", label='best model')
    plt.xticks(x_axis)
    plt.xlabel("model order")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def plot_best_model(x_plt: np.ndarray, y_plt: np.ndarray, x_samples: np.ndarray, y_samples: np.ndarray,
                    model_best, model_predict_func: callable):
    plt.plot(x_plt, y_plt, color='g', label="Ground truth")
    plt.scatter(x_samples, y_samples, label="Noisy data", color="orange")
    f_hat = model_predict_func(model_best, x_plt)
    plt.plot(x_plt, f_hat, label="Best model")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_bars(M, std_mse_val_ho, std_mse_val_cv):
    models = np.arange(1, M + 1)
    fig = plt.figure("Comparison of the Standard Deviations of mse's")
    ax1 = fig.add_subplot(111)
    ax1.bar(models, std_mse_val_ho, yerr=np.zeros(std_mse_val_ho.shape), align='center', alpha=0.5, ecolor='black',
            color='red', capsize=None)
    ax1.bar(models, std_mse_val_cv, yerr=np.zeros(std_mse_val_cv.shape), align='center', alpha=0.5, ecolor='black',
            color='blue', capsize=None)
    ax1.set_xticks(models)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_yscale('log')
    ax1.set_xticklabels(models)
    ax1.set_title('Standard Deviations for HO (red) and CV (blue)')
    ax1.yaxis.grid(True)
    plt.legend(['HO', 'CV'])
    fig.show()

def plot(mse_val: np.ndarray, mse_train:np.ndarray, x_axis, m_star_idx:int, x_plt:np.ndarray, y_plt: np.ndarray,
        x_samples:np.ndarray, y_samples:np.ndarray, model_best, model_predict_func:callable):
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plot_error_curves(mse_val, mse_train, x_axis, m_star_idx)
    plt.subplot(122)
    plot_best_model(x_plt, y_plt, x_samples, y_samples, model_best, model_predict_func)
    plt.show()

def hold_out_method(data_in: np.ndarray, data_out: np.ndarray, split_coeff: float)->Tuple[dict, dict]:
    """
    Splits the data into a training data set and a validation data set.
    :param data_in: the input data which we want to split, shape: [n_data x indim_data]
    :param data_out: the output data which we want to split, shape: [n_data x outdim_data]
    Note: each data point i in data_in and data_out is considered as a training/validation sample -> (x_i, y_i)
    :param split_coeff: a value between [0, 1], which determines the index to split data into test and validation set
                        according to: split_idx = int(n_data*split_coeff)
    :return: Returns a tuple of 2 dictionaries: the first element in the tuple is the training data set dictionary
             containing the input data marked with key 'x' and the output data marked with key 'y'.
             The second element in the tuple is the validation data set dictionary containing the input data
             marked with key 'x' and the output data marked with key 'y'.
    """
    n_data = data_in.shape[0]
    # we use a dictionary to store the training and validation data.
    # Please use 'x' as a key for the input data and 'y' as a key for the output data in the dictionaries
    # for the training data and validation data
    # shuffle_in, shuffle_out = shuffle(data_in.squeeze(), data_out.squeeze(), random_state=0)
    # shuffle_in, shuffle_out = shuffle(data_in, data_out, random_state=0)
    split_idx = int(n_data * split_coeff)
    train_data = {}
    val_data = {}
    train_data["x"], train_data["y"] = data_in[:split_idx, :], data_out[:split_idx, :]
    val_data["x"],  val_data["y"] = data_in[split_idx:, :], data_out[split_idx:, :]
    return train_data, val_data


def eval_ho_method(M: int, split_coeff: float, fit_func: callable, predict_func: callable):
    """
    :param M: Model complexity param: for polynomial regression model order, for kNNR: number of neighbors
    :param split_coeff:a value between [0, 1], which determines the index to split data into test and validation set
                     according to: split_idx = int(n_data*split_coeff)
    :param fit_func: callable function which will fit your model
    :param predict_func: callable function which will make predictions with your model
    """
    n_repetitions = 20  # we have 20 different data sets, we want to perform HO on....
    models = np.arange(1, M + 1)
    mse_train_ho = np.zeros((n_repetitions, M))
    mse_val_ho = np.zeros((n_repetitions, M))

    for rep in range(n_repetitions):
        c_x_samples = x_samples[rep, :, :]  # extract the current data set
        c_y_samples = y_samples[rep, :, :]  # extract the current data set
        train_data, val_data = hold_out_method(c_x_samples, c_y_samples, split_coeff)

        for i, m in enumerate(models):
            # 2: Train on training data to obtain \hat{f}_{D_T}(x)
            p = fit_func(train_data['x'], train_data['y'], m)
            f_hat_D_T = predict_func(p, val_data['x'])

            # 3: Evaluate resulting estimators on validation data
            mse_val_ho[rep, i] = np.mean((f_hat_D_T - val_data['y']) ** 2)

            # MSE on training set for comparison
            y_train = predict_func(p, train_data['x'])
            mse_train_ho[rep, i] = np.mean((y_train - train_data['y']) ** 2)

            # log parameters of best model order
            if i == 0:
                p_best_ho = p
            else:
                if mse_val_ho[rep, i] <= np.min(mse_val_ho[rep, :i].reshape(-1)):
                    p_best_ho = p

    # mean over all repetitions
    mean_mse_train_ho = np.mean(mse_train_ho, axis=0)
    mean_mse_val_ho = np.mean(mse_val_ho, axis=0)

    std_mse_train_ho = np.std(mse_train_ho, axis=0)
    std_mse_val_ho = np.std(mse_val_ho, axis=0)

    # 4: Pick model with best validation loss
    m_star_idx_ho = np.argmin(mean_mse_val_ho)
    m_star_ho = models[m_star_idx_ho]
    print("Best model complexity for Hold out: {}".format(m_star_ho))

    train_data, val_data = hold_out_method(x_samples[0, :, :], y_samples[0, :, :], split_coeff)
    p_best_ho = fit_func(train_data['x'], train_data['y'], m_star_ho)

    # use only the first data set for better readability
    plot(mse_val_ho, mse_train_ho, models, m_star_idx_ho, x_plt, y_plt,
         x_samples[0, :, :], y_samples[0, :, :], p_best_ho, predict_func)
    return std_mse_val_ho


def k_fold_cross_validation(data_in: np.ndarray, data_out: np.ndarray, m: int, k: int, fit: callable,
                            predict_func: callable) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    This function will split the data into a training set and a validation set and will shift the splitted data
    k times and train k different models. It will return the mean squarred error of the training and the validation
    data sets, based on the splits.
    :param data_in: the input data which we want to split, shape: [N indim_data]
    :param data_out: the output data which we want to split, shape: [N x outdim_data]
    :param m: model parameter (e.g. polynomial degree, or number of nearest neighbors, ...). We will use this
              variable to call the 'fit' function of your chosen model. Please see the function in the section
              'Evaluation and Fit Functions'.
              m is e.g. the degree of a polynomial
              m is e.g. the parameter k for kNN Regression
    :param k: number of partitions of the data set (not to be confused with k in kNN)
    :param fit: callable function which will fit your model to the training data you provide -> expects
                train_in (np.ndarray), train_out (np.ndarray), m (model parameter(s))
                (e.g. model order in polynomial regression), returns model params
    :param predict_func: callable function which will use your model to do predictions on the input data
                         you provide -> expects model params (-> m) and data_in (np.ndarray)
    :return mse_train: np.ndarray containg the mean squarred errors for each training data in each split k shape [k]
    :return mse_val: np.ndarray containing the mean squarred errors for the validation data in each split, shape[k]
    """
    n_data = data_in.shape[0]

    if k > n_data:
        k = n_data

    # number of validation data
    n_val_data = n_data // k  # e.g.: 15//2 = 7
    ind = np.arange(0, n_data)
    mse_train = np.zeros(k)
    mse_val = np.zeros(k)


    # shuffle_in, shuffle_out = shuffle(data_in, data_out)
    train_data, val_data = {}, {}
    for i in range(k):
        # 1: Split into 2 data sets
        val_index = np.arange(i * n_val_data, (i + 1) * n_val_data)
        train_index = a = [n for n in ind if not n in val_index]

        # 2: get the training and validation data set
        train_data["x"], train_data["y"] = data_in[train_index], data_out[train_index]
        val_data["x"], val_data["y"] = data_in[val_index], data_out[val_index]

        # 3: fit your model on training data
        # Use here the 'fit' function. Expects (train_in:np.ndarray, train_out: np.ndarray, m)
        p = fit(train_data['x'], train_data['y'], m)

        # 4: evaluate your model on training set and validation set
        # Use here the 'predict_func' function. Expects (model you have fitted, data you want to make predictions)
        f_hat_D_T = predict_func(p, val_data['x'])


        y_train = predict_func(p, train_data['x'])


        # 5: assign performance: Calculate the mean squared error for the training and validation set and
        # write the result into the mse_train and mse_val arrays respectively
        # ---------------------------------------------------------------
        mse_val[i] = np.mean((f_hat_D_T - val_data['y']) ** 2)
        mse_train[i] = np.mean((y_train - train_data['y']) ** 2)
        # ---------------------------------------------------------------
    return mse_train, mse_val




def eval_cv_method(M: int, k: int, fit_func: callable, predict_func: callable):
    """
    :param M: Model complexity param: for polynomial regression model order, for kNNR: k number of neighbors
    :param k: number of partitions
    :param fit_func: callable function which will fit your model
    :param predict_func: callable function which will use your model to perform predictions on data
    """
    n_repetitions = 20  # we have 20 different data sets, we want to perform CV on....
    models = np.arange(1, M + 1)

    mse_train_cv = np.zeros((n_repetitions, M))
    mse_val_cv = np.zeros((n_repetitions, M))

    for rep in range(n_repetitions):
        c_x_samples = x_samples[rep, :, :]  # extract the current data set
        c_y_samples = y_samples[rep, :, :]  # extract the current data set

        for i, m in enumerate(models):
            mse_train, mse_val = k_fold_cross_validation(c_x_samples, c_y_samples, m, k, fit_func, predict_func)
            mse_val_cv[rep, i] = np.mean(mse_val)
            mse_train_cv[rep, i] = np.mean(mse_train)

    mean_mse_val_cv = np.mean(mse_val_cv, axis=0)
    mean_mse_train_cv = np.mean(mse_train_cv, axis=0)
    std_mse_val_cv = np.std(mse_val_cv, axis=0)  # calculates the standard deviation of the mse's over the 20 data sets
    std_mse_train_cv = np.std(mse_train_cv,
                              axis=0)  # calculates the standard deviation of the mse's over the 20 data sets

    m_star_idx_cv = np.argmin(mean_mse_val_cv)
    m_star_cv = models[m_star_idx_cv]
    print("Best model complexity for Cross Validation:", m_star_cv)

    # use only the first data set for better readability
    p_best_cv = fit_func(x_samples[0, :, :], y_samples[0, :, :], m_star_cv)

    plot(mse_val_cv, mse_train_cv, models, m_star_idx_cv, x_plt, y_plt,
         x_samples[0, :, :], y_samples[0, :, :], p_best_cv, predict_func)

    return std_mse_val_cv

def fit_knn_regressor(train_in: np.ndarray, train_out:np.ndarray, k: int)->dict:
    """
    This function will fit a knn model to the data. In fact, it will compactly represent the data provided.
    I.e. it will store the training in- and output data together with the number of k neighbors in a dictionary.
    :param train_in: the training input data, shape [N x input dim]
    :param train_out: the training output data, shape [N x output dim]
    :param k: the parameter how many nearest neighbors to choose.
    :return: returns a dictionary containgin all the information:
             The key 'x' marks the training input data (shape [N x input dim]).
             The key 'y' marks the training output data (shape [N x output dimension]).
             The key 'k' marks the parameter for k-nearest neighbors to be considered.
    """
    model = {'x': train_in, 'y': train_out, 'k': k}
    return model

def predict_knn_regressor(model, data_in: np.ndarray)->np.ndarray:
    """
    This function will perform predictions using a knn regression model given the input data.
    Note that knn is a lazy model and requires to store
    all the training data (see dictionary 'model').
    :param model: dictionary containing the train data and the k parameter for k nearest neighbors.
                  The key 'x' marks the training input data (shape [N x input dim]).
                  The key 'y' marks the training output data (shape [N x output dimension]).
                  The key 'k' marks the parameter for k-nearest neighbors to be considered.
    :param data_in: the data we want to perform predictions (shape [N x input dimension])
    :return prediction based on k nearest neighbors (mean of the k - neares neighbors) (shape[N x output dimension])
    """
    if len(data_in.shape) == 1:
        data_in = np.reshape(data_in, (-1, 1))
    train_data_in = model['x']
    train_data_out = model['y']
    k = model['k']
    if len(train_data_in) == 1:
        train_data_in = np.reshape(train_data_in, (-1, 1))
    predictions = np.zeros((data_in.shape[0], train_data_out.shape[1]))
    for i in range(data_in.shape[0]):
        c_data = data_in[i, :]
        _, nearest_y = get_k_nearest(k, c_data, train_data_in, train_data_out)
        # we take the mean of the nearest samples to perform predictions
        predictions[i, :] = np.mean(nearest_y, axis=0)
    return predictions

def fit_forest_fixed_n_trees(train_in: np.ndarray, train_out:np.ndarray, min_samples_leaf: int):
    """
    This function will fit a forest model based on a fixed number of trees (can not be change when using this
    function, is set globally)
    :param train_in: the training input data, shape [N x input dim]
    :param train_out: the training output data, shape [N x output dim]
    :param min_samples_leaf: the number of samples per leaf to be used
    """
    model = RandomForestRegressor(n_estimators=1, min_samples_leaf=min_samples_leaf)
    model.fit(train_in, train_out)
    return model

def fit_forest_fixed_n_samples_leaf(train_in: np.ndarray, train_out:np.ndarray, n_trees: int):
    """
    This function will fit a forest model based on a fixed number of sample per leaf (can not be change when
    using this function, is set globally)
    :param train_in: the training input data, shape [N x input dim]
    :param train_out: the training output data, shape [N x output dim]
    :param n_trees: the number of trees in the forest
    """
    model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=1)
    model.fit(train_in, train_out)
    return model

def predict_forest(model, data_in: np.ndarray)->np.ndarray:
    """
    This function will perform predictions using a forest regression model on the input data.
    :param model: the forest model from scikit learn (fitted before)
    :param data_in: :param data_in: the data we want to perform predictions (shape [N x input dimension])
    :return prediction based on chosen minimum samples per leaf (shape[N x output dimension]
    """
    y = model.predict(data_in)
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))
    return y



np.random.seed(33)
# We have 20 training sets with 50 samples in each set
x_samples = np.load('x_samples.npy')    # shape: [20, 50, 1]
y_samples = np.load('y_samples.npy')    # shape: [20, 50, 1]


# we load our plot data (shape: [20, 1])
x_plt = np.load('x_plt.npy')
y_plt = np.load('y_plt.npy')

# let's plot our data (for the training data we just use the first training set)
plt.plot(x_plt, y_plt, c="blue", label="Ground Truth Polynomial")
plt.scatter(x_samples[0, :, :], y_samples[0, :, :], c="orange", label="Samples of first trianing set")
plt.legend()
plt.show()
train, vali = hold_out_method(x_samples, y_samples, 0.75)

M_knn = 20     # Number of Neighbors K
split_coeff = 0.8 # between 0,1: how many samples to split in Hold-out
k = 15        # number of splits for Cross Validation


# Hold Out
std_mse_val_ho_knn = eval_ho_method(M=M_knn, split_coeff=split_coeff, fit_func=fit_knn_regressor,
              predict_func=predict_knn_regressor)

# Cross Validation
std_mse_val_cv_knn = eval_cv_method(M=M_knn, k=k, fit_func=fit_knn_regressor,
                                    predict_func=predict_knn_regressor)

# Comparing the standard deviations
plot_bars(M_knn, std_mse_val_ho_knn, std_mse_val_cv_knn)

min_samples_leaf = 10       # used when fixed number of trees and we want to evaluate number of samples per leaf
split_coeff = 0.8           # between 0,1: how many samples to split in Hold-out
k = 15                      # number of splits for Cross Validation

#Hold-Out
std_mse_val_ho_forest_fixed_n_trees = eval_ho_method(M=min_samples_leaf, split_coeff=split_coeff,
                                                     fit_func=fit_forest_fixed_n_trees,
                                                     predict_func=predict_forest)
# Cross Validation
std_mse_val_cv_forest_fixed_n_trees = eval_cv_method(M=min_samples_leaf, k=k, fit_func=fit_forest_fixed_n_trees,
                                    predict_func=predict_forest)

# Comparing the standard deviations
plot_bars(min_samples_leaf, std_mse_val_ho_forest_fixed_n_trees, std_mse_val_cv_forest_fixed_n_trees)

n_trees = 20       # used when fixed number of trees and we want to evaluate number of samples per leaf
split_coeff = 0.8           # between 0,1: how many samples to split in Hold-out
k = 15                      # number of splits for Cross Validation

# Hold out
std_mse_val_ho_forest_fixed_n_samples = eval_ho_method(M=n_trees, split_coeff=split_coeff,
                                                       fit_func=fit_forest_fixed_n_samples_leaf,
                                                       predict_func=predict_forest)
# Cross Validation
std_mse_val_cv_forest_fixed_n_samples = eval_cv_method(M=n_trees, k=k,
                                                       fit_func=fit_forest_fixed_n_samples_leaf,
                                                       predict_func=predict_forest)


# Comparing the standard deviations
plot_bars(n_trees, std_mse_val_ho_forest_fixed_n_samples, std_mse_val_cv_forest_fixed_n_samples)

