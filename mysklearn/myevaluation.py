import mysklearn.myutils as myutils
import random
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       random.seed(random_state)
    
    if shuffle: 
        myutils.randomize_in_place(X, parallel_list=y)

    num_instances = len(X)
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size
    
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []

    x_len = len(X)
    
    fold_modulus = x_len % n_splits
    
    start_idx = 0
    for fold in range(n_splits):    

        if fold < fold_modulus:
            fold_size = x_len // n_splits + 1
        else:
            fold_size = x_len // n_splits

        fold_end = (start_idx + fold_size) - 1

        tmp = []
        for i in range(start_idx, fold_end + 1):
            tmp.append(i)
        X_test_folds.append(tmp)

        tmp = []
        for i in range(0, x_len):
            if i not in X_test_folds[fold]:
                tmp.append(i)
        X_train_folds.append(tmp)

        start_idx = fold_size + start_idx   

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    indices = [x for x in range(0, len(X))]
    labels = []
    uniq_feat = []

    for idx,clss in enumerate(y):

        if clss in uniq_feat:
            labels[uniq_feat.index(clss)].append(indices[idx])
        else:
            labels.append([indices[idx]])
            uniq_feat.append(clss)
    
    index = 0
    X_test_folds = [[] for _ in range(0, n_splits)]

    for label in labels:
        for val in label:
            fold_idx = index%n_splits
            X_test_folds[fold_idx].append(val)
            index += 1
    
    X_train_folds = [[] for _ in range(0, n_splits)]

    for i in range(0, len(X)):
        for j in range(0, n_splits):
            if i not in X_test_folds[j]:
                X_train_folds[j].append(i)
                
    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []

    for i, yt in enumerate(labels):
        matrix.append([])
        for _, yp in enumerate(labels):
            matrix[i].append(0)

    for t, p in zip(y_true, y_pred):
        t_num = labels.index(t)
        p_num = labels.index(p)
        matrix[t_num][p_num] += 1

    return matrix 