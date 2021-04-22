import copy
import mysklearn.myutils as myutils

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x_train = []
        for x in X_train:
            x_train.append(x[0])
        self.slope, self.intercept = myutils.least_squares(x_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            y_predicted.append((x[0] * self.slope) + self.intercept)

        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for i, row in enumerate(X_test):
            distances.append([])
            neighbor_indices.append([])
            rowx = row[0]
            rowy = row[1]
            for j, row_train in enumerate(self.X_train):
                row_train.append(self.y_train[j])
                row_train.append(j)
                row_train.append(myutils.comp_dist(rowx, rowy, row_train[0], row_train[1]))
            # sort by distance
            self.X_train.sort(key = lambda x: x[4])
            
            for k in range(self.n_neighbors):
                distances[i].append(self.X_train[k][4])
                neighbor_indices[i].append(self.X_train[k][3])
            
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        dists, idxs = self.kneighbors(X_test)
        results = []
        for j in dists:
            for i in range(len(j)):
                results.append(self.y_train[int(idxs[0][i])])
            y_predicted.append(max(set(results), key = results.count))

        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.priors = {}
        self.posteriors = {}

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        #extract
        x_train = []
        for x in X_train:
            x_train.append(x[0])
        # extract unique class
        data_class = []
        for y_class in y_train:
            if y_class not in data_class:
                data_class.append(y_class)
        
        # intialize priors each class:
        for classes in data_class:
            self.priors[classes] = 0.00
            self.posteriors[classes] = {}

        # calculate priors 
        for classes in data_class:
            count = 0
            for row in y_train:
                if classes == row:
                    count += 1
            self.priors[classes] = float(count / len(y_train))
        # compute how many attributes there are and how many unique value for each attribute
        attribute_num = len(X_train[0])
        attribute_value = []
        for num in range(attribute_num):
            att_val = []
            for row in X_train:
                if str(row[num]) not in att_val:
                    att_val.append(str(row[num]))
            attribute_value.append(att_val)
        # calculate posteriors
        for c_value in data_class:
            for i in range(attribute_num):
                for value in attribute_value[i]:
                    count = 0
                    for row_num, row in enumerate(X_train):
                        if y_train[row_num] == c_value:
                            if str(X_train[row_num][i]) == value:
                                count += 1
                    probability = count / y_train.count(c_value)
                    self.posteriors[c_value].update({"attribute(" + str(i)  + ")value(" + str(value) + ")": float(probability)})
                

                    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for j in range(len(X_test)):
            y_predicted.append({})
            for i in self.posteriors.keys():
                y_predicted[j][i] = 1

        # iterate through each test
        for test_num, test in enumerate(X_test):
            # iterate through each value in test
            for i, val in enumerate(test):
                # iterate through each possible outcome
                for outcome in self.posteriors:
                    # build string with attribute and value info (to access posteriors)
                    chk_str = f'attribute({i})value({val})'
                    try:
                        y_predicted[test_num][outcome] *= self.posteriors[outcome][chk_str]
                    except KeyError:
                        pass

        # multiply by priors
        for result in y_predicted:
            for key in result.keys():
                result[key] = result[key] * self.priors[key]

        new_y_predicted = []
        ymax = 0
        yidx = 0
        for i, result in enumerate(y_predicted):
            rvals = list(result.values())
            rkeys = list(result.keys())
            new_y_predicted.append(rkeys[rvals.index(max(rvals))])
        
        return new_y_predicted



class MyZeroRClassifier:
    def __init__(self):
        self.prediction = None

    def fit(self, y_train):
        values = [y for y in y_train]
        self.prediction = max(set(values), key= values.count)


    def predict(self, X_test):
        predicted = [self.prediction for i in range(len(X_test))]
        return predicted

class MyRandomClassifier:
    def __init__(self):
        self.prediction = None

    def fit(self, y_train):
        index = myutils.random_index(y_train)
        self.prediction = y_train[index]

    def predict(self, X_test):
        y_predictions = [self.predictions for _ in X_test]
        return y_predictions 

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)

        X_train2 = copy.deepcopy(X_train)
        # construct a dictionary og possible values in the form {attribute: values}
        available_attributes = {}
        for i in range(0, len(X_train[0])):
            att = "att"+str(i)
            available_attributes[att] = []
            for x in X_train:
                if x[i] not in available_attributes[att]:
                    available_attributes[att].append(x[i])

        for i,x in enumerate(y_train):
            X_train2[i].append(x)
        tree = myutils.tdidt(X_train2, [x for x in range(0, len(X_train2[0])-1)], available_attributes)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = []
        predictions = []
        for i in range(0, len(X_test[0])):
            header.append("att" + str(i))
        for instance in X_test:
            prediction = myutils.tdidt_predict(header, self.tree, instance)
            predictions.append(prediction)
        return predictions


    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        default_header = ["att"+str(i) for i in range(0, len(self.X_train))]
        myutils.tdidt_print_rules(self.tree, "", class_name, default_header, attribute_names)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this