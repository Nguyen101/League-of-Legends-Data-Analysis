import math
import random 
import numpy as np
import copy
import matplotlib.pyplot as plt

def least_squares(x,y):
    x_avg = sum(x)/len(x)
    y_avg = sum(y)/len(y)

    numerator = 0
    denominator = 0 
    for i in range(len(x)):
        numerator += (x[i] - x_avg) * (y[i] - y_avg)
        denominator += (x[i] - x_avg)**2
    
    m = numerator/denominator
    y_int = y_avg - m*x_avg

    return m, y_int

def comp_dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
      
def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None: 
            parallel_list[i], parallel_list[rand_index] = \
                parallel_list[rand_index], parallel_list[i]

def random_index (alist):
    return random.choice(alist)

def deep_copy_item(item):
    return copy.deepcopy(item)
    
def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def select_attribute(instances, att_indexes, available_attributes):
    attribute_array = []
    for i in att_indexes:
        attributes = {}
        for value in available_attributes["att"+str(i)]:
            attributes[value] = {}
        for instance in instances:
            att_val = instance[i]
            if instance[-1] in attributes[att_val]:
                attributes[att_val][instance[-1]] += 1
            else:
                attributes[att_val][instance[-1]] = 1
        attribute_array.append(attributes)
    smallest_not_set = True
    smallest = 0
    smallest_index = 0
    for i, attributes in enumerate(attribute_array):
        weighted_sum = 0
        for key in attributes:
            vals = []
            for key2 in attributes[key]:
                vals.append(attributes[key][key2])
            
            entropy_sum = 0
            for v in vals:
                entropy = -((v / sum(vals)) * math.log((v/ sum(vals)), 2))
                entropy_sum += entropy 
            weighted_sum += entropy_sum * (sum(vals) / len(instances))

        if weighted_sum <= smallest or smallest_not_set:
            smallest = weighted_sum
            smallest_index = i
            smallest_not_set = False
        
    return att_indexes[smallest_index]

def compute_partition_stats(instances, class_index):
    stats = {}
    for x in instances:
        if x[class_index] in stats:
            stats[x[class_index]] += 1
        else:
            stats[x[class_index]] = 1
    stats_array = []
    for key in stats:
        stats_array.append([key, stats[key], len(instances)])
    
    return stats_array
    

def tdidt(current_instances, att_indexes, att_domains):
    split_attribute = select_attribute(current_instances, att_indexes, att_domains)
    class_label = "att"+str(split_attribute)
    att_indexes2 = copy.deepcopy(att_indexes)
    att_indexes2.remove(split_attribute)
    
    partitions = {}
    attributes = att_domains[class_label]
    for a in attributes:
        partitions[a] = []
    for instance in current_instances:
        partitions[instance[split_attribute]].append(instance)
    
    tree = ["Attribute", "att"+str(split_attribute)]

    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        if len(partition) > 0 and all_same_class(partition):
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(att_indexes2) == 0:
            partition_stats = compute_partition_stats(partition, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            partition_stats = compute_partition_stats(current_instances, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            return leaf
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, att_indexes2, att_domains)
            values_subtree.append(subtree)
            tree.append(values_subtree)
    return tree

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def tdidt_print_rules(tree, rule, class_name, default_header, attribute_names):
    info_type = tree[0]
    if info_type == "Attribute":
        if rule != "":
            rule += " AND "
        else:
            rule += "IF "
        if attribute_names is None: 
            rule += tree[1]
        else:
            index = default_header.index(tree[1])
            rule += attribute_names[index]
            
        for i in range(2, len(tree)):
            value_list = tree[i]
            rule2 = rule + " = " + str(value_list[1])
            tdidt_print_rules(value_list[2], rule2, class_name, default_header, attribute_names)
    else: # "Leaf"
        print(rule, "THEN", class_name, "=", tree[1])


def distribute_data_by_index(data, indices):
    data_subset = []
    for i in range(len(indices)):
        data_subset.append(data[indices[i]])
    return data_subset

def get_trains_and_tests(X, y, X_train_fold, X_test_fold):
    X_train = [X[x] for x in X_train_fold]
    y_train = [y[x] for x in X_train_fold]
    X_test = [X[x] for x in X_test_fold]
    y_test = [y[x] for x in X_test_fold]
    return X_train, X_test, y_train, y_test

def get_win_count(table, win_col, column):
    win_column = table.get_column(win_col)
    oth_column = table.get_column(column)

    total = 0
    for i in range(len(win_column)):
        if win_column[i] == oth_column[i]:
            total += 1

    return total / len(win_column)

def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample

def get_available_attributes(table):
    available_attributes = {}
    for i in range(0, len(table[0])):
        att = "att"+str(i)
        available_attributes[att] = []
        for x in table:
            if x[i] not in available_attributes[att]:
                available_attributes[att].append(x[i])
    return available_attributes

def compute_random_subset(values, num_values):
    shuffled = values[:]
    random.shuffle(shuffled)
    return sorted(shuffled[:num_values])

def tdidt_random_forest(current_instances, att_indexes, att_domains, F):

    # print(att_indexes)
    att_indexes2 = copy.deepcopy(att_indexes)
    if(len(att_indexes) > F):
        compute_random_subset(att_indexes, F)
    split_attribute = select_attribute(current_instances, att_indexes2, att_domains)
    # print("TEST", split_attribute, "T", att_indexes)
    class_label = "att"+str(split_attribute)
    att_indexes2 = copy.deepcopy(att_indexes)
    att_indexes2.remove(split_attribute)
    
    partitions = {}
    attributes = att_domains[class_label]
    for a in attributes:
        partitions[a] = []
    for instance in current_instances:
        partitions[instance[split_attribute]].append(instance)
    
    tree = ["Attribute", "att"+str(split_attribute)]

    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]

        if len(partition) > 0 and all_same_class(partition):
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(att_indexes2) == 0:
            partition_stats = compute_partition_stats(partition, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)
            
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            partition_stats = compute_partition_stats(current_instances, -1)
            partition_stats.sort(key=lambda x: x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            return leaf
        else: # all base cases are false, recurse!!
            subtree = tdidt_random_forest(partition, att_indexes2, att_domains, F)
            values_subtree.append(subtree)
            tree.append(values_subtree)
    return tree

# functions added for the project
def bar_chart(x, y):
    if len(x) > 10:
        plt.figure(figsize=(18,5))
    else: 
        plt.figure()
    plt.bar(x, y, width=.5)
    plt.xticks(x, rotation=45, horizontalalignment="right", size='small')
    plt.ylabel("Game %")
    plt.show()

def pie_chart(labs, data):
    plt.figure()
    plt.pie(data, labels=labs, autopct="%1.1f%%")
    plt.show()
    
def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []

    for row in table: 
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def team_win_count(table, win_col):
    win_column = table.get_column(win_col)
    team_1_win = 0
    team_2_win = 0
    for i in range(len(win_column)):
        if win_column[i] == 1 :
            team_1_win += 1
        else:
            team_2_win += 1
    return team_1_win, team_2_win

def build_confusion_matrix(mat):
    for i in range(0, len(mat)):
        recognition = 0
        total = 0
        for j in range(0, len(mat[i])):
            if i == j:
                recognition += mat[i][j]
            total += mat[i][j]
        if total != 0:
            recognition = round((recognition / total) * 100, 2)
        mat[i].insert(0, i+1)
        mat[i].append(total)
        mat[i].append(recognition)