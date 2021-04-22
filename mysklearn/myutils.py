import math
from os import major
import random 
import numpy as np
import copy

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
def mpg_to_doe(val):
    if val <= 13:
        return 1
    elif 13 < val < 15:
        return 2
    elif 15 <= val < 17:
        return 3
    elif 17 <= val < 20:
        return 4
    elif 20 <= val < 24 :
        return 5
    elif 24 <= val < 27:
        return 6
    elif 27 <= val < 31:
        return 7
    elif 31 <= val < 37:
        return 8
    elif 37 <= val < 45:
        return 9
    elif val >= 45:
        return 10
    else:
        return "error: val not in bins"

def deep_copy_item(item):
    return copy.deepcopy(item)
    
def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def select_attribute(instances, att_indexes, available_attributes):
    attribute_array = []
    # for each avalailable index
    for i in att_indexes:
        attributes = {}
        # get the frequency of each value
        # eg: {senior: {true: 1, false, 2}, mid: {true: 3, false, 3}}
        for value in available_attributes["att"+str(i)]:
            attributes[value] = {}
        for instance in instances:
            att_val = instance[i]
            if instance[-1] in attributes[att_val]:
                attributes[att_val][instance[-1]] += 1
            else:
                attributes[att_val][instance[-1]] = 1
        attribute_array.append(attributes)
    # loop through all the attributes and get the smallest weighted sum of entropy
    smallest_not_set = True
    smallest = 0
    smallest_index = 0
    for i, attributes in enumerate(attribute_array):
        weighted_sum = 0
        # get the entropy of each
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