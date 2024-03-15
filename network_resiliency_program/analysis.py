# Nathan Zhao
# njz1
# COMP 182 Spring 2024 - Homework 6, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from comp182.py, provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...

from collections import *
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
import numpy
import random

## Graph functions

def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g

def write_graph(g, filename):
    """
    Write a graph to a file.  The file will be in a format that can be
    read by the read_graph function.

    Arguments:
    g        -- a graph
    filename -- name of the file to store the graph

    Returns:
    None
    """
    with open(filename, 'w') as f:
        f.write(repr(g))

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

## Timing functions

def time_func(f, args=[], kw_args={}):
    """
    Times one call to f with args, kw_args.

    Arguments:
    f       -- the function to be timed
    args    -- list of arguments to pass to f
    kw_args -- dictionary of keyword arguments to pass to f.

    Returns: 
    a tuple containing the result of the call and the time it
    took (in seconds).

    Example:

    >>> def sumrange(low, high):
            sum = 0
            for i in range(low, high):
                sum += i
            return sum
    >>> time_func(sumrange, [82, 35993])
    (647726707, 0.01079106330871582)
    >>> 
    """
    start_time = time.time()
    result = f(*args, **kw_args)
    end_time = time.time()

    return (result, end_time - start_time)

## Plotting functions

def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()

def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a bar plot on a linear
    scale.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, False, filename)

def plot_dist_loglog(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a scatter plot on a
    loglog scale.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, True, filename)


def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False

    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))
        

def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
    Plot the distribution provided in data.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    scatter  -- True for loglog scatter plot, False for linear bar plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a dictionary
    if not isinstance(data, dict):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)
    
    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)

    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False), 
                      _pow_10_round(max(data.keys()))])
        gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]), False), 
                      _pow_10_round(max(data.values()))])

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars. 

    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals)+1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals)+1])

def _plot_dict_scatter(d):
    """
    Plot data in the dictionary d on the current plot as points. 

    Arguments:
    d     -- dictionary

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)

def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g            

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with 
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g


def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))

def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
 
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.
 
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
 
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
         
    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value: 
                result[node_key].add(node_value)
 
    return result

def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns: 
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.  
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result



# original code from here
"""
Test cases:
test_graph_1 = {1:{2, 3, 4}, 2:{1, 6}, 3: {1, 5, 6}, 4: {1}, 5: {3, 6}, 6: {2, 3, 5}}
test_graph_2 = {1: {2, 3, 4}, 2: {1}, 3: {1, 4}, 4: {1, 3}}
"""

def compute_largest_cc_size(g: dict) -> int:
    """
    Computes the size of the largest connected component of a given undirected graph.

    Inputs:
        -g: a dictionary where each key represents a node and its value is a set of its neighbors

    Returns an integer that represents the size of the largest connected component
    """
    # Your code here...
    largest_size = 0
    seen = set()
    for node in g:
        if node not in seen:
            size = 1
            seen.add(node)
            queue = [node]
            visited = set()
            while queue:
                vertex = queue.pop(0)
                for neighbor in g[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        seen.add(neighbor)
                        visited.add(neighbor)
                        size += 1
            if size > largest_size:
                largest_size = size
    return largest_size

"""
Test cases:
test_graph_1 = {1:{2, 3, 4}, 2:{1, 6}, 3: {1, 5, 6}, 4: {1}, 5: {3, 6}, 6: {2, 3, 5}}
test_graph_2 = {1: {2, 3, 4}, 2: {1}, 3: {1, 4}, 4: {1, 3}}
"""

def random_attack(g):
    """
    Removes nodes randomly from a copy of graph g and collects data on the connection between removed
    nodes and the size of the largest connected component of the resulting graph

    Inputs:
        - g: an undirected graph

    Returns data: a dictionary that keys removed nodes to size of the largest connected component
    """
    graph = copy_graph(g)
    data = {}
    removed = 0
    for i in range(len(g) // 5):
        removed_node = random.choice(list(graph.keys()))
        for removed_neighbor in graph.pop(removed_node):
            graph[removed_neighbor].remove(removed_node)
        
        removed += 1
        data[removed] = compute_largest_cc_size(graph)

    return data

"""
Test cases:
test_graph_1 = {1:{2, 3, 4}, 2:{1, 6}, 3: {1, 5, 6}, 4: {1}, 5: {3, 6}, 6: {2, 3, 5}}
test_graph_2 = {1: {2, 3, 4}, 2: {1}, 3: {1, 4}, 4: {1, 3}}
"""

def targeted_attack(g):
    """
    Removes nodes starting from nodes of the highest degree and decerasing, from a copy of graph g and 
    collects data on the connection between removed nodes and the size of the largest 
    connected component of the resulting graph

    Inputs:
        - g: an undirected graph

    Returns data: a dictionary that keys removed nodes to size of the largest connected component
    """
    graph = copy_graph(g)
    removed = 0
    data = {}

    while True:
        max_degree = -1
        removed_node = -1
        for prospective_node in graph:
            if len(graph[prospective_node]) > max_degree:
                max_degree = len(graph[prospective_node])
                removed_node = prospective_node

        if removed > len(g) // 5:
            break
        for removed_neighbor in graph.pop(removed_node):
            graph[removed_neighbor].remove(removed_node)
            
        removed += 1

        data[removed] = compute_largest_cc_size(graph)

    return data

"""
The following is the analysis on the given graph using the given models and functions.
"""

real_graph = read_graph("rf7.repr")

# computing parameters for modelling graphs for future comparison

num_nodes = len(real_graph)
num_edges = total_degree(real_graph) / 2
m = int(num_edges // num_nodes)
p = num_edges / ((num_nodes * (num_nodes - 1)) / 2)

upa_graph = upa(num_nodes, m)
er_graph = erdos_renyi(num_nodes, p)

real_data_rand = random_attack(real_graph)
real_data_targ = targeted_attack(real_graph)

upa_data_rand = random_attack(upa_graph)
upa_data_targ = targeted_attack(upa_graph)

er_data_rand = random_attack(er_graph)
er_data_targ = targeted_attack(er_graph)

plt.plot(real_data_rand.keys(), real_data_rand.values(), label="rf7 random")
plt.plot(real_data_targ.keys(), real_data_targ.values(), label="rf7 targeted")
plt.plot(upa_data_rand.keys(), upa_data_rand.values(), label="upa random")
plt.plot(upa_data_targ.keys(), upa_data_targ.values(), label="upa targeted")
plt.plot(er_data_rand.keys(), er_data_rand.values(), label="erdos random")
plt.plot(er_data_targ.keys(), er_data_targ.values(), label="erdos targeted")

plt.xlabel("Number of Nodes Removed")
plt.ylabel("Size of the Largest Connected Component")
plt.title("Largest Connected Component Vs. Number of Removed Nodes")
plt.legend()
plt.show()
