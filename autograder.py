# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 4, Problem 3

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from comp182.py and provided.py, but they have
# to be copied over here.

from collections import *

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