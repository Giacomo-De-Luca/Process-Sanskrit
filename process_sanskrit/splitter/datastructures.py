#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
""" Sandhi split DAG.

    Vendored from kmadathil/sanskrit_parser (MIT), reduced to SandhiGraph.
    Upstream's VakyaGraph -- the dependency parser reached via Parser.parse() --
    lived in this module and accounted for ~1,100 of its 1,373 lines. It is the
    only consumer of DhatuWrapper (tinydb) and DisjointSet. Process-Sanskrit
    calls Parser.split() and nothing else, so both are dropped along with it.

    @author: Karthik Madathil (github: @kmadathil)
"""
import logging
import operator
from itertools import islice

import networkx as nx

from . import scorer

__all__ = ['SandhiGraph']

logger = logging.getLogger(__name__)

class SandhiGraph(object):
    """ DAG class to hold Sandhi Lexical Analysis Results

        Represents the results of lexical sandhi analysis as a DAG
        Nodes are SanskritObjects
    """
    start = "__start__"
    end = "__end__"

    def __init__(self):
        ''' DAG Class Init

        Params:
            elem (SanskritObject :optional:): Optional initial element
            end  (bool :optional:): Add end edge to initial element
        '''
        self.roots = []
        self.G = nx.DiGraph()
        self.scorer = scorer.Scorer()

    def __iter__(self):
        ''' Iterate over nodes '''
        return self.G.__iter__()

    def has_node(self, t):
        ''' Does a given node exist in the graph?

            Params:
               t (SanskritObject): Node
            Returns:
               boolean
        '''
        return t in self.G

    def append_to_node(self, t, nodes):
        """ Create edges from t to nodes

            Params:
                t (SanskritObject)      : Node to append to
                nodes (iterator(nodes)) : Nodes to append to t
        """
        # t is in our graph
        assert t in self.G
        for r in nodes:
            self.G.add_edge(t, r)

    def add_node(self, node):
        """ Extend dag with node inserted at root

            Params:
                Node (SanskritObject)      : Node to add
                root (Boolean)             : Make a root node
                end  (Boolean)             : Add an edge to end
        """
        assert node not in self.G
        self.G.add_node(node)

    def add_end_edge(self, node):
        ''' Add an edge from node to end '''
        assert node in self.G
        self.G.add_edge(node, self.end)

    def add_roots(self, roots):
        self.roots.extend(roots)

    def lock_start(self):
        ''' Make the graph ready for search by adding a start node

        Add a start node, add arcs to all current root nodes, and clear
        self.roots
        '''
        self.G.add_node(self.start)
        for r in self.roots:
            self.G.add_edge(self.start, r)
        self.roots = []

    def score_graph(self):
        edges = self.G.edges()
        edges_list = []
        edges_to_score = []
        for edge in edges:
            edges_list.append(edge)
            if edge[0] == self.start:
                edges_to_score.append([edge[1]])
            elif edge[1] == self.end:
                edges_to_score.append([edge[0]])
            else:
                edges_to_score.append(edge)
        scores = self.scorer.score_splits(edges_to_score)
        for edge, score in zip(edges_list, scores):
            # Score is log-likelihood, so higher is better.
            # For graph path-finding, smaller weight is better, so use negative
            edges[edge]['weight'] = -score
        for u, v, w in self.G.edges(data='weight'):
            logger.debug("u = %s, v = %s, w = %s", u, v, w)

    def find_all_paths(self, max_paths=10, sort=True, score=True):
        """ Find all paths through DAG to End

            Params:
               max_paths (int :default:=10): Number of paths to find
                          If this is > 1000, all paths will be found
               sort (bool)                 : If True (default), sort paths
                                             in ascending order of length
        """
        if self.roots:
            self.lock_start()
        if score:
            self.score_graph()
        # shortest_simple_paths is slow for >1000 paths
        if max_paths <= 1000:
            if score:
                paths = list(map(lambda x: x[1:-1],
                                           islice(nx.shortest_simple_paths(
                                                        self.G, self.start, self.end, weight='weight'),
                                                  max_paths)))
                scores = self.scorer.score_splits(paths)
                path_scores = zip(paths, scores)
                sorted_path_scores = sorted(path_scores, key=operator.itemgetter(1), reverse=True)
                logger.debug("Sorted paths with scores:\n %s", sorted_path_scores)
                # Strip the scores from the returned result, to be consistent with no-scoring option
                sorted_paths, _ = zip(*sorted_path_scores)
                return list(sorted_paths)
            else:
                paths = list(map(lambda x: x[1:-1],
                                           islice(nx.shortest_simple_paths(
                                                        self.G, self.start, self.end),
                                                  max_paths)))
                return paths
        else:  # Fall back to all_simple_paths
            ps = list(map(lambda x: x[1:-1],
                                    nx.all_simple_paths(self.G, self.start, self.end)))
            # If we do not intend to display paths, no need to sort them
            if sort:
                ps.sort(key=lambda x: len(x))
            return ps

    def __str__(self):
        """ Print representation of DAG """
        return str(self.G)

    def draw(self, *args, **kwargs):
        _ncache = {}

        def _uniq(s):
            if s not in _ncache:
                _ncache[s] = 0
                return s
            else:
                _ncache[s] = _ncache[s] + 1
                return s + "_" + str(_ncache[s])

        nx.draw(self.G, *args, **kwargs,
                pos=nx.spring_layout(self.G),
                labels={x: _uniq(str(x)) for x in self})

    def write_dot(self, path):
        # H = nx.convert_node_labels_to_integers(G, label_attribute=’node_label’)
        # H_layout = nx.nx_pydot.pydot_layout(G, prog=’dot’)
        # G_layout = {H.nodes[n][‘node_label’]: p for n, p in H_layout.items()}
        nx.drawing.nx_pydot.write_dot(self.G, path)

