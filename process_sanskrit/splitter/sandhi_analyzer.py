#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intro
=====

Sandhi Analyzer for Sanskrit words

@author: Karthik Madathil (github: @kmadathil)

Usage
=====

Use the ``LexicalSandhiAnalyzer`` to split a sentence (wrapped in a
``SanskritObject``) and retrieve the top 10 splits:

.. code:: python

    >>> from process_sanskrit.splitter.sandhi_analyzer import LexicalSandhiAnalyzer
    >>> from process_sanskrit.splitter.sanskrit_base import SanskritObject
    >>> sentence = SanskritObject("astyuttarasyAMdishidevatAtmA")
    >>> analyzer = LexicalSandhiAnalyzer()
    >>> splits = analyzer.getSandhiSplits(sentence).find_all_paths(10)
    >>> for split in splits:
    ...    print(split)
    ...
    [u'asti', u'uttarasyAm', u'diSi', u'devatA', u'AtmA']
    [u'asti', u'uttarasyAm', u'diSi', u'devat', u'AtmA']
    [u'asti', u'uttarasyAm', u'diSi', u'devata', u'AtmA']

Upstream's morphological tagging (``getMorphologicalTags``/``hasTag``) is not
vendored: it is only reachable via ``Parser.parse()``, and Process-Sanskrit does
its own morphology in ``process_sanskrit.functions.inflect``. Dropping it is what
lets the lookup be a plain trie instead of two sqlite databases behind an ORM.
"""

import logging
from functools import lru_cache

from indic_transliteration import sanscript

from . import sanskrit_base as SanskritBase
from .datastructures import SandhiGraph
from .lookup import TrieLookup
from .sandhi import Sandhi

logger = logging.getLogger(__name__)


class LexicalSandhiAnalyzer(object):
    """ Singleton class to hold methods for Sanskrit lexical sandhi analysis.

        We define lexical sandhi analysis to be the process of taking an input sequence
        and transforming it to a collection (represented by a DAG) of potential sandhi
        splits of the sequence. Each member of a split is guaranteed to be a valid
        lexical form.
    """

    sandhi = Sandhi()  # Singleton!

    def __init__(self):
        self.forms = TrieLookup()

    def preSegmented(self, sl):
        ''' Get a SandhiGraph for a pre-segmented sentence

            Params:
              sl (list of SanskritString): Input object
            Returns:
              SandhiGraph : DAG all possible splits
        '''
        self.sentence = SandhiGraph()
        prev = None
        for s in sl[::-1]:
            self.sentence.add_node(s)
            if prev is None:
                self.sentence.add_end_edge(s)
            else:
                self.sentence.append_to_node(s, [prev])
            prev = s
        self.sentence.add_roots([prev])
        self.sentence.lock_start()
        return self.sentence

    def getSandhiSplits(self, o, pre_segmented=False):
        ''' Get all valid Sandhi splits for a string

            Params:
              o(SanskritString): Input object
            Returns:
              SandhiGraph : DAG all possible splits
        '''
        if pre_segmented:
            return self.preSegmented(o)
        self.dynamic_scoreboard = {}
        # Transform to internal canonical form
        s = o.canonical()
        # Initialize an empty graph to hold the splits
        self.splits = SandhiGraph()
        # _possible_splits updates graph in self.splits with nodes and returns roots
        roots = self._possible_splits(s)
        if len(roots) == 0:
            return None
        else:
            self.splits.add_roots(roots)
            return self.splits

    def _possible_splits(self, s):
        ''' private method to dynamically compute all sandhi splits

            Used by getSandhiSplits
            Adds the individual splits to the graph self.splits and returns
            the roots of the subgraph corresponding to the split of s
           Params:
              s(string): Input SLP1 encoded string
            Returns:
              roots : set of roots of subgraph corresponding to possible splits of s
        '''
        logger.debug("Splitting " + s)

        @lru_cache(256)
        def _is_valid_word(ss):
            r = self.forms.valid(ss)
            return r

        def _sandhi_splits_all(s, start=None, stop=None):
            obj = SanskritBase.SanskritImmutableString(s, encoding=sanscript.SLP1)
            splits = self.sandhi.split_all(obj, start, stop)
            return splits

        roots = set()

        # Memoization for dynamic programming - remember substrings that've
        # been seen before
        if s in self.dynamic_scoreboard:
            logger.debug("Found {} in scoreboard".format(s))
            return self.dynamic_scoreboard[s]

        # If a space is found in a string, stop at that space
        spos = s.find(" ")
        stop = None if spos == -1 else spos

        s_c_list = _sandhi_splits_all(s, start=0, stop=stop)
        logger.debug("s_c_list: " + str(s_c_list))
        if s_c_list is None:
            s_c_list = []

        node_cache = {}

        for (s_c_left, s_c_right) in s_c_list:
            # Is the left side a valid word?
            if _is_valid_word(s_c_left):
                logger.debug("Valid left word: " + s_c_left)
                # For each split with a valid left part, check it there are
                # valid splits of the right part
                if s_c_right and s_c_right != '':
                    logger.debug("Trying to split:" + s_c_right)
                    r_roots = self._possible_splits(s_c_right.strip())
                    # if there are valid splits of the right side
                    if r_roots:
                        # Make sure we got a set of roots back
                        assert isinstance(r_roots, set)
                        # if there are valid splits of the right side
                        if s_c_left not in node_cache:
                            # Extend splits list with s_c_left appended with
                            # possible splits of s_c_right
                            t = SanskritBase.SanskritObject(s_c_left, encoding=sanscript.SLP1)
                            node_cache[s_c_left] = t
                        else:
                            t = node_cache[s_c_left]
                        roots.add(t)
                        if not self.splits.has_node(t):
                            self.splits.add_node(t)
                        self.splits.append_to_node(t, r_roots)
                else:  # Null right part
                    # Why cache s_c_left here? To handle the case
                    # where the same s_c_left appears with a null and non-null
                    # right side.
                    if s_c_left not in node_cache:
                        t = SanskritBase.SanskritObject(s_c_left, encoding=sanscript.SLP1)
                        node_cache[s_c_left] = t
                    else:
                        t = node_cache[s_c_left]
                    # Extend splits list with s_c_left appended with
                    # possible splits of s_c_right
                    roots.add(t)
                    if not self.splits.has_node(t):
                        self.splits.add_node(t)
                    self.splits.add_end_edge(t)
            else:
                logger.debug("Invalid left word: " + s_c_left)
        # Update scoreboard for this substring, so we don't have to split
        # again
        self.dynamic_scoreboard[s] = roots
        if len(roots) == 0:
            logger.debug("No splits found, returning empty set")
        else:
            logger.debug("Roots: %s", roots)
        return roots
