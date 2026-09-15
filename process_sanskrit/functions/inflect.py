"""Resolve split words and compound pieces before dictionary lookup."""

import time

from process_sanskrit.functions.rootAnyWord import root_any_word
from process_sanskrit.functions.compoundAnalysis import root_compounds


prefixes = ['sva', 'anu', 'sam', 'pra', 'upa', 'vi', 'nis', 'abhi', 'ni', 'pari', 'prati', 'parā', 'ava', 'adhi', 'api', 'ati', 'ud', 'dvi', 'su', 'dur', 'duḥ']


class Inflector:
    """Share one morphology fallback and memo across a sequence of words."""

    PREFIX_SPELLINGS = {"sam": ("sam", "saṃ"), "vi": ("vi", "vy")}

    def __init__(self, *, debug=False, session=None, memo=None):
        self.debug = debug
        self.session = session
        self.memo = {} if memo is None else memo

    def _lookup(self, word, *, compound=False):
        started = time.perf_counter() if self.debug else None
        lookup = root_compounds if compound else root_any_word
        options = {"inflection": True} if compound else {}
        result = lookup(word, session=self.session, _memo=self.memo, **options)
        if self.debug:
            print(f"{lookup.__name__}({word}) took {time.perf_counter() - started:.6f} seconds")
        return result

    def analyze(self, words):
        roots = []
        i = 0
        while i < len(words):
            word = words[i]
            if word in prefixes and i + 1 < len(words):
                next_word = words[i + 1]
                if self.debug:
                    print(f"Found prefix: {word}, next word: {next_word}")
                rooted = None
                for prefix in self.PREFIX_SPELLINGS.get(word, (word,)):
                    rooted = self._lookup(prefix + next_word)
                    if rooted:
                        break
                if rooted:
                    roots.extend(rooted)
                    i += 2
                    continue

            rooted = self._lookup(word)
            if not rooted:
                rooted = self._lookup(word, compound=True)
            # root_compounds returns [] on a complete miss. Preserve the token
            # so dictionary lookup can report an unresolved entry to the caller.
            roots.extend(rooted or [word])
            i += 1

        for index, root in enumerate(roots):
            if isinstance(root, list):
                root[0] = root[0].replace('-', '')
            else:
                roots[index] = root.replace('-', '')
        return roots


def inflect(splitted_text, debug=False, session=None, _memo=None):
    return Inflector(debug=debug, session=session, memo=_memo).analyze(splitted_text)
