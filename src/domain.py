# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import re
import numpy as np
from collections import OrderedDict


def get_domain(name):
    if name == 'object_division':
        return ObjectDivisionDomain()
    if name == 'trade':
        return ObjectTradeDomain()
    raise()


class Domain(object):
    """ Domain interface. """
    def selection_length(self):
        pass

    def input_length(self):
        pass

    def generate_choices(self, input):
        pass

    def parse_context(self, ctx):
        pass

    def score(self, context, choice):
        pass

    def parse_choice(self, choice):
        pass

    def parse_human_choice(self, input, output):
        pass

    def score_choices(self, choices, ctxs):
        pass


class ObjectDivisionDomain(Domain):
    def __init__(self):
        self.item_pattern = re.compile('^item([0-9])=(-?[0-9])+$')

    def selection_length(self):
        return 6

    def input_length(self):
        return 3

    def num_choices(self):
        return len(self.idx2sel)

    def generate_choices(self, input, with_disagreement=True):
        cnts, _ = self.parse_context(input)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                right_choice = ['item%d=%d' % (i, n - c) for i, (n, c) in enumerate(zip(cnts, choice))]
                return [left_choice + right_choice]
            choices = []
            for c in range(cnts[idx] + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()
            return choices
        choices = gen(cnts)
        if with_disagreement:
            choices.append(['<no_agreement>'] * self.selection_length())
            choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def parse_context(self, ctx):
        cnts = [int(n) for n in ctx[0::2]]
        vals = [int(v) for v in ctx[1::2]]
        return cnts, vals

    def score(self, context, choice):
        assert len(choice) == (self.selection_length())
        choice = choice[0:len(choice) // 2]
        if choice[0] in ('<no_agreement>', '<disconnect>', '<disagree>'):
            return 0
        _, vals = self.parse_context(context)
        score = 0
        for i, (c, v) in enumerate(zip(choice, vals)):
            idx, cnt = self.parse_choice(c)
            # Verify that the item idx is correct
            assert idx == i
            score += cnt * v
        return score

    def parse_choice(self, choice):
        match = self.item_pattern.match(choice)
        assert match is not None, 'choice %s' % choice
        # Returns item idx and it's count
        return (int(match.groups()[0]), int(match.groups()[1]))

    def parse_human_choice(self, input, output):
        cnts = self.parse_context(input)[0]
        choice = [int(x) for x in output.strip().split()]

        if len(choice) != len(cnts):
            raise
        for x, n in zip(choice, cnts):
            if x < 0 or x > n:
                raise
        return ['item%d=%d' % (i, x) for i, x in enumerate(choice)]

    def _to_int(self, x):
        try:
            return int(x)
        except:
            return 0

    def score_choices(self, choices, ctxs):
        assert len(choices) == len(ctxs)
        cnts = [int(x) for x in ctxs[0][0::2]]
        agree, scores = True, [0 for _ in range(len(ctxs))]
        for i, n in enumerate(cnts):
            for agent_id, (choice, ctx) in enumerate(zip(choices, ctxs)):
                taken = self._to_int(choice[i][-1])
                n -= taken
                scores[agent_id] += int(ctx[2 * i + 1]) * taken
            agree = agree and (n == 0)
        return agree, scores

    def is_pareto_opt(self, contexts, choices):
        """Checks whether an agreed choice is pareto optimal."""
        #print(contexts)
        #print(choices)
        cnts = [int(x) for x in contexts[0][0::2]]
        _, vals1 = self.parse_context(contexts[0])
        _, vals2 = self.parse_context(contexts[1])
        picks1 = [self._to_int(choice[-1]) for choice in choices[0][:3]]
        picks2 = [self._to_int(choice[-1]) for choice in choices[1][:3]]

        agree = True
        if np.min(picks1) == -1 or np.min(picks2) == -1:
            agree= False
        
        for p1, p2, n in zip(picks1, picks2, cnts):
            agree = agree and (p1 + p2 == n)

        if not agree:
            return int(agree)

        score1 = self.compute_score(vals1, picks1)
        score2 = self.compute_score(vals2, picks2)
        choices = self.gen_choices(cnts)
        can_improve = False
        for cand1, cand2 in choices:
            cand_score1 =  self.compute_score(vals1, cand1)
            cand_score2 =  self.compute_score(vals2, cand2)
            if (cand_score1 > score1 and cand_score2 >= score2) or (cand_score1 >= score1 and cand_score2 > score2):
                can_improve = True

        optimal = int(not can_improve)
        return optimal

    def compute_score(self, vals, picks):
        """Compute the score of the selection."""
        assert len(vals) == len(picks)
        return np.sum([v * p for v, p in zip(vals, picks)])


    def gen_choices(self, cnts, idx=0, choice=[]):
        """Generate all the valid choices.
        It generates both yours and your opponent choices.
        """
        if idx >= len(cnts):
            return [(choice[:], [n - c for n, c in zip(cnts, choice)]),]
        choices = []
        for c in range(cnts[idx] + 1):
            choice.append(c)
            choices +=  self.gen_choices(cnts, idx + 1, choice)
            choice.pop()
        return choices


class ObjectTradeDomain(ObjectDivisionDomain):
    def __init__(self, max_items=1):
        super(ObjectTradeDomain, self).__init__()
        self.max_items = max_items

    def selection_length(self):
        return 3

    def input_length(self):
        return 3

    def generate_choices(self, input):
        cnts, _ = self.parse_context(input)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                return [left_choice]
            choices = []
            for c in range(-cnts[idx], self.max_items + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()

            return choices
        choices = gen(cnts)
        choices.append(['<no_agreement>'] * self.selection_length())
        choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def score_choices(self, choices, ctxs):
        assert len(choices) == len(ctxs)
        cnts = [int(x) for x in ctxs[0][0::2]]
        agree, scores = True, [0 for _ in range(len(ctxs))]
        for i in range(len(cnts)):
            n = 0
            for agent_id, (choice, ctx) in enumerate(zip(choices, ctxs)):
                taken = self._to_int(choice[i][choice[i].find("=") + 1:])
                n += taken
                scores[agent_id] += int(ctx[2 * i + 1]) * taken
            agree = agree and (n == 0)
        return agree, scores

    def score(self, context, choice):
        assert len(choice) == (self.selection_length())
        if choice[0] == '<no_agreement>':
            return 0
        _, vals = self.parse_context(context)
        score = 0
        for i, (c, v) in enumerate(zip(choice, vals)):
            idx, cnt = self.parse_choice(c)
            # Verify that the item idx is correct
            assert idx == i
            score += cnt * v
        return score

    def parse_human_choice(self, input, output):
        cnts = self.parse_context(input)[0]
        choice = [int(x) for x in output.strip().split()]

        if len(choice) != len(cnts):
            raise
        for x, n in zip(choice, cnts):
            if x < -n or x > 4:
                raise
        return ['item%d=%d' % (i, x) for i, x in enumerate(choice)]
