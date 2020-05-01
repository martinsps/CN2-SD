import numpy as np
from utils import data_frame_difference


class CN2_SD:
    """
    Class that represents the CN2-SD algorithm, modification of the CN2
    algorithm, used for subgroup discovery.
    It is initialized with the following parameters:
    1. input_data: Data frame object with the input data in which subgroups
    are going to be discovered
    2. col_output: Name of the column that represents the output class value
    3. positive_class: Name of the class value chosen to be positive (the one
    that the algorithm is going to find subgroups)
    4. max_exp: Maximum of rules allowed in the rule set used for finding best rule
    5. min_wracc: Minimum value for a rule weighted relative accuracy for
    considering it a suitable one
    6. weight_method: Method for updating weights in the algorithm. Can be:
    0 -> not updating (normal CN2), 1 -> multiplicative or 2 -> additive
    7. gamma: Parameter (> 0 and < 1) that controls the flow of the multiplicative method
    """

    def __init__(self,
                 input_data,
                 col_output,
                 positive_class,
                 max_exp,
                 min_wracc,
                 weight_method,
                 gamma=0.5):
        """
        Initializes the CN2-SD algorithm object.
        :param input_data: Data frame object with the input data in which subgroups
        are going to be discovered
        :param col_output: Name of the column that represents the output class value
        :param positive_class: Name of the class value chosen to be positive (the one
        that the algorithm is going to find subgroups)
        :param max_exp: Maximum of rules allowed in the rule set used for finding best rule
        :param min_wracc: Minimum value for a rule weighted relative accuracy for
        considering it a suitable one
        :param weight_method: Method for updating weights in the algorithm. Can be:
        0 -> not updating (normal CN2), 1 -> multiplicative or 2 -> additive
        :param gamma: Parameter (> 0 and < 1) that controls the flow of the multiplicative method
        """
        # Initialize attributes
        # Size of input data
        self.N = len(input_data.index)
        self.col_output = col_output
        self.positive_class = positive_class
        self.max_expr = max_exp
        self.min_wracc = min_wracc
        # 0: none (not SD) | 1: multiplicative | 2: additive
        self.weight_method = weight_method
        # Parameter for multiplicative method (0 < gamma < 1)
        self.gamma = gamma
        # Initialize current_data (we keep track of weights)
        self.current_data = self.add_weights(input_data)
        # Initialize current rule list as empty
        self.rule_list = []

    def execute(self):
        """
        Executes the CN2-SD algorithm step by step. Prints the resulting rules at the end.
        :return:
        """
        end = False
        while not end:
            end, best_rule = self.do_step()
        print("Rule list:")
        for rule in self.rule_list:
            print(rule, rule.wracc)
        return self.rule_list

    def do_step(self):
        """
        Performs a step in the CN2-SD algorithm. First,
        it tries to find the best rule to apply, it applies it to the
        data (updating weights if necessary) and checks end condition.
        :return: End condition (True or False) and best rule found (can be "None")
        """
        end = False
        best_rule = self.find_best_rule()
        if best_rule and best_rule.wracc >= self.min_wracc:
            # Update weights
            self.apply_rule(best_rule)
            self.rule_list.append(best_rule)
        if self.stop_condition(best_rule):
            end = True
        return end, best_rule

    def find_best_rule(self):
        """
        Method that finds the best rule to apply to the current data to
        generate a subgroup with the highest weighted relative accuracy possible.
        It can not return a rule already used.
        :return: Best rule found ("None" if no valid rule is found)
        """
        rule_set = []
        best_rule = None
        selector_list = self.generate_selectors()
        # for sel in selector_list:
        #     print(sel)
        end_find = False
        first = True
        while not end_find:
            new_rule_set = self.generate_rule_set(rule_set, selector_list, first)
            first = False
            best_rule = self.find_best(new_rule_set, best_rule)
            while len(new_rule_set) > self.max_expr:
                self.eliminate_worst(new_rule_set)
            rule_set = new_rule_set
            if len(rule_set) == 0:
                end_find = True
        return best_rule

    def generate_selectors(self):
        """
        Method that generates all selectors (antecedents of a rule)
        possible in current data. Only takes into account variables of
        categorical ("object" and "category") or integer types. Real variables must be
        passed discretized into intervals and, thus, categorical.
        :return: A list of selectors
        """
        # Only categorical and integer variables are used:
        # real variables have to be in intervals
        level_categories = ["object", "category", "int64", "int32"]
        selectors = []
        for col_name in self.current_data.columns:
            if col_name != self.col_output:
                col_type = self.current_data[col_name].dtype
                if col_type.name in level_categories:
                    column = self.current_data[col_name].astype("category")
                    levels = column.unique()
                    for level in levels:
                        new_antecedent = Antecedent(col_name, level)
                        selectors.append(new_antecedent)
        return selectors

    def generate_rule_set(self, rule_set, selector_list, first):
        """
        Generates a rule set using current rule set and the list of
        selectors, making all pairs. If it is the first time this
        method is called (first attribute), only selectors are turned
        into rules because rule_set will be empty.
        It can not return a rule already used.
        :param rule_set: Current rule set
        :param selector_list: List of selectors / antecedent in current data
        :param first: Boolean that tells if it is first time the function is called
        :return: A new rule set made from previous one and selectors
        """
        new_rule_set = []
        if first:
            for sel in selector_list:
                rule = Rule()
                rule.add_antecedent(sel)
                if rule.is_valid(self.rule_list):
                    new_rule_set.append(rule)
        else:
            for rule in rule_set:
                for sel in selector_list:
                    new_rule = Rule.rule_copy(rule)
                    new_rule.add_antecedent(sel)
                    if new_rule.is_valid(self.rule_list):
                        new_rule_set.append(new_rule)
        return new_rule_set

    def eliminate_worst(self, rule_set):
        """
        Eliminates worst rule out a rule set, according
        to the rules' weighted relative accuracy.
        :param rule_set:
        :return:
        """
        worst_wracc = 1
        worst_rule = None
        for rule in rule_set:
            if rule.wracc < worst_wracc:
                worst_wracc = rule.wracc
                worst_rule = rule
        if worst_rule:
            # print("We eliminate", worst_rule)
            rule_set.remove(worst_rule)

    def apply_rule(self, rule):
        """
        Applies the rule parameter updating the weights of the
        elements with positive class according to the method
        selected (multiplicative or additive)
        :param rule: Rule to apply
        :return: Nothing, current data is updated from the algorithm's variable
        """
        data_rule = self.current_data
        for antecedent in rule.antecedents:
            data_rule = data_rule[data_rule[antecedent.variable] == antecedent.value]
        # Only the ones of the positive class
        data_rule_positive = data_rule[data_rule[self.col_output] == self.positive_class]
        # Multiplicative method
        if self.weight_method == 1:
            self.current_data.loc[data_rule_positive.index, "weights"] = data_rule_positive["weight_times"].apply(
                lambda x: self.gamma ** x)

        # Additive method
        elif self.weight_method == 2:
            self.current_data.loc[data_rule_positive.index, "weights"] = data_rule_positive["weight_times"].apply(
                lambda x: 1 / (x + 1))
        # Normal CN2: examples covered eliminated
        elif self.weight_method == 0:
            self.current_data = data_frame_difference(self.current_data, data_rule)
        self.current_data.loc[data_rule_positive.index, "weight_times"] = data_rule_positive["weight_times"].apply(
            lambda x: x + 1)

    def stop_condition(self, best_rule):
        """
        Determines if the algorithm has ended. It can be because of three reasons:
        1. No good rule has been found (best_rule is "None")
        2. Best rule is not valid (its relative weighted accuracy is lower than
        the minimum established)
        3. Sum of weights of positive class examples is "almost zero"
        :param best_rule:
        :return:
        """
        almost_zero = 0.1
        positive_examples = self.current_data[self.current_data[self.col_output] == self.positive_class]
        total_positive = len(positive_examples)
        total_almost_zero = positive_examples["weights"] < almost_zero
        return best_rule is None or best_rule.wracc < self.min_wracc or sum(total_almost_zero) == total_positive

    def add_weights(self, data):
        """
        Adds a column of weights to the data, initialized
        to ones and a column of weight times, to control
        how many times weights have been updated.
        :param data: Initial data
        :return: Data with the added weights column
        """
        initial_weights = np.ones(len(data.index))
        data["weights"] = initial_weights
        updates = np.zeros(len(data.index))
        data["weight_times"] = updates
        return data

    def find_best(self, rule_set, best_rule):
        """
        Finds best rule within a rule set and a rule chosen
        before as best rule. For comparing, weighted relative
        accuracy is used.
        :param rule_set: List of rules to compare
        :param best_rule: Best rule found so far (may be "None")
        :return: New best rule
        """
        new_best = best_rule
        # Get values needed for WRAcc calculation
        N = sum(self.current_data["weights"])
        data_class = self.current_data[self.current_data[self.col_output] == self.positive_class]
        n_class = sum(data_class["weights"])
        for rule in rule_set:
            new_best = self.check_rule(rule, new_best, N, n_class)
        return new_best

    def check_rule(self, rule, new_best, N, n_class):
        """
        Compares two rules (rule and new_best), checking if rule is
        better than new best and returns the better one.
        :param rule: Rule to compare
        :param new_best: Best rule found so far (can be "None")
        :param N: Sum of weights of all examples
        :param n_class: Sum of weights of positive examples
        :return: Best rule out of those two
        """
        data_cond = self.current_data
        for antecedent in rule.antecedents:
            data_cond = data_cond[data_cond[antecedent.variable] == antecedent.value]
        rule.wracc = self.calculate_WRAcc(data_cond, N, n_class)
        if new_best is None:
            return rule
        else:
            if rule.wracc > new_best.wracc:
                return rule
            else:
                return new_best

    def calculate_WRAcc(self, data_cond, N, n_class):
        """
        Calculates weighted relative accuracy of a data set
        derived from applying a rule to the data.
        :param data_cond: Data after applying the rule antecedents
        :param N: Sum of weights of all examples
        :param n_class: Sum of weights of positive examples
        :return: WRAcc of the data (of the rule)
        """
        n_cond = sum(data_cond["weights"])
        data_cond_correct = data_cond[data_cond[self.col_output] == self.positive_class]
        n_class_cond = sum(data_cond_correct["weights"])
        if n_cond == 0:
            return 0
        return (n_cond / N) * ((n_class_cond / n_cond) - (n_class / N))


class Rule:

    @staticmethod
    def rule_copy(rule):
        new_rule = Rule()
        new_rule.wracc = rule.wracc
        for ant in rule.antecedents:
            new_rule.add_antecedent(ant)
        return new_rule

    def __init__(self):
        self.antecedents = []
        self.wracc = 0

    def is_valid(self, current_rule_list):
        """
        Checks if a rule is valid. It is not valid if one of these
        happens:
        1. It is null or impossible (i.e. a = 1 AND a = 2)
        2. It is already in the current rule list
        :return: True if it is valid, False if it is not
        """
        variables = []
        for ant in self.antecedents:
            if ant.variable in variables:
                return False
            variables.append(ant.variable)
        for rule in current_rule_list:
            # if rule.antecedents[0].variable == "Sex":
            #     print(f"Comparando {rule} con {self} y sale {self.is_equal(rule)}")
            if self.is_equal(rule):
                return False
        return True

    def is_equal(self, rule):
        """
        Checks if the rule is the same as the one passed
        as a parameter.
        """
        if len(self.antecedents) != len(rule.antecedents):
            return False
        coincidences = 0
        for antecedent in self.antecedents:
            for new_antecedent in rule.antecedents:
                if antecedent.variable == new_antecedent.variable and antecedent.value == new_antecedent.value:
                    coincidences += 1
        return coincidences == len(self.antecedents)

    def add_antecedent(self, antecedent):
        self.antecedents.append(antecedent)

    def __str__(self):
        msg = "Rule: "
        for ant in self.antecedents:
            msg += f"{str(ant.variable)} = {str(ant.value)} AND "
        return msg[:-4]


class Antecedent:
    def __init__(self, variable, value):
        self.variable = variable
        # self.operator = operator
        self.value = value

    def __str__(self):
        return "Antecedent: %s = %s" % (self.variable, str(self.value))
