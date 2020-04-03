import numpy as np


class CN2_SD:
    def __init__(self,
                 input_data,
                 col_output,
                 positive_class,
                 max_exp,
                 weight_method,
                 gamma=0.5):
        # Initialize attributes
        # Size of input data
        self.N = len(input_data.index)
        self.col_output = col_output
        self.positive_class = positive_class
        self.max_expr = max_exp
        # 0: none (not SD) | 1: multiplicative | 2: additive
        self.weight_method = weight_method
        # Parameter for multiplicative method (0 < gamma < 1)
        self.gamma = gamma
        # Initialize current_data (we keep track of weights)
        self.current_data = self.add_weights(input_data)

    def execute(self):
        rule_list = []
        end = False
        while not end:
            best_rule = self.find_best_rule()
            if best_rule:
                # Update weights
                self.apply_rule(best_rule)
                rule_list.append(best_rule)
            print(best_rule, best_rule.wracc)
            # print(self.current_data[self.current_data["weights"] < 1])
            if self.stop_condition(best_rule):
                end = True
        print("Rule list:")
        for rule in rule_list:
            print(rule, rule.wracc)
        return rule_list

    def find_best_rule(self):
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
        # TODO: What to do with real variables
        level_categories = ["object", "category", "int64"]
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
        new_rule_set = []
        if first:
            for sel in selector_list:
                rule = Rule()
                rule.add_antecedent(sel)
                new_rule_set.append(rule)
        else:
            for rule in rule_set:
                for sel in selector_list:
                    new_rule = Rule.rule_copy(rule)
                    new_rule.add_antecedent(sel)
                    if new_rule.is_valid():
                        new_rule_set.append(new_rule)
        return new_rule_set

    def eliminate_worst(self, rule_set):
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
        data_rule = self.current_data
        for antecedent in rule.antecedents:
            data_rule = data_rule[data_rule[antecedent.variable] == antecedent.value]
        # Multiplicative method
        if self.weight_method == 1:
            self.current_data.loc[data_rule.index, "weights"] = data_rule["weights"].apply(lambda x: x * self.gamma)
        # Additive method
        elif self.weight_method == 2:
            self.current_data.loc[data_rule.index, "weights"] = data_rule["weights"].apply(lambda x: 1 / (x+1))


    # TODO: end_condition associated with current_data
    def stop_condition(self, best_rule):
        return best_rule is None

    def add_weights(self, data):
        initial_weights = np.ones(len(data.index))
        data["weights"] = initial_weights
        return data

    def find_best(self, rule_set, best_rule):
        new_best = best_rule
        # Get values needed for WRAcc calculation
        N = sum(self.current_data["weights"])
        data_class = self.current_data[self.current_data[self.col_output] == self.positive_class]
        n_class = sum(data_class["weights"])
        for rule in rule_set:
            new_best = self.check_rule(rule, new_best, N, n_class)
        return new_best

    def check_rule(self, rule, new_best, N, n_class):
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

    def is_valid(self):
        variables = []
        for ant in self.antecedents:
            if ant.variable in variables:
                return False
            variables.append(ant.variable)
        return True

    def add_antecedent(self, antecedent):
        self.antecedents.append(antecedent)

    def __str__(self):
        msg = "Rule: "
        for ant in self.antecedents:
            msg += f"{str(ant.variable)} = {str(ant.value)} AND "
        return msg[:-4]


class Antecedent:
    # TODO: take into account operators for numeric variables
    # def __init__(self, variable, operator, value):
    def __init__(self, variable, value):
        self.variable = variable
        # self.operator = operator
        self.value = value

    def __str__(self):
        return "Antecedent: %s = %s" % (self.variable, str(self.value))