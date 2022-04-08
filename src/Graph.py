from itertools import combinations_with_replacement as comb
from itertools import chain
class Graph():

    def __init__(self, integration, rule, height, deg, derivative = False):
        # Integration function that creates integration({tree: values}) returns {I[tree]: I[values]}
        self.I = integration 
        self.deg = deg # maximum degree of the created trees
        self.H = height # maximum height of the trees 
        self.R = rule  # Rule involving several extra trees and widths
        self.models = None
        self.size = 0 # number of realizations of models
        self.derivative = derivative # True if derivatives are present in the model. At the moment only differentiation order <= 1 are allowed

    # A helper function that returns degree of the tree with dictionary dic.
    def tree_deg(self, dic, done):
        return sum([done[w] * dic[w] for w in dic])

    # Helper function that multiplies trees. 
    # Given a dictionary {tree: power} outputs \prod_{tree} tree^power
    def trees_multiply(self, model, dic):
        
        trees = list(dic.keys())
        w1 = trees[0]
        
        if len(dic) == 1:  # If there is only one tree, it is faster to just return tree^n
            return model[w1] ** dic[w1]
        if len(dic) == 2:  # If only two unique trees is multiplied faster to return this
            w2 = trees[1]
            return (model[w1] ** dic[w1]) * (model[w2] ** dic[w2])

        tree_val = model[w1] ** dic[w1]
        for i in range(1,len(trees)):
            tree_val *= model[trees[i]]**dic[trees[i]]
        return tree_val
    
    # Creates all possible combinations of the values of the trees that can multiply planted trees.

    def extra_trees(self, W):
        trees_vals = self.R.values.copy()
        if 'xi' in self.R.degrees:
            trees_vals['xi'] = W
        dic_values = {}
        for i in self.R.rule_extra:
            dic_values[i] = self.trees_multiply(trees_vals, self.R.rule_extra[i])

        return dic_values
    
    # Given a Rule, creates a Model where all trees conform to the rule and are of degree <= 'deg'.
    def create_model_graph(self, W, lollipop = None, extra_planted = None, extra_deg = None):

        # first let the model consist of the I[xi] only.
        if lollipop is None: # if lollipop is not given Integrate noise W
            model = self.I({'xi': W}, derivative = self.derivative)
        else: # otherwise simply add lollipop
            model = {'I[xi]' : lollipop}
        # 'planted' is a set that keeps track of the planted trees I[\tau].
        # 'done' is the dictionary that keeps track of all trees that were created together with their degree.

        graph = dict()
        
        planted, done = {'I[xi]'}, self.R.degrees.copy() #create set of planted trees and a dictinary of the trees degrees
        
        graph['xi'] = {}
        # Add planted trees that correspond to functions u^i for i in \mathcal{J}. 
        if extra_planted is not None: 
            model.update(extra_planted)
            planted = planted.union(set(extra_planted.keys()))
            # for key in extra_deg.keys():
            #     if 'u_0' in key:
            #         graph.update({'u_0' : {}})
            # graph.update({key: {'u_0' : 1} if 'u_0' in key else {} for key in extra_deg.keys()})
            graph.update({key: {} for key in extra_deg.keys()})
            # graph.update(extra_deg)
            # print(planted)
            done.update(extra_deg)
            # print(done)
        # If necessary add spatial derivative of the I[xi] denoted by I'[xi]
        if self.derivative:
            planted.add("I'[xi]")
            done["I'[xi]"] = done["I[xi]"] - 1
            
        extra_trees_values = self.extra_trees(W)

        graph['I[xi]'] = {'xi':1}
        # Add trees of greater height
        for j in range(1, self.H):

            # Compute multiplications of the planted trees. self.R.max is the maximum possible width
            for k in range(1, self.R.max + 1):  # k is the number of trees multiplied
                # check all possible combinations of product of k planted trees
                for words in comb([w for w in planted], k):
                    tree, dic = self.R.words_to_tree(words)  # create one tree product out of the list of trees
                    temp_deg = self.tree_deg(dic, done) # compute the degree of this tree
                    # check if the tree needs to be added. k <= self.R.free_num checks if the product of k trees can exist
                    if tree not in done and tree not in self.R.exceptions and k <= self.R.free_num and temp_deg + self.R.degrees['I'] <= self.deg:
                        model[tree] = self.trees_multiply(model, dic)  # add this tree to the model
                        graph[tree] = dic
                        # if necessary add the tree multiplied by extra trees.
                    done[tree] = temp_deg  # include the tree to the done dictionary together with its degree
                    # multiply by the extra trees if such are present in the rule
                    for i in extra_trees_values: # add extra trees that correspond to multiplicative width
                        if k <= self.R.rule_power[i]: # check if extra tree can multiply the k product of planted trees
                            extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(i))
                            new_tree = extra_tree +'(' + tree +')' #shape of the new tree
                            deg = done[tree] + self.tree_deg(extra_dic, done) # degree of a new tree
                            if new_tree not in done and new_tree not in self.R.exceptions and deg <= self.deg:
                                if tree in model:
                                    model[new_tree] = model[tree]*extra_trees_values[i]
                                    graph[new_tree] = dict(chain.from_iterable(d.items() for d in (extra_dic, {tree:1})))
                                else:
                                    model[new_tree] = self.trees_multiply(model, dic)*extra_trees_values[i]
                                    graph[new_tree] = dict(chain.from_iterable(d.items() for d in (extra_dic, dic)))
                                done[new_tree] = done[tree] + self.tree_deg(extra_dic, done)                
            # integrate trees from the previous iteration.
            this_round = self.I(model, planted, self.R.exceptions, self.derivative)  
            keys = [tree for tree in this_round.keys() if tree not in self.R.degrees and tree not in planted]
            # include theese integrated trees to the model. Don't include trees that are not of the form I[\tau]
            for IZ in keys:  
                if IZ[1] == "[":
                    Z = IZ[2:-1]  # IZ = I[Z]
                else:  
                    Z = IZ[3:-1]  # IZ = I'[Z]
                model[IZ] = this_round.pop(IZ)
                graph.update({IZ:graph[Z] if Z[0] != 'I' else {Z:1}})
                planted.add(IZ)  # add tree IZ to planted
                if Z not in planted and Z in model:
                    model.pop(Z)  # Delete Z tree from the model if it is not planted
                    graph.pop(Z)

                if IZ[1] == "[":
                    done[IZ] = done[Z] + self.R.degrees['I']  # add degree to IZ  
                else:  
                    done[IZ] = done[Z] + self.R.degrees['I'] - 1
        # delete all the trees of the form \partial_x I[\tau] and only keep
        # trees of the form I[\tau]
        if self.derivative:
            model = {IZ: model[IZ] for IZ in model if IZ[1] != "'"}

        return graph