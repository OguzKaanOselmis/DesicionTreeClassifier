from typing import List

class Node:
    def __init__(self, feature_index=None, threshold=None, left_node=None, right_node=None, impurity=None, classification=None):
        
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.impurity = impurity

        self.classification = classification    #for leaf node

class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.root = None
        self.feature_count = 0
    
    def fit(self, X: List[List[float]], y: List[int]):
        dataset = [[i for i in row] for row in X]                           # copy of X
        [dataset[index].append(y[index]) for index in range(len(dataset))]  # y concatinated inside of X
        self.feature_count = len(X[0])

        self.root = self.generate_desicion_tree(dataset,0)


    def generate_desicion_tree(self,dataset,current_depth):     #creates tree recursively
        feature_values,target_values = self.extract_feature_and_target_values(dataset)
        if(current_depth < self.max_depth):    
            if(self.are_all_values_same(target_values)):
                return Node(classification=target_values[0])
            else:
                split_node = self.determine_split_node(dataset)
                if split_node.impurity < 1:
                    left_tree = self.generate_desicion_tree(split_node.left_node,current_depth+1)
                    right_tree = self.generate_desicion_tree(split_node.right_node,current_depth+1)

                    return Node(split_node.feature_index,split_node.threshold,left_tree,right_tree,split_node.impurity)
        else:
            leaf_classification = self.determine_majority(target_values)
            return Node(classification=leaf_classification)

    
    def extract_feature_and_target_values(self,dataset):    # takes last column of 2d array as target_values, takes all columns except the last one as feature_values
        feature_values =  [row[:-1] for row in dataset]
        target_values = [row[-1] for row in dataset]
        return feature_values,target_values


    def determine_split_node(self,dataset):     # loop over all features and try all thresholds belongs to that feature to determine optimal split
        split_node = Node()
        min_impurity = float("inf")

        feature_values,target_values = self.extract_feature_and_target_values(dataset)
        for feat_index in range(self.feature_count):
            column_values = [row[feat_index] for row in feature_values]
            possible_thresholds = self.calculate_possible_thresholds(column_values)

            for threshold in possible_thresholds:
                left_dataset = [row for row in dataset if row[feat_index] <= threshold]
                right_dataset = [row for row in dataset if row[feat_index] > threshold]

                if(len(left_dataset) != 0 and len(right_dataset) != 0):
                    split_impurity = self.calculate_impurity(left_dataset,right_dataset)
                    if(split_impurity < min_impurity):
                        split_node.feature_index = feat_index
                        split_node.threshold = threshold
                        split_node.left_node = left_dataset
                        split_node.right_node = right_dataset
                        split_node.impurity = split_impurity
                        min_impurity = split_impurity
        return split_node
                    

    def calculate_possible_thresholds(self,value_list):
        sorted_value_list = sorted(list(set(value_list)))
        return [(sorted_value_list[i]+sorted_value_list[i+1])/2 for i in range(len(sorted_value_list)-1)]


    def are_all_values_same(self,list):
        if(len(list) == 0):
            return True
        else:
            initial = list[0]
            for i in list:
                if initial != i:
                    return False
            
            return True


    def calculate_impurity(self,left_dataset,right_dataset):
        left_weight = len(left_dataset) / (len(left_dataset) + len(right_dataset))
        right_weight = len(right_dataset) / (len(left_dataset) + len(right_dataset))

        dummy,left_target_list = self.extract_feature_and_target_values(left_dataset)
        dummy,right_target_list = self.extract_feature_and_target_values(right_dataset)

        return left_weight*self.calculate_gini(left_target_list) + right_weight*self.calculate_gini(right_target_list)
    

    def calculate_gini(self,target_values):
        uniq_targets = list(set(target_values))
        temp = 0

        for curr_target in uniq_targets:
            counter = 0

            for target in target_values:
                if target == curr_target: 
                    counter+=1

            probability = counter/len(target_values)
            temp += probability**2

        return 1-temp


    def determine_majority(self,value_list):
        value_counts = {}

        for value in value_list:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1
        
        return max(value_counts, key=value_counts.get)
        


    def predict(self, X: List[List[float]]):
        preditions = [self.predict_recursive(x, self.root) for x in X]
        return preditions
    
    def predict_recursive(self,feature_values,node : Node):
         if node.classification != None: 
             return node.classification
         else:
            if feature_values[node.feature_index] <= node.threshold:
                return self.predict_recursive(feature_values, node.left_node)
            else:
                return self.predict_recursive(feature_values, node.right_node)


# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    