import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import results
from IPython.display import display

Data = pd.read_csv('C:/Vyara/Uni/Market_Basket_Optimisation.csv', header=None)

# Intializing the list
transacts = []
# populating a list of transactions
for i in range(0, 7501):
    transacts.append([str(Data.values[i, j]) for j in range(0, 20)])

from apyori import apriori
rule = apriori(transactions=transacts, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,
                max_length=2) # returns RelationRecord generator
 
output = list(rule)  # returns a non-tabular output, To get list of rules from RelationRecord, convert RelationRecord to list

# putting output into a pandas dataframe
def inspect(output):
    lhs = [tuple(result[2][0][0])[0] for result in output]
    rhs = [tuple(result[2][0][1])[0] for result in output]
    support = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))


output_DataFrame = pd.DataFrame(inspect(output),
                                columns=['Left_Hand_Side', 'Right_Hand_Side', 'Support', 'Confidence', 'Lift'])

#display(output_DataFrame)
display(output_DataFrame.nlargest(n = 10, columns = 'Lift'))
