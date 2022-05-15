import pandas as pd # to deal with dataframe
from IPython.display import display
from pyECLAT import ECLAT
from pyECLAT import Example2

dataset = Example2().get()

# create an instance of eclat
my_eclat = ECLAT(data=dataset, verbose=True)

# fit the algorithm
rule_indices, rule_supports = my_eclat.fit(min_support=0.03,
                                           min_combination=2,
                                           max_combination=2)

result = pd.DataFrame(rule_supports.items(),columns=['Item', 'Support'])
display(result.nlargest(n = 10, columns = 'Support'))
