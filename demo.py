import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_excel("online_retail_II.xlsx")
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['Invoice'],inplace=True)

df['Invoice'] = df['Invoice'].astype(str)
df = df[~df['Invoice'].str.contains('C')]

basket = (df[df['Country'] == 'France'].groupby(['Invoice','Description'])['Quantity']
          .sum().unstack().reset_index().ffill(axis=0).set_index('Invoice'))



basket_sets = (basket > 0).astype(bool)
basket_sets.drop("POSTAGE",inplace=True,axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True, max_len=3)
rules = association_rules(frequent_itemsets,metric="lift",min_threshold=1.5)


rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

# Display top 10 rules
print("\nTop 10 Rules:")
pd.set_option('display.max_columns', None)
print(rules.head(10))