import pandas as pd

df = pd.read_excel('./data/fcs_analyzed_result.xlsx')

df["Population Definition Reported"].value_counts().to_csv("population_definition_reported_counts_total.csv", index=True)
