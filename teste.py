import pandas as pd

df = pd.read_csv('population_definition_counts.csv')
definitions = df['Population Definition Reported'].dropna().astype(str).unique()

print("=== COMO O CD103 FOI USADO? ===")
for d in definitions:
    if 'cd103' in d.lower():
        print(f"-> {d}")

print("\n=== COMO O CD11C FOI USADO? (Exibindo 5 exemplos) ===")
cd11c_defs = [d for d in definitions if 'cd11c' in d.lower()]
for d in cd11c_defs[:5]:
    print(f"-> {d}")