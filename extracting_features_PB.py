import pandas as pd
import re

def rankear_estudos_features_maximizadas(parquet_path):
    df = pd.read_parquet(parquet_path)
    
    COL_DEF = 'Population Definition Reported'
    COL_PATIENT = 'Participant ID'         
    
    definitions = df[COL_DEF].dropna().astype(str).unique()

    #Trying to define the rules for feature extraction based on the definitions
    def pos(m): return rf"{m}(?![0-9a-zA-Z])\s*(?:hi|high|bright|dim|\+|\+\+|(?!-|lo|low|neg))"
    def neg(m): return rf"{m}(?![0-9a-zA-Z])\s*(?:-|lo|low|neg)"

    dict_rules = [
        {"feature": "cd4.treg", "include": [pos("cd4"), pos("cd25"), neg("cd127")], "remove": [pos("cd8")]},
        {"feature": "CD8.GrzmB", "include": [pos("cd8"), pos(r"(gzb|grzmb|granzyme\s*b)")], "remove": [pos("cd4")]},
        {"feature": "B.plasma", "include": [pos("cd138")], "remove": [pos("cd3"), pos("cd4"), pos("cd8")]},
        {"feature": "B.plasma", "include": [r"cd19", pos("cd38"), pos("cd27")], "remove": [pos("cd3")]},
        {"feature": "B.memory", "include": [pos("cd19"), pos("cd27")], "remove": [pos("cd3"), pos("cd38")]},
        {"feature": "Mo0", "include": [pos("cd14"), neg("cd16")], "remove": [pos("cd3"), pos("cd19"), pos("cd56")]},
        {"feature": "DC", "include": [pos(r"(hladr|mhc-?ii)")], "remove": [pos("cd3"), pos("cd14"), pos("cd19"), pos("cd56")]},
        {"feature": "Neutro", "include": [pos(r"(cd66b|cd15)")], "remove": []},
        {"feature": "NK", "include": [pos(r"(cd56|ncam)")], "remove": [pos("cd3"), pos("cd14"), pos("cd19"), pos("cd4")]},
        {"feature": "Macro", "include": [pos("cd14")], "remove": [pos("cd3"), pos("cd19")]},
        {"feature": "live.cd4tconv", "include": [pos("cd4")], "remove": [pos("cd8"), pos("cd19"), pos("cd14"), pos("cd56")]},
        {"feature": "live.cd8", "include": [pos("cd8")], "remove": [pos("cd4"), pos("cd19"), pos("cd14"), pos("cd56")]}
    ]

    def_map = {}
    
    for def_str in definitions:
        def_lower = str(def_str).lower()
        feture_found = None
        
        for rule in dict_rules:
            attending_inclusion = all(re.search(pattern, def_lower) for pattern in rule["include"])
            attending_exclusion = not any(re.search(pattern, def_lower) for pattern in rule["remove"])
            
            if attending_inclusion and attending_exclusion:
                feture_found = rule["feature"]
                break
                
        if feture_found:
            def_map[def_str] = feture_found

    df['Feature'] = df[COL_DEF].map(def_map)
    df_filtered = df.dropna(subset=['Feature']).copy()
    
    #Extracting studies with features
    df_filtered['Study'] = df_filtered[COL_PATIENT].apply(
        lambda x: str(x).split('.')[-1] if '.' in str(x) else 'Desconhecido'
    )
    
    # Agrupa por Estudo e cria uma lista com as features únicas que cada um tem
    ranking_df = df_filtered.groupby('Study')['Feature'].unique().reset_index()
    
    # Conta quantas features cada estudo tem (máximo possível = 11)
    ranking_df['Total_Features'] = ranking_df['Feature'].apply(len)
    
    # Ordena do maior para o menor
    ranking_df = ranking_df.sort_values(by='Total_Features', ascending=False).reset_index(drop=True)

    
    for idx, row in ranking_df.iterrows():
        estudo = row['Study']
        total = row['Total_Features']
        lista_features = ", ".join(sorted(row['Feature']))
        
        print(f"{idx + 1}º LUGAR: Estudo {estudo} | Cobre {total}/11 features")
        print(f"   ↳ {lista_features}\n")
    
    # Exportando os resultados para CSV
    output_name = 'ranking_estudos_features.csv'
    ranking_df['Feature'] = ranking_df['Feature'].apply(lambda x: ", ".join(sorted(x)))
    ranking_df.to_csv(output_name, index=False)
    
    print(f"✅ Tabela do ranking salva em: {output_name}")
    return ranking_df

if __name__ == "__main__":
    # Certifique-se de apontar para o seu parquet original que contém a coluna Participant ID
    df_ranking = rankear_estudos_features_maximizadas('./Dataset/Public_dt/fcs_analyzed_result.parquet')