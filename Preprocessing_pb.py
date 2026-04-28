import pandas as pd
import re

def processar_citometria_pivot(citometria_path):
    
    df_cito = pd.read_excel(citometria_path) 
    
    COL_DEF = 'Population Definition Reported'
    COL_PATIENT = 'Participant ID'
    COL_TIME = 'Study Time Collected'
    COL_VALUE = 'Population Cell Number'
    COL_UNIT = 'Cell Number Unit'
    
    # Filtro de Baseline (0) e Peak (7)
    df_cito = df_cito[df_cito[COL_TIME].isin([0, 7])].copy()
    print(f"Linhas após filtrar tempo (0 e 7): {len(df_cito)}")

    def pos(m): return rf"{m}(?![0-9a-zA-Z])\s*(?:hi|high|bright|dim|\+|\+\+|(?!-|lo|low|neg))"
    def neg(m): return rf"{m}(?![0-9a-zA-Z])\s*(?:-|lo|low|neg)"

    regras_ordenadas = [
        {"feature": "cd4.treg", "incluir": [pos("cd4"), pos("cd25"), neg("cd127")], "excluir": [pos("cd8")]},
        {"feature": "CD8.GrzmB", "incluir": [pos("cd8"), pos(r"(gzb|grzmb|granzyme\s*b)")], "excluir": [pos("cd4")]},
        {"feature": "B.plasma", "incluir": [pos("cd138")], "excluir": [pos("cd3"), pos("cd4"), pos("cd8")]},
        {"feature": "B.plasma", "incluir": [r"cd19", pos("cd38"), pos("cd27")], "excluir": [pos("cd3")]},
        {"feature": "B.memory", "incluir": [pos("cd19"), pos("cd27")], "excluir": [pos("cd3"), pos("cd38")]},
        {"feature": "Mo0", "incluir": [pos("cd14"), neg("cd16")], "excluir": [pos("cd3"), pos("cd19"), pos("cd56")]},
        {"feature": "DC", "incluir": [pos(r"(hladr|mhc-?ii)")], "excluir": [pos("cd3"), pos("cd14"), pos("cd19"), pos("cd56")]},
        {"feature": "Neutro", "incluir": [pos(r"(cd66b|cd15)")], "excluir": []},
        {"feature": "NK", "incluir": [pos(r"(cd56|ncam)")], "excluir": [pos("cd3"), pos("cd14"), pos("cd19"), pos("cd4")]},
        {"feature": "Macro", "incluir": [pos("cd14")], "excluir": [pos("cd3"), pos("cd19")]},
        {"feature": "live.cd4tconv", "incluir": [pos("cd4")], "excluir": [pos("cd8"), pos("cd19"), pos("cd14"), pos("cd56")]},
        {"feature": "live.cd8", "incluir": [pos("cd8")], "excluir": [pos("cd4"), pos("cd19"), pos("cd14"), pos("cd56")]}
    ]

    definitions = df_cito[COL_DEF].dropna().astype(str).unique()
    mapa_definicoes = {}
    
    for def_str in definitions:
        def_lower = str(def_str).lower()
        for regra in regras_ordenadas:
            if all(re.search(p, def_lower) for p in regra["incluir"]) and not any(re.search(p, def_lower) for p in regra["excluir"]):
                mapa_definicoes[def_str] = regra["feature"]
                break

    df_cito['Feature'] = df_cito[COL_DEF].map(mapa_definicoes)
    df_cito = df_cito.dropna(subset=['Feature'])

    # =========================================================================
    # 3. TRANSFORMAÇÃO PIVOT (LONG TO WIDE)
    # =========================================================================
    print("3. Pivotando a tabela (Transformando Features em colunas)...")
    
    # Criamos o Pivot. 
    # Index: O que define uma linha única (Paciente + Dia + Unidade)
    # Columns: O que vai virar cabeçalho (Nossas Features mapeadas)
    # Values: O valor numérico
    
    df_pivot = df_cito.pivot_table(
        index=[COL_PATIENT, COL_TIME, COL_UNIT], 
        columns='Feature', 
        values=COL_VALUE,
        aggfunc='first' # Caso haja duplicatas exatas, pega a primeira
    ).reset_index()

    print("-" * 50)
    print(f"✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"Número de pacientes únicos: {df_pivot[COL_PATIENT].nunique()}")
    print(f"Formato final: {df_pivot.shape[0]} linhas e {df_pivot.shape[1]} colunas.")
    print("-" * 50)
    
    output_name = 'citometria_pivotada.csv'
    df_pivot.to_csv(output_name, index=False)
    print(f"Arquivo salvo como: {output_name}")
    
    return df_pivot

if __name__ == "__main__":
    # Ajuste o caminho para o seu arquivo
    df_final = processar_citometria_pivot('./Dataset/Public_dt/fcs_404.xlsx')
    print(df_final.head())