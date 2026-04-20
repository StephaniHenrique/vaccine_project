import pandas as pd
import re

def processar_deltas_citometria(parquet_path):
    print("1. Carregando dados...")
    df = pd.read_parquet(parquet_path)
    
    # =========================================================================
    # CONFIGURAÇÕES DO SEU DATASET 
    # =========================================================================
    COL_DEF = 'Population Definition Reported'
    COL_PATIENT = 'Participant ID'         
    COL_TIME = 'Study Time Collected'          
    COL_VALUE = 'Population Cell Number'              
    
    # NOVAS CONFIGURAÇÕES DE TEMPO (Janelas)
    # Como os seus dados apareceram como float (0., 14., etc), garantimos ambos os formatos
    VAL_BASELINE = [0, 0.0] 
    VAL_PEAK = [13, 13.0, 14, 14.0, 15, 15.0]
    
    definitions = df[COL_DEF].dropna().astype(str).unique()

    # =========================================================================
    # 2. MOTOR DE REGEX
    # =========================================================================
    def pos(m): return rf"{m}(?![0-9])(?:hi|high|bright|\+|\+\+|(?=[^a-z-]|$))"
    def neg(m): return rf"{m}(?![0-9])(?:-|lo|low|dim|neg)"

    regras = {
        "live.cd4": [pos("cd4")],
        "cd4.treg": [pos("cd4"), pos("cd25"), neg("cd127")], 
        "tconv": [pos("cd4")], 
        "cd4tconv.cd103": [pos("cd4"), pos("cd103")],
        "cd4tconv.th2": [pos("cd4"), r"(th/?2|crth2)"],
        "live.cd8": [pos("cd8")],
        "CD8.CD103": [pos("cd8"), pos("cd103")],
        "CD8.GrzmB": [pos("cd8"), pos("(gzb|grzmb|granzyme\s*b)")],
        "ILC1": [neg("cd3"), neg("cd14"), neg("cd19"), pos("cd127"), pos("cd161")],
        "NK": [pos("cd56"), neg("cd3")],
        "NK.GrzmB": [pos("cd56"), neg("cd3"), pos("(gzb|grzmb|granzyme\s*b)")],
        "Neutro": [pos("cd66b")],
        "Macro": [pos("cd14")], 
        "Mo0": [pos("cd14"), neg("cd16")], 
        "mo0.CD11c": [pos("cd14"), neg("cd16"), pos("cd11c")],
        "DC": [pos("hladr"), neg("cd14"), neg("cd19"), neg("cd3")],
        "B.memory": [pos("cd19"), pos("cd27")],
        "B.naive.FO": [pos("cd19"), neg("cd27"), pos("igd")],
        "B.plasma": [pos("cd19"), pos("cd38"), pos("cd138")],
        "B.GC": [pos("cd19"), pos("cd38"), pos("cd27")],
        "B.immature": [pos("cd19"), pos("cd24"), pos("cd38")]
    }

    print("2. Mapeando definições para Features...")
    mapa_definicoes = {}
    
    for def_str in definitions:
        def_lower = def_str.lower()
        feature_encontrada = None
        
        for feature_name, padroes in regras.items():
            atende_condicoes = all(re.search(padrao, def_lower) for padrao in padroes)
            
            # Regra de Exclusão (Tconv não é Treg)
            if atende_condicoes and (feature_name.startswith("cd4tconv") or feature_name == "tconv"):
                if re.search(pos("cd25"), def_lower) and re.search(neg("cd127"), def_lower):
                    atende_condicoes = False
                    
            if atende_condicoes:
                feature_encontrada = feature_name
                break 
                
        if feature_encontrada:
            mapa_definicoes[def_str] = feature_encontrada

    df['Feature'] = df[COL_DEF].map(mapa_definicoes)
    df_filtrado = df.dropna(subset=['Feature'])
    
    # =========================================================================
    # 3. PIVOT: TRANSFORMAR VERTICAL EM HORIZONTAL
    # =========================================================================
    print("3. Pivotando os dados (Long -> Wide)...")
    
    # Mantemos a coluna de tempo no Pivot para podermos filtrar na próxima etapa
    df_wide = pd.pivot_table(
        df_filtrado, 
        index=[COL_PATIENT, COL_TIME], 
        columns='Feature', 
        values=COL_VALUE, 
        aggfunc='mean'
    ).reset_index()

    # =========================================================================
    # 4. CÁLCULO DO DELTA (PEAK - BASELINE) COM DIAGNÓSTICO
    # =========================================================================
    print("\n4. Calculando Deltas pareados por paciente...")
    
    # Filtra usando .isin() para capturar qualquer dia dentro da nossa lista
    df_base_raw = df_wide[df_wide[COL_TIME].isin(VAL_BASELINE)]
    df_peak_raw = df_wide[df_wide[COL_TIME].isin(VAL_PEAK)]
    
    # Agrupamos por paciente tirando a média. 
    # Isso resolve se o paciente tiver coleta no dia 13 e 14 simultaneamente, e já joga o ID para o index!
    df_base = df_base_raw.groupby(COL_PATIENT).mean()
    df_peak = df_peak_raw.groupby(COL_PATIENT).mean()
    
    # Tira a coluna COL_TIME do caminho da matemática (ela virou lixo depois do groupby)
    df_base = df_base.drop(columns=[COL_TIME], errors='ignore')
    df_peak = df_peak.drop(columns=[COL_TIME], errors='ignore')
    
    # Diagnóstico para o usuário
    pacientes_base = set(df_base.index)
    pacientes_peak = set(df_peak.index)
    pacientes_com_ambos = pacientes_base.intersection(pacientes_peak)
    
    print("-" * 50)
    print(f"🔎 DIAGNÓSTICO DE PACIENTES PAREADOS:")
    print(f" -> Pacientes com amostra no Baseline (Dia 0): {len(pacientes_base)}")
    print(f" -> Pacientes com amostra no Peak (Dias 13, 14, 15): {len(pacientes_peak)}")
    print(f" -> ✅ PACIENTES COM OS DOIS TEMPOS: {len(pacientes_com_ambos)}")
    print("-" * 50)
    
    if len(pacientes_com_ambos) == 0:
        print("❌ Nenhum paciente tem os dois tempos simultaneamente. Não é possível calcular o Delta.")
        return None

    # Calcula a diferença
    df_delta = df_peak - df_base
    
    # Mantém apenas os pacientes que tinham as duas coletas (os que caíram no intersection)
    df_delta = df_delta.loc[list(pacientes_com_ambos)]
    
    print("\nConcluído!")
    print("\n=== VISUALIZAÇÃO DO DELTA (Peak - Baseline) ===")
    print(df_delta.head())
    
    output_name = 'deltas_pacientes_citometria.csv'
    df_delta.to_csv(output_name)
    print(f"\n✅ Arquivo salvo com sucesso como: {output_name}")
    
    return df_delta

if __name__ == "__main__":
    processar_deltas_citometria('fcs_analyzed_result.parquet')