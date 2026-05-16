import pandas as pd
import re

def normalize_marker(marker):
    """
    Normaliza um marcador individual com base nas regras da Cell Ontology.
    """

    m = marker.strip()
    print(f"Marker after stripping: '{m}'")
    m_lower = m.lower()
    
    # 1. Regras para populações e sinónimos celulares
    synonyms = {
        'singlet': ['sing', 'singlets', 'singlet', 'doublet_excluded', 'sing-f', 'intact_singlet', 'single_cells', 'singlet_gate'],
        'lymphocyte': ['ly', 'lymp', 'lymph', 'lymphocyte', 'lymphs', 'lymphocytes', 'lymo', 'lymphos', 'lym'],
        'monocyte': ['mo', 'mono', 'monos', 'mnc', 'monocytes', 'pmo'],
        'gran': ['gran', 'granulocytes', 'granulocyte', 'granulo', 'pmns'], 
        'intact': ['intact_cells', 'intact_cells_population', 'intact'],
        'viable': ['live', 'annexin-', 'live/dead stain', 'live/dead', '7aad-', 'pi-', 'dapi-', 'viability', 'viable'],
        'proliferated': ['cfse-', 'tracerviolet', 'cfse_low', 'cfse_dim', 'ctv_low', 'celltrace_low', 'proliferating'],
        'neutro': ['neutro', 'neutrophil', 'neutrophils', 'neutros', 'pmn'],
        'mDC': ['mdc'],
        'pDC': ['pdc'],
        'mo2': ['mo2'],
        'mo3': ['mo3'],
        'TH': ['th'],
        'TFH': ['tfh'],
        'PB': ['pb'],
        'NK': ['nk'],
        'WBC': ['wbc'],
        'Q1': ['q1', 'q1 cd14'],
        'Q2': ['q2', 'q2 cd16', 'q2 cd19'], # NOVO
        'Q3': ['q3', 'q3 cd19'],
        'TH': ['th'],
        'TH1': ['th1'],
        'TH2': ['th2'],
        'TH17': ['th17'],
        'TH1-17': ['th1-17', 'th117'],
        'TFH': ['tfh'],
        'TFH1': ['tfh1'],
        'TFH2': ['tfh2'],
        'TFH17': ['tfh17'],
        }
    
    for standard, variations in synonyms.items():
        if m_lower in variations:
            return standard


    print(f"Marker after synonym check: '{m}'")
            
    # 2. Regras estritas para estados dos marcadores (nível de deteção)
    # A notação (?i) diz ao Regex para ignorar maiúsculas/minúsculas.
    # O símbolo $ garante que só substituímos se estiver no FINAL da palavra (ex: não vai alterar a palavra 'medio' no meio de um nome).
    state_mappings = [
        (r'(neg)', '-'),          
        (r'(pos)', '+'),                
        (r'(dim|lo|di)', '+'),                 
        (r'(int|medium|med)', '+'),  
        (r'(bright|hi|bri|high|br)', '+')          
    ]

    for pattern, replacement in state_mappings:
        if re.search(pattern, m):
            print(re.sub(pattern, replacement, m, flags=re.IGNORECASE))
            # Faz a substituição mantendo o resto do nome do marcador intacto
            m = re.sub(pattern, replacement, m, flags=re.IGNORECASE)
    
    if "+" not in m and "-" not in m and "+-" not in m and "+~" not in m and "++" not in m:
        m = m+"+"  # Se não tiver nenhum estado, assumimos que é positivo (ex: CD3 -> CD3+)

    m = re.sub(r'^(dr|hladr)([+\-~]*)$', r'HLA-DR\2', m, flags=re.IGNORECASE)

    return m

def parse_definition(definition):
    """
    Limpa e normaliza uma definição completa de população.
    """
    # Se a linha for nula, ignora
    if pd.isna(definition):
        return ""

    def_clean = re.sub(r',Freq\.? of.*', '', str(definition), flags=re.IGNORECASE)
    def_clean = re.sub(r'_?pP', '', str(def_clean), flags=re.IGNORECASE)  # Remove 'pP' que é um artefato comum
    

    def_clean = re.sub(r'\b(AND|OR|Sum|of|cells?|small|B)\b', ' ', def_clean, flags=re.IGNORECASE)
    
    # Remove a palavra 'small' (já que 'lymphocyte' será capturado normalmente depois)
    def_clean = re.sub(r'\bsmall\b', ' ', def_clean, flags=re.IGNORECASE)

    def_clean = re.sub(r'HLA-DR', 'HLADR', def_clean, flags=re.IGNORECASE)
        
    # Remove as anotações estatísticas do final da string (ex: ,Freq. of Parent)
    def_clean = re.sub(r'(TH|TFH)/([0-9\-]+)', r'\1\2', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'[^a-zA-Z0-9\s+-]', ' ', def_clean)  # Remove caracteres especiais, mantendo apenas letras, números e espaçosß
    def_clean = re.sub(r'(pos|neg|dim|lo|di)(?=[A-Z])', r'\1 ', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'([+-])(?=[a-zA-Z])', r'\1 ', def_clean)
    # Divide a string onde houver uma barra (/) ou vírgula (,)
    markers = re.split(r'\s+', def_clean.strip())
    
    # Normaliza cada marcador encontrado
    normalized_markers = [normalize_marker(m) for m in markers if m.strip()]
    
    # Remove duplicatas (ex: se "Lymph" e "ly" caírem na mesma linha e virarem "lymphocyte")
    normalized_markers = list(set(normalized_markers))
    
    # Ordena alfabeticamente para garantir correspondência (ex: CD3+, CD4+ == CD4+, CD3+)
    normalized_markers.sort()
    
    # Junta tudo novamente
    return ", ".join(normalized_markers)


df = pd.read_csv("cell_definitions_list.csv")

# 2. Aplica a função de normalização criando uma nova coluna
nome_da_coluna = 'Population Definition Reported' # Confirme se o nome da coluna é este
df['Normalized_Definition'] = df[nome_da_coluna].apply(parse_definition)
df.to_csv("cell_definitions_with_normalized.csv", index=False)
# # 3. Encontra e agrupa todas as linhas que se revelaram idênticas após a limpeza
duplicates = df[df.duplicated('Normalized_Definition', keep=False)].sort_values('Normalized_Definition')

print(f"Total de linhas processadas: {len(df)}")
print(f"Total de definições únicas após normalização: {df['Normalized_Definition'].nunique()}")

# # 4. Guarda as correspondências encontradas num novo ficheiro CSV
# duplicates.to_csv('normalized_cell_definitions.csv', index=False)
print("Ficheiro 'normalized_cell_definitions.csv' guardado com sucesso!")