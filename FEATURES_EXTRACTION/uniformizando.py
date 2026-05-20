import pandas as pd
import re

def normalize_marker(marker):
    """
    Normaliza um marcador individual com base nas regras da Cell Ontology.
    """
    print(f"Original marker: '{marker}'")
    m = marker.strip()
    # print(f"Marker after stripping: '{m}'")
    m_lower = m.lower()
    print(f"Marker after lowercasing: '{m_lower}'")
    # 1. Regras para populações e sinónimos celulares
    synonyms = {
        'singlet': ['sing', 'singlets', 'singlet', 'doublet_excluded', 'sing-f', 'intact_singlet', 'single_cells', 'singlet_gate'],
        'lymphocyte': ['ly', 'lymp', 'lymph', 'lymphocyte', 'lymphs', 'lymphocytes', 'lymo', 'lymphos', 'lym'],
        'monocyte': ['mo', 'mono', 'monos', 'mnc', 'monocytes', 'pmo', 'monocyte'],
        'gran': ['gran', 'grans', 'granulocytes', 'granulocyte', 'granulo', 'pmns'], 
        'intact': ['intact_cells', 'intact_cells_population', 'intact'],
        'viable': ['live', 'annexin-', 'live/dead stain', 'live/dead', '7aad-', 'pi-', 'dapi-', 'viability', 'viable'],
        'proliferated': ['cfse-', 'tracerviolet', 'cfse_low', 'cfse_dim', 'ctv_low', 'celltrace_low', 'proliferating'],
        'neutro': ['neutro', 'neutrophil', 'neutrophils', 'neutros', 'pmn', 'neutr', 'neu'],
        'naive': ['naive', 'naïve', 'nv'],
        'EM': ['em', 'effector_memory', 'effector-memory'],
        'CM': ['cm', 'central_memory', 'central-memory'],
        'TEMRA': ['temra', 'emra'],
        'NK': ['nk', 'nk_cells', 'nkcells'],
        'NKT': ['nkt', 'nkt_cells'],
        'plasmablast': ['pb', 'plasmablast', 'plasmablasts', 'nplasmablast'],
        'plasma_cell': ['pc', 'plasma_cells', 'plasmacell'],
        'mDC': ['mdc'],
        'pDC': ['pdc'],
        'mo2': ['mo2'],
        'mo3': ['mo3'],
        'TH': ['th'],
        'TFH': ['tfh'],
        'PB': ['pb'],
        'NK': ['nk'],
        'WBC': ['wbc', 'swbc'],
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
    
    m_base = re.sub(r'(_bright|_dim|_int|[+\-~]+)$', '', m_lower)
    for standard, variations in synonyms.items():
        if m_base in variations or m_lower in variations:
            # Substitui a base pelo padrão, mas preserva os sinais finais
            state = m_lower.replace(m_base, '') if m_base in variations else ''
            if state == '':
                return standard
            else:
                return standard + state


    print(f"Marker after synonym check: '{m}'")
            
    # 2. Regras estritas para estados dos marcadores (nível de deteção)
    # A notação (?i) diz ao Regex para ignorar maiúsculas/minúsculas.
    # O símbolo $ garante que só substituímos se estiver no FINAL da palavra (ex: não vai alterar a palavra 'medio' no meio de um nome).
    state_mappings = [
        (r'(neg)$', '-'),          
        (r'(pos)$', '+'),                
        (r'(dim|lo|di)$', '+'),                 
        (r'(int|medium|med)$', '+'),  
        (r'(bright|hi|bri|high|br)$', '+')          
    ]

    for pattern, replacement in state_mappings:
        if re.search(pattern, m):
            # print(re.sub(pattern, replacement, m, flags=re.IGNORECASE))
            # Faz a substituição mantendo o resto do nome do marcador intacto
            m = re.sub(pattern, replacement, m, flags=re.IGNORECASE)
    
    if "+" not in m and "-" not in m and "+-" not in m and "+~" not in m and "++" not in m:
        m = m+"+"  # Se não tiver nenhum estado, assumimos que é positivo (ex: CD3 -> CD3+)

    m = re.sub(r'^(dr|hladr)([+\-~]*)$', r'HLA-DR\2', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?1([+\-]*)$', r'CD1c\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?2([+\-]*)$', r'CD303\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?3([+\-]*)$', r'CD141\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?4([+\-]*)$', r'CD304\1', m, flags=re.IGNORECASE)
    print(f"Marker after state mapping: '{m}'")
    print("-" * 50)
    return m

def parse_definition(definition):
    """
    Limpa e normaliza uma definição completa de população.
    """
    # Se a linha for nula, ignora
    if pd.isna(definition):
        return ""

    def_clean = re.sub(r'\b\d+-(?=[a-zA-Z])', '', str(definition), flags=re.IGNORECASE)
    def_clean = re.sub(r'\bQ\d+:\s*', ' ', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'\bnQ\d+:\s*', ' ', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r',Freq\.? of.*', '', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'_?pP', '', str(def_clean), flags=re.IGNORECASE)  # Remove 'pP' que é um artefato comum
    
    # Protege "Plasma cells" antes de a palavra "cells" ser apagada
    def_clean = re.sub(r'\bplasma cells?\b', 'plasmacell', str(def_clean), flags=re.IGNORECASE)
    
    # ... (o resto das suas regras de limpeza continuam aqui por baixo)
    def_clean = re.sub(r'\b(AND|OR|Sum|of|cells?|small|count|SCC|B)\b', ' ', def_clean, flags=re.IGNORECASE)
    
    # Remove a palavra 'small' (já que 'lymphocyte' será capturado normalmente depois)
    def_clean = re.sub(r'\bsmall\b', ' ', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'\bmo-([a-zA-Z0-9])', r'\1', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'\bNeu-CD', 'Neu, CD', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'HLA-DR', 'HLADR', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(f'K1-67', 'Ki67', def_clean, flags=re.IGNORECASE)

    def_clean = re.sub(r'sing-F', 'singlet', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'2-WBC', 'WBC', def_clean, flags=re.IGNORECASE)    
    def_clean = re.sub(r'lymono', 'lymphocyte, monos', def_clean, flags=re.IGNORECASE)

    def_clean = re.sub(r'\bdn\b', 'CD14-, CD16-', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo1', 'CD14+, CD16-', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo2', 'CD14+, CD16+', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo3', 'CD14+, CD16+', str(def_clean), flags=re.IGNORECASE)

    # Remove as anotações estatísticas do final da string (ex: ,Freq. of Parent)
    def_clean = re.sub(r'(TH|TFH)/([0-9\-]+)', r'\1\2', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'[^a-zA-Z0-9\s+-]', ' ', def_clean)  # Remove caracteres especiais, mantendo apenas letras, números e espaçosß
    def_clean = re.sub(r'(pos|neg|dim|lo|di|bright|hi|br|bri)(?=[A-Z])', r'\1 ', def_clean)
    def_clean = re.sub(r'([+-])(?=[a-zA-Z])', r'\1 ', def_clean)
    # Divide a string onde houver uma barra (/) ou vírgula (,)
    markers = re.split(r'\s+', def_clean.strip())
    
    # Normaliza cada marcador encontrado
    normalized_markers = [normalize_marker(m) for m in markers if m.strip()]

    marker_set = set(normalized_markers)

    if 'CD45RA+' in marker_set and 'CCR7+' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA+', 'CCR7+']]
        normalized_markers.append('naive')
        print(f"Found naive signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")
        
    # 2. Se contiver a assinatura de EM (CD45RA- e CCR7-)
    elif 'CD45RA-' in marker_set and 'CCR7-' in marker_set:
        # Remove os marcadores individuais e adiciona o termo unificado 'EM'
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA-', 'CCR7-']]
        normalized_markers.append('EM')
        print(f"Found EM signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")    
        
    # 3. Se contiver a assinatura de CM (CD45RA- e CCR7+)
    elif 'CD45RA-' in marker_set and 'CCR7+' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA-', 'CCR7+']]
        normalized_markers.append('CM')
        print(f"Found CM signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")
        
    # 4. Se contiver a assinatura de TEMRA (CD45RA+ e CCR7-)
    elif 'CD45RA+' in marker_set and 'CCR7-' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA+', 'CCR7-']]
        normalized_markers.append('TEMRA')
        print(f"Found TEMRA signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")

    if 'CD19+' in marker_set or 'CD20+' in marker_set:
        if 'IgD+' in marker_set and 'CD27-' in marker_set:
            normalized_markers.append('B_naive')
        elif 'IgD-' in marker_set and 'CD27+' in marker_set:
            normalized_markers.append('B_switched_memory')
        elif 'IgD+' in marker_set and 'CD27+' in marker_set:
            normalized_markers.append('B_unswitched_memory')

    if 'HLADR+' in marker_set and 'CD11c+' in marker_set and 'CD14-' in marker_set:
        if 'CD1c+' in marker_set: normalized_markers.append('mDC1')
        if 'CD141+' in marker_set: normalized_markers.append('mDC2')
    if 'HLADR+' in marker_set and 'CD123+' in marker_set and 'CD11c-' in marker_set:
        normalized_markers.append('pDC')

    if 'CD303+' in marker_set or 'CD304+' in marker_set or \
       ('CD123+' in marker_set and 'CD11c-' in marker_set):
        normalized_markers.append('pDC')

    # 2. Identificar mDCs (Mieloides/Convencionais) e os seus subtipos
    # Só avaliamos mDCs se a célula não for uma pDC
    if 'pDC' not in normalized_markers:
        
        # Subtipo mDC1 (As mais comuns no sangue, marcadas por CD1c)
        if 'CD1c+' in marker_set:
            normalized_markers.append('mDC1')
            
        # Subtipo mDC2 (As especialistas em apresentação cruzada, marcadas por CD141)
        elif 'CD141+' in marker_set:
            normalized_markers.append('mDC2')
            
        # mDC Genérica (Se o investigador só usou CD11c+ para as separar das pDCs, mas não usou CD1c ou CD141)
        elif 'CD11c+' in marker_set and 'CD123-' in marker_set:
            normalized_markers.append('mDC')
    
    # Remove duplicatas (ex: se "Lymph" e "ly" caírem na mesma linha e virarem "lymphocyte")
    normalized_markers = list(set(normalized_markers))
    
    # AJUSTE DE ORDEM ALFABÉTICA: key=str.lower força o Python a não dar prioridade às letras maiúsculas!
    normalized_markers.sort(key=str.lower)
    
    # Junta tudo novamente
    return ", ".join(normalized_markers)


df = pd.read_csv("population_definition_and_unit_counts.csv")

# 2. Aplica a função de normalização criando uma nova coluna
nome_da_coluna = 'Population Definition Reported' # Confirme se o nome da coluna é este
df['Normalized_Definition'] = df[nome_da_coluna].apply(parse_definition)
df.to_csv("cell_definitions_with_normalized.csv", index=False)
# # 3. Encontra e agrupa todas as linhas que se revelaram idênticas após a limpeza
resultados_unicos = df['Normalized_Definition'].dropna().unique()

# 3. Imprime os resultados no terminal
print(f"\n=== {len(resultados_unicos)} RESULTADOS ÚNICOS APÓS LIMPEZA ===")
print("=" * 60)

df_unicos = pd.DataFrame(resultados_unicos, columns=['Normalized_Definition'])

# 4. Guarda diretamente num novo ficheiro CSV limpo
df_unicos.to_csv("resultados_unicos.csv", index=False)