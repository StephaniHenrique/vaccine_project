import pandas as pd
import re

def normalize_marker(marker):
    """
    Normaliza um marcador individual com base nas regras da Cell Ontology.
    """
    m = marker.strip()
    m_lower = m.lower()
    
    # 1. Regras para populações e sinónimos celulares
    synonyms = {
        'singlet': ['sing', 'singlets', 'singlet', 'doublet_excluded', 'sing-f', 'intact_singlet'],
        'lymphocyte': ['ly', 'lymp', 'lymph', 'lymphocyte', 'lymphs', 'lymphocytes', 'lymo'],
        'monocyte': ['mo', 'mono', 'monos', 'mnc', 'monocytes'],
        'granulocyte': ['gran', 'granulocytes'],
        'intact': ['intact_cells', 'intact_cells_population'],
        'viable': ['live', 'annexin-', 'live/dead stain'],
        'proliferated': ['cfse-', 'tracerviolet']
    }
    
    for standard, variations in synonyms.items():
        if m_lower in variations:
            return standard
            
    # 2. Regras estritas para estados dos marcadores (nível de deteção)
    # A notação (?i) diz ao Regex para ignorar maiúsculas/minúsculas.
    # O símbolo $ garante que só substituímos se estiver no FINAL da palavra (ex: não vai alterar a palavra 'medio' no meio de um nome).
    state_mappings = [
        (r'(?i)neg$', '-'),                  # Negative
        (r'(?i)pos$', '+'),                  # Positive
        (r'(?i)(dim|lo)$', '+-'),            # Low
        (r'(?i)(int|medium|med)$', '+~'),    # Intermediate
        (r'(?i)(bright|hi)$', '++')          # High
    ]
    
    for pattern, replacement in state_mappings:
        if re.search(pattern, m):
            # Faz a substituição mantendo o resto do nome do marcador intacto
            return re.sub(pattern, replacement, m, flags=re.IGNORECASE)
            
    return m

def parse_definition(definition):
    """
    Limpa e normaliza uma definição completa de população.
    """
    # Se a linha for nula, ignora
    if pd.isna(definition):
        return ""
        
    # Remove as anotações estatísticas do final da string (ex: ,Freq. of Parent)
    def_clean = re.sub(r',Freq\.? of.*', '', str(definition), flags=re.IGNORECASE)
    
    # Divide a string onde houver uma barra (/) ou vírgula (,)
    markers = re.split(r'[/,]', def_clean)
    
    # Normaliza cada marcador encontrado
    normalized_markers = [normalize_marker(m) for m in markers if m.strip()]
    
    # Ordena alfabeticamente para garantir correspondência (ex: CD3+, CD4+ == CD4+, CD3+)
    normalized_markers.sort()
    
    # Junta tudo novamente
    return ", ".join(normalized_markers)

# ==========================================
# CÓDIGO DE EXECUÇÃO
# ==========================================

# 1. Carrega o ficheiro CSV
# Substitua 'cell_definitions_list.csv' pelo caminho correto do seu ficheiro
df = pd.read_csv("cell_definitions_list.csv")

# 2. Aplica a função de normalização criando uma nova coluna
nome_da_coluna = 'Population Definition Reported' # Confirme se o nome da coluna é este
df['Normalized_Definition'] = df[nome_da_coluna].apply(parse_definition)
df.to_csv("cell_definitions_with_normalized.csv", index=False)
# # 3. Encontra e agrupa todas as linhas que se revelaram idênticas após a limpeza
# duplicates = df[df.duplicated('Normalized_Definition', keep=False)].sort_values('Normalized_Definition')

# print(f"Total de linhas processadas: {len(df)}")
# print(f"Total de definições únicas após normalização: {df['Normalized_Definition'].nunique()}")

# # 4. Guarda as correspondências encontradas num novo ficheiro CSV
# duplicates.to_csv('normalized_cell_definitions.csv', index=False)
print("Ficheiro 'normalized_cell_definitions.csv' guardado com sucesso!")