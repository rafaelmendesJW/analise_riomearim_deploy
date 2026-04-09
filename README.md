# Análise Rio Mearim (2019-2024)

Aplicação em `Streamlit` para ler a planilha **DADOS QUALIÁGUA - REMQAS PARA NATILENE**, tratar variações de colunas entre abas e gerar análise comparativa do **Rio Mearim** por ano, período e campanha.

## O que a aplicação faz

- Lê múltiplas abas da planilha (`.xlsx`).
- Padroniza colunas com nomes diferentes (ex.: variações de acento e escrita).
- Filtra automaticamente somente o **Rio Mearim**.
- Trata datas, números e coordenadas (inclusive formato grau/minuto/segundo).
- Calcula indicadores e comparativos entre 2019 e 2024.
- Gera gráficos e mapa com os pontos de coleta.

## Campos utilizados

- Nome do município
- Nome do corpo d'água
- Data da coleta
- Latitude
- Longitude
- Temperatura da água
- Temperatura do ar
- Oxigênio dissolvido
- Condutividade elétrica
- Turbidez
- Salinidade
- Alcalinidade

## Ambiente virtual (venv)

### Opção 1: `venv` padrão

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Opção 2: fallback com `virtualenv` (quando o `venv` padrão falhar)

```powershell
python -m pip install --user virtualenv
python -m virtualenv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Execução

```powershell
streamlit run app.py
```

A aplicação tenta detectar automaticamente um arquivo `.xlsx` na pasta atual. Você também pode enviar o arquivo manualmente pela barra lateral.
