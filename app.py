from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Análise Rio Mearim (2019-2024)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg-main: #f3f7f6;
    --bg-card: #ffffff;
    --text-main: #1f2d2a;
    --accent: #0f766e;
    --accent-soft: #d7efeb;
}
.stApp {
    background:
        radial-gradient(circle at 10% 10%, #dff3f0 0%, rgba(223, 243, 240, 0) 40%),
        radial-gradient(circle at 85% 15%, #e8f5ef 0%, rgba(232, 245, 239, 0) 35%),
        var(--bg-main);
    color: var(--text-main);
}
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid #d9e3e0;
    border-radius: 14px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] {
    color: var(--accent);
}
</style>
""",
    unsafe_allow_html=True,
)

DISPLAY_NAMES = {
    "municipio": "Nome do município",
    "corpo_agua": "Nome do corpo d'água",
    "data_coleta": "Data da coleta",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "temp_agua": "Temperatura da água (°C)",
    "temp_ar": "Temperatura do ar (°C)",
    "oxigenio_dissolvido": "Oxigênio dissolvido (mg/L)",
    "condutividade_eletrica": "Condutividade elétrica (µS/cm)",
    "turbidez": "Turbidez (NTU)",
    "salinidade": "Salinidade (‰)",
    "alcalinidade": "Alcalinidade (mg/L)",
}

INDICATOR_COLUMNS = [
    "temp_agua",
    "temp_ar",
    "oxigenio_dissolvido",
    "condutividade_eletrica",
    "turbidez",
    "salinidade",
    "alcalinidade",
]

COLUMN_RULES = {
    "municipio": [["nome", "municipio"], ["municipio"]],
    "corpo_agua": [["nome", "corpo", "agua"], ["corpo", "agua"]],
    "data_coleta": [["data", "coleta"], ["data"]],
    "latitude": [["posicao", "horizontal", "coleta", "latitude"], ["latitude"]],
    "longitude": [["posicao", "vertical", "coleta", "longitude"], ["longitude"]],
    "temp_agua": [["temperatura", "agua"]],
    "temp_ar": [["temperatura", "ar"]],
    "oxigenio_dissolvido": [["oxigenio", "dissolvido"]],
    "condutividade_eletrica": [["condutividade", "eletrica"]],
    "turbidez": [["turbidez"]],
    "salinidade": [["salinidade"]],
    "alcalinidade": [["alcalinidade"]],
}

MISSING_TOKENS = {
    "",
    "na",
    "n a",
    "nan",
    "none",
    "null",
    "nd",
    "n d",
    "ni",
    "nm",
    "nao medido",
    "nao informado",
}

CONAMA_LIMITS_DOCES = {
    "Classe 1": {
        "oxigenio_dissolvido": {"comparison": "min", "limit": 6.0, "display": "OD >= 6 mg/L"},
        "turbidez": {"comparison": "max", "limit": 40.0, "display": "Turbidez <= 40 UNT"},
    },
    "Classe 2": {
        "oxigenio_dissolvido": {"comparison": "min", "limit": 5.0, "display": "OD >= 5 mg/L"},
        "turbidez": {"comparison": "max", "limit": 100.0, "display": "Turbidez <= 100 UNT"},
    },
    "Classe 3": {
        "oxigenio_dissolvido": {"comparison": "min", "limit": 4.0, "display": "OD >= 4 mg/L"},
        "turbidez": {"comparison": "max", "limit": 100.0, "display": "Turbidez <= 100 UNT"},
    },
}

CONAMA_STATUS_ORDER = ["Conforme", "Nao conforme", "Sem dados"]
MAP_STYLE_OPTIONS = {
    "OpenStreetMap": "open-street-map",
    "Carto Positron (claro)": "carto-positron",
    "Carto DarkMatter (escuro)": "carto-darkmatter",
    "White Background": "white-bg",
}

BRAZIL_STATE_LABELS = [
    {"uf": "AC", "estado": "Acre", "lat": -9.97, "lon": -67.81},
    {"uf": "AL", "estado": "Alagoas", "lat": -9.66, "lon": -35.74},
    {"uf": "AP", "estado": "Amapa", "lat": 0.03, "lon": -51.05},
    {"uf": "AM", "estado": "Amazonas", "lat": -3.10, "lon": -60.02},
    {"uf": "BA", "estado": "Bahia", "lat": -12.97, "lon": -38.50},
    {"uf": "CE", "estado": "Ceara", "lat": -3.73, "lon": -38.52},
    {"uf": "DF", "estado": "Distrito Federal", "lat": -15.79, "lon": -47.88},
    {"uf": "ES", "estado": "Espirito Santo", "lat": -20.31, "lon": -40.34},
    {"uf": "GO", "estado": "Goias", "lat": -16.68, "lon": -49.25},
    {"uf": "MA", "estado": "Maranhao", "lat": -2.53, "lon": -44.30},
    {"uf": "MT", "estado": "Mato Grosso", "lat": -15.60, "lon": -56.10},
    {"uf": "MS", "estado": "Mato Grosso do Sul", "lat": -20.47, "lon": -54.62},
    {"uf": "MG", "estado": "Minas Gerais", "lat": -19.92, "lon": -43.94},
    {"uf": "PA", "estado": "Para", "lat": -1.45, "lon": -48.50},
    {"uf": "PB", "estado": "Paraiba", "lat": -7.12, "lon": -34.86},
    {"uf": "PR", "estado": "Parana", "lat": -25.43, "lon": -49.27},
    {"uf": "PE", "estado": "Pernambuco", "lat": -8.05, "lon": -34.88},
    {"uf": "PI", "estado": "Piaui", "lat": -5.09, "lon": -42.80},
    {"uf": "RJ", "estado": "Rio de Janeiro", "lat": -22.91, "lon": -43.17},
    {"uf": "RN", "estado": "Rio Grande do Norte", "lat": -5.79, "lon": -35.21},
    {"uf": "RS", "estado": "Rio Grande do Sul", "lat": -30.03, "lon": -51.23},
    {"uf": "RO", "estado": "Rondonia", "lat": -8.76, "lon": -63.90},
    {"uf": "RR", "estado": "Roraima", "lat": 2.82, "lon": -60.67},
    {"uf": "SC", "estado": "Santa Catarina", "lat": -27.59, "lon": -48.55},
    {"uf": "SP", "estado": "Sao Paulo", "lat": -23.55, "lon": -46.64},
    {"uf": "SE", "estado": "Sergipe", "lat": -10.91, "lon": -37.07},
    {"uf": "TO", "estado": "Tocantins", "lat": -10.18, "lon": -48.33},
]


def normalize_text(value: object) -> str:
    text = str(value) if value is not None else ""
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    text = text.lower().replace("°", " ").replace("º", " ").replace("ª", " ")
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_sheet_metadata(sheet_name: str) -> dict[str, object]:
    cleaned = normalize_text(sheet_name)
    year_match = re.search(r"(20\d{2})", cleaned)
    period_match = re.search(r"(\d+)\s*periodo", cleaned)
    campaign_match = re.search(r"(\d+)\s*a?\s*campanha", cleaned)

    return {
        "ano": int(year_match.group(1)) if year_match else np.nan,
        "periodo": int(period_match.group(1)) if period_match else np.nan,
        "campanha": int(campaign_match.group(1)) if campaign_match else np.nan,
    }


def choose_column(
    columns: list[str], token_sets: list[list[str]], used_columns: set[str]
) -> str | None:
    best_col = None
    best_score = -1
    normalized = {col: normalize_text(col) for col in columns}

    for col in columns:
        if col in used_columns:
            continue
        name = normalized[col]
        for tokens in token_sets:
            if all(token in name for token in tokens):
                score = len(tokens) * 100 + len(name)
                if score > best_score:
                    best_score = score
                    best_col = col
    return best_col


def map_columns(raw_columns: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    used: set[str] = set()
    for target, token_sets in COLUMN_RULES.items():
        chosen = choose_column(raw_columns, token_sets, used)
        if chosen:
            selected[target] = chosen
            used.add(chosen)
    return selected


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan
    if normalize_text(text) in MISSING_TOKENS:
        return np.nan

    text = (
        text.replace(",", ".")
        .replace("−", "-")
        .replace("–", "-")
        .replace("~", "")
        .replace("<", "")
        .replace(">", "")
    )
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def parse_coordinate(value: object, coord_kind: str) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        coord = float(value)
    else:
        text = str(value).strip().upper()
        if not text or normalize_text(text) in MISSING_TOKENS:
            return np.nan

        text = text.replace(",", ".").replace("º", "°").replace("’", "'").replace("”", '"')
        parts = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not parts:
            return np.nan

        is_dms = any(symbol in text for symbol in ("°", "'", '"'))
        if is_dms:
            deg = abs(float(parts[0]))
            minutes = float(parts[1]) if len(parts) > 1 else 0.0
            seconds = float(parts[2]) if len(parts) > 2 else 0.0
            coord = deg + (minutes / 60.0) + (seconds / 3600.0)
        else:
            coord = float(parts[0])

        if float(parts[0]) < 0:
            coord = -abs(coord)
        else:
            if coord_kind == "lat":
                coord = -abs(coord) if "S" in text else abs(coord)
            else:
                west_markers = ("W", "O")
                coord = -abs(coord) if any(m in text for m in west_markers) else abs(coord)

    if coord_kind == "lat" and coord > 0:
        coord = -coord
    if coord_kind == "lon" and coord > 0:
        coord = -coord

    if coord_kind == "lat" and not (-15 <= coord <= 5):
        return np.nan
    if coord_kind == "lon" and not (-60 <= coord <= -35):
        return np.nan

    return coord


def build_sheet_frame(sheet_df: pd.DataFrame, sheet_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    mapping = map_columns(sheet_df.columns.tolist())
    metadata = parse_sheet_metadata(sheet_name)

    required = {"municipio", "corpo_agua", "data_coleta"}
    if not required.issubset(mapping.keys()):
        return pd.DataFrame(), {"aba": sheet_name, "status": "ignoradas_colunas"}

    prepared = pd.DataFrame({canon: sheet_df[src] for canon, src in mapping.items()})
    prepared["aba_origem"] = sheet_name
    prepared["ano"] = metadata["ano"]
    prepared["periodo"] = metadata["periodo"]
    prepared["campanha"] = metadata["campanha"]
    return prepared, {"aba": sheet_name, "status": "ok", "colunas_mapeadas": len(mapping)}


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["corpo_norm"] = df["corpo_agua"].map(normalize_text)
    df = df[df["corpo_norm"].str.contains("mearim", na=False)].copy()

    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce", dayfirst=True)
    for meta_col in ["ano", "periodo", "campanha"]:
        df[meta_col] = pd.to_numeric(df[meta_col], errors="coerce").astype("Int64")

    df["ano"] = df["ano"].fillna(df["data_coleta"].dt.year).astype("Int64")
    df["mes"] = df["data_coleta"].dt.month
    df["periodo_hidrologico"] = np.where(
        df["mes"].between(1, 6, inclusive="both"),
        "Chuvoso (jan-jun)",
        "Seco (jul-dez)",
    )
    df.loc[df["mes"].isna(), "periodo_hidrologico"] = np.nan

    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].astype(str).str.strip()

    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].apply(lambda v: parse_coordinate(v, "lat"))
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].apply(lambda v: parse_coordinate(v, "lon"))

    for col in INDICATOR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)
        else:
            df[col] = np.nan

    df.sort_values(by=["data_coleta", "municipio"], inplace=True, na_position="last")
    return df


def _load_dataset_from_excel(excel_source: str | BytesIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = pd.ExcelFile(excel_source, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for sheet_name in workbook.sheet_names:
        raw = workbook.parse(sheet_name=sheet_name)
        frame, status = build_sheet_frame(raw, sheet_name)
        diagnostics.append(status)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(diagnostics)

    merged = pd.concat(frames, ignore_index=True)
    finalized = finalize_dataframe(merged)
    diag_df = pd.DataFrame(diagnostics)
    return finalized, diag_df


@st.cache_data(show_spinner=False)
def load_dataset_from_path(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_dataset_from_excel(path)


@st.cache_data(show_spinner=False)
def load_dataset_from_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_dataset_from_excel(BytesIO(file_bytes))


def detect_default_excel() -> Path | None:
    patterns = [
        "*REMQAS*",
        "*NATILENE*",
        "*QUALI*",
        "*.xlsx",
    ]
    for pattern in patterns:
        files = sorted(Path(".").glob(pattern))
        if files:
            return files[0]
    return None


def detect_ifma_logo() -> Path | None:
    candidates = [
        Path("images.png"),
        Path("ifma.png"),
        Path("logo_ifma.png"),
        Path("assets/images.png"),
        Path("..") / "images.png",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def filter_dataframe(
    df: pd.DataFrame,
    anos: list[int],
    periodos: list[int],
    campanhas: list[int],
    municipios: list[str],
) -> pd.DataFrame:
    data = df.copy()
    if anos:
        data = data[data["ano"].isin(anos)]
    if periodos:
        data = data[data["periodo"].isin(periodos)]
    if campanhas:
        data = data[data["campanha"].isin(campanhas)]
    if municipios:
        data = data[data["municipio"].isin(municipios)]
    return data


def create_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    total = len(df)
    for col in INDICATOR_COLUMNS + ["latitude", "longitude", "data_coleta"]:
        valid = int(df[col].notna().sum()) if col in df.columns else 0
        rows.append(
            {
                "Campo": DISPLAY_NAMES.get(col, col),
                "Registros válidos": valid,
                "Cobertura (%)": round((valid / total) * 100, 1) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def classify_conama_limit(value: float, comparison: str, limit: float) -> str:
    if pd.isna(value):
        return "Sem dados"
    if comparison == "min":
        return "Conforme" if value >= limit else "Nao conforme"
    if comparison == "max":
        return "Conforme" if value <= limit else "Nao conforme"
    return "Sem dados"


def classify_conama_salinity(value: float) -> str:
    if pd.isna(value):
        return "Sem dados"
    if value <= 0.5:
        return "Agua doce (<=0.5 ‰)"
    if value < 30:
        return "Agua salobra (>0.5 e <30 ‰)"
    return "Agua salina (>=30 ‰)"


def apply_conama_classification(df: pd.DataFrame, classe_agua_doce: str) -> pd.DataFrame:
    if df.empty:
        return df

    rules = CONAMA_LIMITS_DOCES[classe_agua_doce]
    data = df.copy()
    tracked_cols: list[str] = []

    for param_col, rule in rules.items():
        status_col = f"conama_{param_col}_status"
        data[status_col] = data[param_col].apply(
            lambda v: classify_conama_limit(v, rule["comparison"], rule["limit"])
        )
        tracked_cols.append(status_col)

    data["conama_salinidade_classe"] = data["salinidade"].apply(classify_conama_salinity)

    status_matrix = data[tracked_cols]
    data["conama_regras_avaliadas"] = status_matrix.ne("Sem dados").sum(axis=1)
    data["conama_conformes"] = status_matrix.eq("Conforme").sum(axis=1)
    data["conama_nao_conformes"] = status_matrix.eq("Nao conforme").sum(axis=1)

    has_data = data["conama_regras_avaliadas"] > 0
    all_conforme = data["conama_nao_conformes"] == 0

    data["conama_conformidade_geral"] = np.where(
        ~has_data,
        "Sem dados",
        np.where(all_conforme, "Conforme", "Nao conforme"),
    )
    return data


def ratio_conforme(series: pd.Series) -> float:
    valid = series[series != "Sem dados"]
    if valid.empty:
        return np.nan
    return float((valid == "Conforme").mean() * 100)


def ratio_agua_doce(series: pd.Series) -> float:
    valid = series[series != "Sem dados"]
    if valid.empty:
        return np.nan
    return float(valid.str.startswith("Agua doce", na=False).mean() * 100)


def render_conama_section(df: pd.DataFrame, classe_agua_doce: str) -> None:
    if df.empty:
        return

    rules = CONAMA_LIMITS_DOCES[classe_agua_doce]
    od_rule = rules["oxigenio_dissolvido"]["display"]
    turb_rule = rules["turbidez"]["display"]

    st.subheader(f"Classificacao CONAMA 357/2005 ({classe_agua_doce} - aguas doces)")
    st.caption(
        f"Limites aplicados: {od_rule}; {turb_rule}. "
        "Salinidade: agua doce <=0.5 ‰, salobra >0.5 e <30 ‰, salina >=30 ‰."
    )

    conformidade_geral = ratio_conforme(df["conama_conformidade_geral"])
    conformidade_od = ratio_conforme(df["conama_oxigenio_dissolvido_status"])
    conformidade_turbidez = ratio_conforme(df["conama_turbidez_status"])
    proporcao_agua_doce = ratio_agua_doce(df["conama_salinidade_classe"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Conformidade geral",
        f"{conformidade_geral:.1f}%" if pd.notna(conformidade_geral) else "n/d",
    )
    c2.metric(
        "OD conforme",
        f"{conformidade_od:.1f}%" if pd.notna(conformidade_od) else "n/d",
    )
    c3.metric(
        "Turbidez conforme",
        f"{conformidade_turbidez:.1f}%" if pd.notna(conformidade_turbidez) else "n/d",
    )
    c4.metric(
        "Amostras agua doce",
        f"{proporcao_agua_doce:.1f}%" if pd.notna(proporcao_agua_doce) else "n/d",
    )

    col1, col2 = st.columns(2)
    with col1:
        annual_status = (
            df.groupby(["ano", "conama_conformidade_geral"], as_index=False)
            .size()
            .rename(columns={"size": "amostras"})
        )
        annual_status = annual_status[annual_status["conama_conformidade_geral"].isin(["Conforme", "Nao conforme"])]
        if not annual_status.empty:
            annual_status["percentual"] = (
                annual_status["amostras"]
                / annual_status.groupby("ano")["amostras"].transform("sum")
                * 100
            )
            fig = px.bar(
                annual_status,
                x="ano",
                y="percentual",
                color="conama_conformidade_geral",
                barmode="stack",
                color_discrete_map={"Conforme": "#0f766e", "Nao conforme": "#b45309"},
                title="Conformidade geral por ano (%)",
                labels={
                    "ano": "Ano",
                    "percentual": "Percentual (%)",
                    "conama_conformidade_geral": "Status",
                },
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados suficientes para a conformidade geral anual.")

    with col2:
        parameter_status = pd.DataFrame(
            {
                "Parametro": ["Oxigenio dissolvido", "Turbidez"],
                "Conformidade (%)": [conformidade_od, conformidade_turbidez],
            }
        ).dropna(subset=["Conformidade (%)"])

        if not parameter_status.empty:
            fig = px.bar(
                parameter_status,
                x="Parametro",
                y="Conformidade (%)",
                color="Parametro",
                title="Conformidade por parametro (%)",
                text_auto=".1f",
            )
            fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para calcular conformidade por parametro.")

    with st.expander("Amostras classificadas pela CONAMA"):
        preview_cols = [
            "data_coleta",
            "municipio",
            "periodo",
            "campanha",
            "oxigenio_dissolvido",
            "conama_oxigenio_dissolvido_status",
            "turbidez",
            "conama_turbidez_status",
            "salinidade",
            "conama_salinidade_classe",
            "conama_conformidade_geral",
        ]
        preview_cols = [col for col in preview_cols if col in df.columns]
        preview_df = df[preview_cols].sort_values(by="data_coleta", ascending=False)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


def indicator_insights(df: pd.DataFrame, indicator_col: str) -> list[str]:
    if df.empty:
        return []

    series = df.dropna(subset=["ano", indicator_col]).groupby("ano", as_index=False)[indicator_col].mean()
    if series.empty:
        return []

    insights: list[str] = []
    high = series.loc[series[indicator_col].idxmax()]
    low = series.loc[series[indicator_col].idxmin()]
    insights.append(
        f"Maior média anual de {DISPLAY_NAMES[indicator_col]}: {high[indicator_col]:.2f} em {int(high['ano'])}."
    )
    insights.append(
        f"Menor média anual de {DISPLAY_NAMES[indicator_col]}: {low[indicator_col]:.2f} em {int(low['ano'])}."
    )

    if len(series) >= 2:
        first = series.iloc[0]
        last = series.iloc[-1]
        if first[indicator_col] != 0:
            change = ((last[indicator_col] - first[indicator_col]) / first[indicator_col]) * 100
            insights.append(
                f"Variação de {int(first['ano'])} para {int(last['ano'])}: {change:+.1f}% em média anual."
            )

    return insights


def render_chart_area(df: pd.DataFrame, indicator_col: str) -> None:
    label = DISPLAY_NAMES[indicator_col]
    color_scale = "Tealgrn"

    col1, col2 = st.columns(2)
    with col1:
        yearly = (
            df.dropna(subset=["ano", indicator_col])
            .groupby("ano", as_index=False)[indicator_col]
            .mean()
            .sort_values("ano")
        )
        if not yearly.empty:
            fig = px.line(
                yearly,
                x="ano",
                y=indicator_col,
                markers=True,
                title=f"Evolução anual: {label}",
                color_discrete_sequence=["#0f766e"],
                labels={"ano": "Ano", indicator_col: label},
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados suficientes para a evolução anual.")

    with col2:
        box_df = df.dropna(subset=["ano", indicator_col]).copy()
        if not box_df.empty:
            box_df["ano"] = box_df["ano"].astype(str)
            fig = px.box(
                box_df,
                x="ano",
                y=indicator_col,
                color="ano",
                title=f"Distribuição por ano: {label}",
                labels={"ano": "Ano", indicator_col: label},
            )
            fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados suficientes para distribuição anual.")

    comp_df = (
        df.dropna(subset=["ano", "periodo", "campanha", indicator_col])
        .groupby(["ano", "periodo", "campanha"], as_index=False)[indicator_col]
        .mean()
        .sort_values(["ano", "periodo", "campanha"])
    )
    if not comp_df.empty:
        comp_df["campanha"] = comp_df["campanha"].astype("Int64").astype(str)
        fig = px.bar(
            comp_df,
            x="ano",
            y=indicator_col,
            color="campanha",
            facet_col="periodo",
            facet_col_wrap=4,
            barmode="group",
            title=f"Comparativo por período e campanha: {label}",
            labels={"ano": "Ano", indicator_col: label, "campanha": "Campanha"},
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=70, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados para comparar períodos e campanhas.")

    pivot_df = df.pivot_table(values=indicator_col, index="periodo", columns="ano", aggfunc="mean")
    if not pivot_df.empty:
        fig = px.imshow(
            pivot_df.sort_index(),
            text_auto=".2f",
            color_continuous_scale=color_scale,
            aspect="auto",
            labels={"x": "Ano", "y": "Período", "color": label},
            title=f"Mapa de calor (média): {label}",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

def render_collection_points_map(df: pd.DataFrame, map_style: str) -> None:
    points = df.dropna(subset=["latitude", "longitude"]).copy()
    if points.empty:
        st.info("Sem coordenadas válidas para exibir os pontos de coleta.")
        return

    points_plot = pd.DataFrame(
        {
            "latitude": pd.to_numeric(points["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(points["longitude"], errors="coerce"),
            "municipio": points["municipio"].astype(str),
            "ano": pd.to_numeric(points["ano"], errors="coerce"),
            "campanha": pd.to_numeric(points["campanha"], errors="coerce"),
            "data_coleta": points["data_coleta"],
        }
    ).dropna(subset=["latitude", "longitude"])

    if points_plot.empty:
        st.info("Sem coordenadas válidas para exibir os pontos de coleta.")
        return

    city_plot = (
        points_plot.groupby("municipio", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            qtd_coletas=("municipio", "size"),
            anos_monitorados=("ano", "nunique"),
            campanhas_monitoradas=("campanha", "nunique"),
            primeira_coleta=("data_coleta", "min"),
            ultima_coleta=("data_coleta", "max"),
        )
        .sort_values("qtd_coletas", ascending=False)
    )
    city_plot["primeira_coleta_txt"] = city_plot["primeira_coleta"].dt.strftime("%d/%m/%Y").fillna("n/d")
    city_plot["ultima_coleta_txt"] = city_plot["ultima_coleta"].dt.strftime("%d/%m/%Y").fillna("n/d")

    city_plot["qtd_coletas"] = city_plot["qtd_coletas"].astype(int)
    city_plot["anos_monitorados"] = city_plot["anos_monitorados"].astype(int)
    city_plot["campanhas_monitoradas"] = city_plot["campanhas_monitoradas"].astype(int)
    city_plot["latitude"] = city_plot["latitude"].round(6)
    city_plot["longitude"] = city_plot["longitude"].round(6)

    center = {"lat": -14.20, "lon": -52.90}
    zoom_level = 3.5
    effective_style = "carto-positron" if map_style == "white-bg" else map_style

    try:
        fig = px.scatter_map(
            city_plot,
            lat="latitude",
            lon="longitude",
            color="qtd_coletas",
            size="qtd_coletas",
            size_max=34,
            text="municipio",
            hover_name="municipio",
            hover_data={
                "qtd_coletas": True,
                "anos_monitorados": True,
                "campanhas_monitoradas": True,
                "primeira_coleta_txt": True,
                "ultima_coleta_txt": True,
                "latitude": ":.5f",
                "longitude": ":.5f",
            },
            map_style=effective_style,
            zoom=zoom_level,
            center=center,
            color_continuous_scale="Tealgrn",
            title="Pontos de coleta do Rio Mearim (Latitude/longitude)",
        )
        state_df = pd.DataFrame(BRAZIL_STATE_LABELS)
        fig.add_trace(
            go.Scattermap(
                lat=state_df["lat"].tolist(),
                lon=state_df["lon"].tolist(),
                mode="text",
                text=state_df["uf"].tolist(),
                customdata=state_df[["estado"]].to_numpy(),
                textfont=dict(size=10, color="#334155"),
                hovertemplate="<b>%{customdata[0]}</b> (%{text})<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_traces(
            textposition="top center",
        )
        if fig.data:
            fig.data[0].marker.opacity = 0.88
        fig.update_layout(
            height=640,
            margin=dict(l=10, r=10, t=60, b=10),
            coloraxis_colorbar=dict(title="Nº coletas"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        marker_sizes = (city_plot["qtd_coletas"].astype(float) * 2.2).clip(lower=12, upper=38)
        fig = go.Figure(
            data=[
                go.Scattergeo(
                    lon=city_plot["longitude"].astype(float).tolist(),
                    lat=city_plot["latitude"].astype(float).tolist(),
                    text=city_plot["municipio"].astype(str).tolist(),
                    mode="markers+text",
                    textposition="top center",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + "Coletas: %{customdata[0]}<br>"
                        + "Anos monitorados: %{customdata[1]}<br>"
                        + "Campanhas monitoradas: %{customdata[2]}<br>"
                        + "Primeira coleta: %{customdata[3]}<br>"
                        + "Última coleta: %{customdata[4]}<br>"
                        + "Latitude: %{lat:.5f}<br>"
                        + "Longitude: %{lon:.5f}<extra></extra>"
                    ),
                    customdata=np.stack(
                        [
                            city_plot["qtd_coletas"].to_numpy(),
                            city_plot["anos_monitorados"].to_numpy(),
                            city_plot["campanhas_monitoradas"].to_numpy(),
                            city_plot["primeira_coleta_txt"].to_numpy(),
                            city_plot["ultima_coleta_txt"].to_numpy(),
                        ],
                        axis=-1,
                    ),
                    marker=dict(
                        size=marker_sizes.to_list(),
                        color=city_plot["qtd_coletas"].astype(float).to_list(),
                        colorscale="Tealgrn",
                        line=dict(width=0.8, color="white"),
                        opacity=0.9,
                        colorbar=dict(title="Nº coletas"),
                        showscale=True,
                    ),
                )
            ]
        )
        fig.update_geos(
            fitbounds="locations",
            showcountries=True,
            showcoastlines=True,
            showland=True,
            landcolor="#eef4f2",
            countrycolor="#98a5a0",
            coastlinecolor="#98a5a0",
        )
        fig.update_layout(
            title="Pontos de coleta do Rio Mearim (Latitude/longitude)",
            height=620,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalhamento por cidade (coordenadas médias)"):
        details = city_plot[
            [
                "municipio",
                "latitude",
                "longitude",
                "qtd_coletas",
                "anos_monitorados",
                "campanhas_monitoradas",
                "primeira_coleta_txt",
                "ultima_coleta_txt",
            ]
        ].rename(
            columns={
                "municipio": "Município",
                "latitude": "Latitude média",
                "longitude": "Longitude média",
                "qtd_coletas": "Qtd. coletas",
                "anos_monitorados": "Anos monitorados",
                "campanhas_monitoradas": "Campanhas monitoradas",
                "primeira_coleta_txt": "Primeira coleta",
                "ultima_coleta_txt": "Última coleta",
            }
        )
        st.dataframe(details, use_container_width=True, hide_index=True)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["ano", "periodo", "campanha"], dropna=False)
        .agg(
            amostras=("municipio", "size"),
            municipios=("municipio", "nunique"),
            temp_agua=("temp_agua", "mean"),
            temp_ar=("temp_ar", "mean"),
            oxigenio_dissolvido=("oxigenio_dissolvido", "mean"),
            condutividade_eletrica=("condutividade_eletrica", "mean"),
            turbidez=("turbidez", "mean"),
            salinidade=("salinidade", "mean"),
            alcalinidade=("alcalinidade", "mean"),
            conama_geral_pct=("conama_conformidade_geral", ratio_conforme),
            conama_od_pct=("conama_oxigenio_dissolvido_status", ratio_conforme),
            conama_turbidez_pct=("conama_turbidez_status", ratio_conforme),
            conama_agua_doce_pct=("conama_salinidade_classe", ratio_agua_doce),
        )
        .reset_index()
        .sort_values(["ano", "periodo", "campanha"])
    )

    rename_map = {
        "ano": "Ano",
        "periodo": "Período",
        "campanha": "Campanha",
        "amostras": "Amostras",
        "municipios": "Municípios",
        "temp_agua": DISPLAY_NAMES["temp_agua"],
        "temp_ar": DISPLAY_NAMES["temp_ar"],
        "oxigenio_dissolvido": DISPLAY_NAMES["oxigenio_dissolvido"],
        "condutividade_eletrica": DISPLAY_NAMES["condutividade_eletrica"],
        "turbidez": DISPLAY_NAMES["turbidez"],
        "salinidade": DISPLAY_NAMES["salinidade"],
        "alcalinidade": DISPLAY_NAMES["alcalinidade"],
        "conama_geral_pct": "CONAMA geral (%)",
        "conama_od_pct": "CONAMA OD (%)",
        "conama_turbidez_pct": "CONAMA turbidez (%)",
        "conama_agua_doce_pct": "CONAMA agua doce (%)",
    }
    summary.rename(columns=rename_map, inplace=True)
    percent_cols = [
        "CONAMA geral (%)",
        "CONAMA OD (%)",
        "CONAMA turbidez (%)",
        "CONAMA agua doce (%)",
    ]
    for col in percent_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(1)
    return summary


def main() -> None:
    logo_path = detect_ifma_logo()
    header_col_logo, header_col_text = st.columns([1, 5], vertical_alignment="top")
    with header_col_logo:
        if logo_path is not None:
            st.image(str(logo_path), width=170)
    with header_col_text:
        st.markdown("**Instituto Federal do Maranhão (IFMA)**")
        st.markdown("Trabalho de Conclusão de Curso - Sistemas de Informação")
        st.markdown("Aluno: **Rafael Mendes Carneiro**")
        st.caption("Campus São Luís - Monte Castelo")
        st.title("Rio Mearim: análise da qualidade da água (2019-2024)")

    if logo_path is None:
        st.info("Logo IFMA não encontrada. Coloque `images.png` na pasta do projeto.")

    st.caption(
        "Aplicação para leitura e tratamento da planilha multiabas com comparativos por ano, período e campanha."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Carregar planilha .xlsx (opcional)", type=["xlsx"], accept_multiple_files=False
    )
    default_path = detect_default_excel()

    if uploaded_file is not None:
        data, diagnostics = load_dataset_from_bytes(uploaded_file.getvalue())
        source_name = uploaded_file.name
    elif default_path:
        data, diagnostics = load_dataset_from_path(str(default_path))
        source_name = default_path.name
    else:
        st.error("Nenhuma planilha .xlsx encontrada. Faça upload na barra lateral.")
        return

    st.sidebar.markdown(f"**Fonte:** `{source_name}`")

    if data.empty:
        st.error("Não foi possível carregar dados do Rio Mearim com as colunas mínimas esperadas.")
        if not diagnostics.empty:
            st.dataframe(diagnostics, use_container_width=True)
        return

    anos = sorted(int(v) for v in data["ano"].dropna().unique())
    periodos = sorted(int(v) for v in data["periodo"].dropna().unique())
    campanhas = sorted(int(v) for v in data["campanha"].dropna().unique())
    municipios = sorted(m for m in data["municipio"].dropna().unique())

    selected_years = st.sidebar.multiselect("Ano", anos, default=anos)
    selected_periods = st.sidebar.multiselect("Período", periodos, default=periodos)
    selected_campaigns = st.sidebar.multiselect("Campanha", campanhas, default=campanhas)
    selected_municipios = st.sidebar.multiselect("Município", municipios, default=municipios)
    selected_map_style_label = st.sidebar.selectbox(
        "Tipo de mapa",
        list(MAP_STYLE_OPTIONS.keys()),
        index=1,
    )
    selected_map_style = MAP_STYLE_OPTIONS[selected_map_style_label]
    classe_conama = st.sidebar.selectbox(
        "Classe CONAMA (aguas doces)",
        list(CONAMA_LIMITS_DOCES.keys()),
        index=1,
    )

    indicator_label = st.sidebar.selectbox(
        "Indicador principal",
        [DISPLAY_NAMES[c] for c in INDICATOR_COLUMNS],
        index=2,
    )
    indicator_col = next(col for col, name in DISPLAY_NAMES.items() if name == indicator_label)

    filtered = filter_dataframe(
        data,
        anos=selected_years,
        periodos=selected_periods,
        campanhas=selected_campaigns,
        municipios=selected_municipios,
    )
    filtered = apply_conama_classification(filtered, classe_conama)

    if filtered.empty:
        st.warning("Os filtros atuais não retornaram dados.")
        return

    total_samples = len(filtered)
    total_municipios = filtered["municipio"].nunique()
    total_campaigns = (
        filtered[["ano", "periodo", "campanha"]].dropna().drop_duplicates().shape[0]
    )
    mean_indicator = filtered[indicator_col].mean(skipna=True)
    median_indicator = filtered[indicator_col].median(skipna=True)
    valid_geo = filtered.dropna(subset=["latitude", "longitude"]).shape[0]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Amostras", f"{total_samples}")
    m2.metric("Municípios", f"{total_municipios}")
    m3.metric("Campanhas", f"{total_campaigns}")
    m4.metric("Média indicador", f"{mean_indicator:.2f}" if pd.notna(mean_indicator) else "n/d")
    m5.metric("Mediana indicador", f"{median_indicator:.2f}" if pd.notna(median_indicator) else "n/d")
    m6.metric("Pontos georreferenciados", f"{valid_geo}")

    st.subheader("Análise comparativa")
    for insight in indicator_insights(filtered, indicator_col):
        st.markdown(f"- {insight}")

    render_chart_area(filtered, indicator_col)
    st.subheader("Pontos de coleta (coordenadas)")
    render_collection_points_map(filtered, selected_map_style)
    render_conama_section(filtered, classe_conama)

    st.subheader("Resumo por ano, período e campanha")
    summary_table = build_summary_table(filtered)
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    csv_data = summary_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Baixar resumo em CSV",
        data=csv_data,
        file_name="resumo_rio_mearim_2019_2024.csv",
        mime="text/csv",
    )

    quality_table = create_quality_table(filtered)
    with st.expander("Diagnóstico de qualidade dos dados"):
        st.markdown("Cobertura dos campos após o tratamento e filtros aplicados.")
        st.dataframe(quality_table, use_container_width=True, hide_index=True)
        if not diagnostics.empty:
            st.markdown("Status da leitura por aba:")
            st.dataframe(diagnostics, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
