import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# ---------------------------------------------------
#  Caching the data load/clean step so UI stays snappy
# ---------------------------------------------------
@st.cache_data
def load_and_clean(excel_bytes: bytes):
    """Loads the three relevant sheets and performs basic cleaning/typing."""
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    df_main     = pd.read_excel(xls, sheet_name="Baseline + Actual Dates")
    df_critical = pd.read_excel(xls, sheet_name="Need date and critical status")
    df_cirm     = pd.read_excel(xls, sheet_name="CIRM")

    # normalise PNOC IDs
    for df in (df_main, df_critical, df_cirm):
        df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # parse dates
    df_main['Baseline'] = pd.to_datetime(df_main['Baseline'], format="%m/%d/%Y", errors="coerce")
    df_main['Actual']   = pd.to_datetime(df_main['Actual'],   format="%m/%d/%Y", errors="coerce")
    df_critical['needed_date'] = pd.to_datetime(df_critical['needed_date'], format="%d-%b-%y", errors="coerce")

    # merge CIRM fields (RM / CI) into main
    df = df_main.merge(df_cirm[['PNOC ID', 'RM', 'CI']], on='PNOC ID', how='left')

    # derive Total FAN Reviews
    df['Total FAN Reviews']  = 4
    df['Total FAN Reviews'] += df['RM'].apply(lambda x: 1 if pd.notna(x) and x != 0 else 0)
    df['Total FAN Reviews'] += df['CI'].apply(lambda x: max(0, x-1) if pd.notna(x) else 0)

    # bring in critical flag
    df = df.merge(df_critical[['PNOC ID', 'critical']], on='PNOC ID', how='left')

    return df

# ---------------------------------------------------
#  Business‑day helper that is NaT‑safe
# ---------------------------------------------------
US_HOL_2024 = np.array([
    datetime(2024,1,1),  datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4),  datetime(2024,9,2),  datetime(2024,10,14),
    datetime(2024,11,11), datetime(2024,11,28), datetime(2024,12,25)
], dtype='datetime64[D]')

def safe_busday_count(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Vectorised wrapper around np.busday_count that skips NaT rows."""
    # mask valid rows
    valid = start.notna() & end.notna()
    result = np.full(len(start), np.nan)
    if valid.any():
        start_arr = start[valid].dt.normalize().values.astype('datetime64[D]')
        end_arr   = end[valid].dt.normalize().values.astype('datetime64[D]')
        result[valid] = np.busday_count(start_arr, end_arr, holidays=US_HOL_2024)
    return result

# ---------------------------------------------------
#  Analysis logic
# ---------------------------------------------------

def analyze(df: pd.DataFrame, types: list[str], min_fans: int):
    # filter by critical / routine
    mask = pd.Series(False, index=df.index)
    if 'Critical' in types:
        mask |= (df['critical'] == 'Y')
    if 'Routine' in types:
        mask |= (df['critical'] != 'Y')
    df = df[mask]

    # fan‑review filter
    df = df[df['Total FAN Reviews'] >= min_fans]
    if df.empty:
        return None, None

    # compute business‑day differences with NaT‑safety
    df = df.copy()
    df['Business_Days_Difference'] = safe_busday_count(df['Baseline'], df['Actual'])

    # drop any rows that are still NaN (means Baseline/Actual were NaT)
    df = df.dropna(subset=['Business_Days_Difference'])
    if df.empty:
        return None, None

    # average per PNOC
    bsa = (
        df.groupby('PNOC ID')['Business_Days_Difference']
          .mean()
          .sort_values(ascending=False)
    )

    def category(v):
        if v < 0: return 'Delayed'
        if v <= 1: return 'On Time'
        return 'Ahead'

    summary = pd.DataFrame({
        'PNOC ID': bsa.index,
        'BSA (Avg Days)': bsa.values,
        'Category': [category(v) for v in bsa.values]
    })
    return bsa, summary

# ---------------------------------------------------
#  Streamlit UI
# ---------------------------------------------------

def main():
    st.set_page_config(page_title='Advanced Coordinated Analysis', layout='wide')
    st.title('Advanced Coordinated Analysis (ACA)')

    # ---- sidebar: upload ----
    st.sidebar.header('1. Upload your Excel')
    upload = st.sidebar.file_uploader('PNOCs 2024‑2025 .xlsm', type=['xls','xlsx','xlsm'])
    if not upload:
        st.info('🔍 Please upload your PNOCs Excel file to get started.')
        return

    df = load_and_clean(upload.read())

    # ---- filters ----
    st.sidebar.header('2. Filter PNOC Type')
    types = st.sidebar.multiselect('Choose PNOC categories', ['Routine','Critical'], default=['Routine','Critical'])
    st.sidebar.header('3. FAN review cycles')
    min_fans = st.sidebar.number_input('Min FAN reviews', min_value=4, value=4, step=1)

    # ---- run analysis ----
    if st.sidebar.button('Run Analysis'):
        bsa, summary = analyze(df, types, min_fans)
        if summary is None:
            st.warning('No data left after filtering / date cleanup. Try different filters or inspect your input file.')
            return

        import altair as alt
        chart = (
            alt.Chart(summary)
                .mark_bar()
                .encode(
                    x=alt.X('PNOC ID', sort='-y'),
                    y='BSA (Avg Days)',
                    color=alt.Color('Category', scale=alt.Scale(domain=['Ahead','On Time','Delayed'], range=['green','gold','crimson']))
                )
                .properties(width=850, height=400)
        )

        st.subheader('Baseline vs. Actual ▶️ BSA per PNOC')
        st.altair_chart(chart, use_container_width=True)
        st.subheader('Detailed Summary Table')
        st.dataframe(summary.set_index('PNOC ID'))

    st.sidebar.markdown('---')
    st.sidebar.write('© Your Team – Advanced Coordinated Analysis')

if __name__ == '__main__':
    main()
