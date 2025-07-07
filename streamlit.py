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
#  Business‑day helper (Actual − Baseline, so late ⇒ negative)
# ---------------------------------------------------
US_HOL_2024 = np.array([
    datetime(2024,1,1),  datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4),  datetime(2024,9,2),  datetime(2024,10,14),
    datetime(2024,11,11), datetime(2024,11,28), datetime(2024,12,25)
], dtype='datetime64[D]')

def schedule_variance(baseline: pd.Series, actual: pd.Series) -> np.ndarray:
    """Return working‑day variance *Actual − Baseline* (late ⇒ −, early ⇒ +)."""
    valid = baseline.notna() & actual.notna()
    result = np.full(len(baseline), np.nan)
    if valid.any():
        base_arr = baseline[valid].dt.normalize().values.astype('datetime64[D]')
        act_arr  = actual[valid].dt.normalize().values.astype('datetime64[D]')
        # invert order so a late Actual gives negative days
        result[valid] = np.busday_count(act_arr, base_arr, holidays=US_HOL_2024)
    return result

# ---------------------------------------------------
#  Analysis logic
# ---------------------------------------------------

def analyze(df: pd.DataFrame, types: list[str], min_fans: int):
    mask = pd.Series(False, index=df.index)
    if 'Critical' in types:
        mask |= (df['critical'] == 'Y')
    if 'Routine' in types:
        mask |= (df['critical'] != 'Y')
    df = df[mask]

    df = df[df['Total FAN Reviews'] >= min_fans]
    if df.empty:
        return None, None

    df = df.copy()
    df['Schedule_Variance'] = schedule_variance(df['Baseline'], df['Actual'])
    df = df.dropna(subset=['Schedule_Variance'])
    if df.empty:
        return None, None

    # mean variance per PNOC
    bsa = (
        df.groupby('PNOC ID')['Schedule_Variance']
          .mean()
          .sort_values(ascending=False)
    )

    def bucket(v: float) -> str:
        if v >= 1:
            return 'Ahead'
        if v >= -1:
            return 'On Time'
        return 'Delayed'

    summary = pd.DataFrame({
        'PNOC ID': bsa.index,
        'Variance (Bus Days)': bsa.values,
        'Bucket': [bucket(v) for v in bsa.values]
    })
    return summary

# ---------------------------------------------------
#  Streamlit UI
# ---------------------------------------------------

def main():
    st.set_page_config(page_title='Advanced Coordinated Analysis', layout='wide')
    st.title('Advanced Coordinated Analysis (ACA)')

    # ---- sidebar upload ----
    st.sidebar.header('1. Upload your Excel')
    upload = st.sidebar.file_uploader('PNOCs 2024‑2025 .xlsm', type=['xls','xlsx','xlsm'])
    if not upload:
        st.info('🔍 Please upload your PNOCs Excel file to get started.')
        return

    df = load_and_clean(upload.read())

    # ---- sidebar filters ----
    st.sidebar.header('2. Filter PNOC Type')
    types = st.sidebar.multiselect('Choose PNOC categories', ['Routine','Critical'], default=['Routine','Critical'])
    st.sidebar.header('3. FAN review cycles')
    min_fans = st.sidebar.number_input('Min FAN reviews', min_value=4, value=4, step=1)

    if st.sidebar.button('Run Analysis'):
        summary = analyze(df, types, min_fans)
        if summary is None:
            st.warning('No data left after filtering / date cleanup. Try different filters or inspect your input file.')
            return

        # dynamic diverging color scale centred at 0
        max_abs = max(1, abs(summary['Variance (Bus Days)']).max())

        import altair as alt
        chart = (
            alt.Chart(summary)
               .mark_bar()
               .encode(
                    x=alt.X('PNOC ID:N', sort='-y', title='PNOC ID'),
                    y=alt.Y('Variance (Bus Days):Q', title='Schedule Variance (Business Days)'),
                    color=alt.Color('Variance (Bus Days):Q',
                                    scale=alt.Scale(domain=[-max_abs, 0, max_abs],
                                                    range=['crimson','lightgrey','seagreen']),
                                    legend=alt.Legend(title='Variance')),
                    tooltip=['PNOC ID','Variance (Bus Days)','Bucket']
               )
               .properties(width=900, height=450)
        )

        st.subheader('Schedule Variance ▶️ Actual vs Baseline (Business Days)')
        st.altair_chart(chart, use_container_width=True)

        st.subheader('Detailed Summary Table')
        st.dataframe(summary.set_index('PNOC ID'))

    st.sidebar.markdown('---')
    st.sidebar.write('© Your Team – Advanced Coordinated Analysis')

if __name__ == '__main__':
    main()


