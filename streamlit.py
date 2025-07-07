import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# ---------------------------------------------------
#  Caching the data load/clean step so UI stays snappy
# ---------------------------------------------------
@st.cache_data
def load_and_clean(excel_bytes: bytes) -> pd.DataFrame:
    """Loads and cleans the PNOC sheets, merging needed_date for critical PNOCs."""
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    df_main = pd.read_excel(xls, sheet_name="Baseline + Actual Dates")
    df_crit = pd.read_excel(xls, sheet_name="Need date and critical status")
    df_cirm = pd.read_excel(xls, sheet_name="CIRM")

    # Trim IDs
    for df in (df_main, df_crit, df_cirm):
        df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # Parse dates
    df_main['Baseline'] = pd.to_datetime(df_main['Baseline'], format="%m/%d/%Y", errors="coerce")
    df_main['Actual']   = pd.to_datetime(df_main['Actual'],   format="%m/%d/%Y", errors="coerce")
    df_crit['needed_date'] = pd.to_datetime(df_crit['needed_date'], format="%d-%b-%y", errors="coerce")

    # Merge CIRM data
    df = df_main.merge(
        df_cirm[['PNOC ID', 'RM', 'CI']], on='PNOC ID', how='left'
    )

    # Total FAN Reviews
    df['Total FAN Reviews'] = 4
    df['Total FAN Reviews'] += df['RM'].apply(lambda x: 1 if pd.notna(x) and x != 0 else 0)
    df['Total FAN Reviews'] += df['CI'].apply(lambda x: max(0, x - 1) if pd.notna(x) else 0)

    # Merge critical flag + needed_date
    df = df.merge(
        df_crit[['PNOC ID', 'critical', 'needed_date']],
        on='PNOC ID', how='left'
    )

    return df

# ---------------------------------------------------
#  Business-day variance: Actual - Baseline (busdays)
# ---------------------------------------------------
US_HOL_2024 = np.array([
    datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4), datetime(2024,9,2), datetime(2024,10,14),
    datetime(2024,11,11), datetime(2024,11,28), datetime(2024,12,25)
], dtype='datetime64[D]')

def schedule_variance(baseline: pd.Series, actual: pd.Series) -> np.ndarray:
    """Compute business-day variance: Actual -> Baseline. Early => +, Late => -."""
    valid = baseline.notna() & actual.notna()
    arr = np.full(len(baseline), np.nan)
    if valid.any():
        b = baseline[valid].dt.normalize().values.astype('datetime64[D]')
        a = actual[valid].dt.normalize().values.astype('datetime64[D]')
        arr[valid] = np.busday_count(a, b, holidays=US_HOL_2024)
    return arr

# ---------------------------------------------------
#  Main analysis: filter then compute summary
# ---------------------------------------------------
def analyze(df: pd.DataFrame, types: list[str], min_fans: int) -> pd.DataFrame | None:
    # Choose baseline: needed_date for critical, otherwise Baseline
    df = df.copy()
    df['Calc_Baseline'] = np.where(df['critical'] == 'Y', df['needed_date'], df['Baseline'])

    # Filter PNOC type
    mask = pd.Series(False, index=df.index)
    if 'Critical' in types:
        mask |= df['critical'] == 'Y'
    if 'Routine' in types:
        mask |= df['critical'] != 'Y'
    df = df[mask]

    # Filter FAN reviews
    df = df[df['Total FAN Reviews'] >= min_fans]
    if df.empty:
        return None

    # Compute variance
    df['Variance'] = schedule_variance(df['Calc_Baseline'], df['Actual'])
    df = df.dropna(subset=['Variance'])
    if df.empty:
        return None

    # Mean per PNOC
    mean_var = df.groupby('PNOC ID')['Variance'].mean()

    # Build summary DataFrame
    summary = mean_var.rename('Variance (Bus Days)').to_frame().reset_index()
    # Bucket status
    def bucket(v):
        if v >= 1:
            return 'Ahead'
        if v >= -1:
            return 'On Time'
        return 'Delayed'
    summary['Bucket'] = summary['Variance (Bus Days)'].apply(bucket)

    return summary

# ---------------------------------------------------
#  Streamlit UI
# ---------------------------------------------------
def main():
    st.set_page_config(page_title='Advanced Coordinated Analysis', layout='wide')
    st.title('Advanced Coordinated Analysis (ACA)')

    # Sidebar: upload
    st.sidebar.header('1. Upload your Excel')
    upload = st.sidebar.file_uploader('Upload PNOCs .xlsm', type=['xls','xlsx','xlsm'])
    if not upload:
        st.sidebar.info('Upload your Excel to begin')
        return

    df = load_and_clean(upload.read())

    # Sidebar: filters
    st.sidebar.header('2. Filter PNOC Type')
    types = st.sidebar.multiselect('PNOC Type', ['Routine','Critical'], default=['Routine','Critical'])
    st.sidebar.header('3. FAN review cycles (≥4)')
    min_fans = st.sidebar.number_input('Min FAN reviews', min_value=4, value=4, step=1)

    if st.sidebar.button('Run Analysis'):
        summary = analyze(df, types, min_fans)
        if summary is None or summary.empty:
            st.warning('No data after filters — try different selections or check your data.')
            return

        # Color scale with diverging scheme
        max_abs = max(1, abs(summary['Variance (Bus Days)']).max())
        import altair as alt
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('PNOC ID:N', sort='-y', title='PNOC ID'),
            y=alt.Y('Variance (Bus Days):Q', title='Schedule Variance (Bus Days)'),
            color=alt.Color('Variance (Bus Days):Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
            tooltip=['PNOC ID','Variance (Bus Days)','Bucket']
        ).properties(height=400)

        st.subheader('Schedule Variance ▶️ Actual vs Baseline (Business Days)')
        st.altair_chart(chart, use_container_width=True)
        st.subheader('Detailed Summary Table')
        st.dataframe(summary.set_index('PNOC ID'))

if __name__ == '__main__':
    main()



