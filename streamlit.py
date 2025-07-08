import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# ---------------------------------------------------
#  Load & clean data
# ---------------------------------------------------
@st.cache_data
def load_data(excel_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    # Main schedule sheet
    df_main = pd.read_excel(xls, sheet_name="Baseline + Actual Dates")
    # Critical flag
    df_crit = pd.read_excel(xls, sheet_name="Need date and critical status")
    # Comments/Resolution
    df_cirm = pd.read_excel(xls, sheet_name="CIRM")
    # Group classification
    df_group = pd.read_excel(xls, sheet_name="Group + Days Passed")

    # Standardize PNOC ID
    for df in (df_main, df_crit, df_cirm, df_group):
        df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # Parse dates
    df_main['Baseline'] = pd.to_datetime(df_main['Baseline'], format="%m/%d/%Y", errors="coerce")
    df_main['Actual']   = pd.to_datetime(df_main['Actual'],   format="%m/%d/%Y", errors="coerce")

    # Merge all into one table
    # Prepare CIRM sheet, ensure Contractor col exists
    df_cirm = df_cirm.copy()
    if 'Contractor' not in df_cirm.columns:
        df_cirm['Contractor'] = ''

    # Merge all into one table
    df = (
        df_main
        .merge(df_crit[['PNOC ID','critical']], on='PNOC ID', how='left')
        .merge(df_cirm[['PNOC ID','CI','RM','Contractor']], on='PNOC ID', how='left')
        .merge(df_group[['PNOC ID','Group']], on='PNOC ID', how='left')
    )
[['PNOC ID','Group']], on='PNOC ID', how='left')
    )

    # Fill NA in numeric
    df['CI'] = df['CI'].fillna(0)
    df['RM'] = df['RM'].fillna(0)
    df['Total FAN Reviews'] = 4 + df['RM'].apply(lambda x: 1 if x>0 else 0) + df['CI'].apply(lambda x: max(0, x-1))

    return df

# ---------------------------------------------------
#  Business-day variance: Actual -> Baseline (busdays)
# ---------------------------------------------------
US_HOL_2024 = np.array([
    datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4), datetime(2024,9,2), datetime(2024,10,14),
    datetime(2024,11,11), datetime(2024,11,28), datetime(2024,12,25)
], dtype='datetime64[D]')

def bus_variance(baseline: pd.Series, actual: pd.Series) -> np.ndarray:
    valid = baseline.notna() & actual.notna()
    var = np.full(len(baseline), np.nan)
    if valid.any():
        b = baseline[valid].dt.normalize().values.astype('datetime64[D]')
        a = actual[valid].dt.normalize().values.astype('datetime64[D]')
        var[valid] = np.busday_count(a, b, holidays=US_HOL_2024)
    return var

# ---------------------------------------------------
#  KPI Pages
# ---------------------------------------------------
def page_home():
    st.image('logo.png', width=600)
    st.markdown("""
    **Advanced Coordinated Analysis (ACA)** is your one-stop tool for tracking PNOC performance across three KPI dashboards:
    1. **Baseline Schedule Analysis (BSA)** – monitor schedule variance vs. baseline
    2. **Phase-based Average Delay** – track per-phase delays (coming soon)
    3. **Comment Resolution Time** – assess comment & resolution responsiveness (coming soon)
    """)
    col1, col2, col3 = st.columns(3)
    if col1.button('🗓️ Baseline Schedule Analysis'):
        st.session_state.page = 'bsa'
    if col2.button('⏳ Phase-based Average Delay'):
        st.warning('Phase-based Delay KPI coming soon!')
    if col3.button('✉️ Comment Resolution Time'):
        st.warning('Comment Resolution KPI coming soon!')


def page_bsa(df: pd.DataFrame):
    st.header('Baseline Schedule Analysis')
    c1, c2 = st.columns([3,1])
    with c2:
        if st.button('⬅️ Back'): st.session_state.page = 'home'
    # Filters
    st.subheader('Filters')
    g_opts = sorted(df['Group'].dropna().unique())
    sel_group = st.multiselect('Contractor Group', g_opts, default=g_opts)
    t_opts = ['Routine','Critical']
    sel_type = st.multiselect('PNOC Type', t_opts, default=t_opts)
    ci = st.number_input('Min Comment Issues (CI)', min_value=0, step=1, value=0)
    rm = st.number_input('Min Resolution Messages (RM)', min_value=0, step=1, value=0)
    contractors = sorted(df['Contractor'].dropna().unique())
    sel_con = st.multiselect('Contractors', contractors, default=contractors)
    over120 = st.selectbox('120-day Issuance Filter', ['All','>120 days','≤120 days'])

    # Apply filters
    d = df[df['Group'].isin(sel_group)]
    d = d[d['critical'].map(lambda x: ('Critical' in sel_type) if x=='Y' else ('Routine' in sel_type))]
    d = d[d['CI']>=ci]
    d = d[d['RM']>=rm]
    d = d[d['Contractor'].isin(sel_con)]
    # Compute variance
    d['Variance'] = bus_variance(d['Baseline'], d['Actual'])
    # Average per PNOC
    summary = (
        d.groupby('PNOC ID')['Variance']
         .mean()
         .rename('Variance (Bus Days)')
         .reset_index()
    )
    # 120-day filter
    if over120=='>120 days': summary = summary[summary['Variance']<=-120]
    if over120=='≤120 days': summary = summary[summary['Variance']>-120]
    if summary.empty:
        st.warning('No PNOCs match these filters')
        return
    # Bucket
    def bucket(v):
        if v>=1: return 'Ahead'
        if v>=-1: return 'On Time'
        return 'Delayed'
    summary['Bucket'] = summary['Variance'].apply(bucket)

    # Chart
    max_abs = max(1, abs(summary['Variance']).max())
    import altair as alt
    chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X('PNOC ID:N', sort='-y'),
        y=alt.Y('Variance:Q', title='Variance (Bus Days)'),
        color=alt.Color('Variance:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
        tooltip=['PNOC ID','Variance','Bucket']
    ).properties(width=800, height=400)
    st.altair_chart(chart)
    st.dataframe(summary.set_index('PNOC ID'))

# ---------------------------------------------------
#  Main app
# ---------------------------------------------------
def main():
    st.set_page_config(page_title='ACA', layout='wide')
    if 'page' not in st.session_state: st.session_state.page = 'home'
    st.sidebar.title('ACA Navigation')
    nav = st.sidebar.radio('', ['Home','Baseline Schedule Analysis'])
    st.session_state.page = 'home' if nav=='Home' else 'bsa'
    upload = None
    if st.session_state.page=='home':
        page_home()
    else:
        if upload is None:
            upload = st.file_uploader('Upload PNOCs .xlsm', type=['xls','xlsx','xlsm'])
        if upload:
            df = load_data(upload.read())
            page_bsa(df)
        else:
            st.sidebar.info('Please upload your file to continue.')

if __name__=='__main__':
    main()






