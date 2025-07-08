import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

############################################################
# Robust utility helpers
############################################################
REQUIRED_SHEETS = {
    'main': "Baseline + Actual Dates",
    'crit': "Need date and critical status",
    'cirm': "CIRM",
    'group': "Group + Days Passed",
}

US_HOL_2024 = np.array([
    datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4), datetime(2024,9,2), datetime(2024,10,14),
    datetime(2024,11,11), datetime(2024,11,28), datetime(2024,12,25)
], dtype="datetime64[D]")

############################################################
# Data‑loading (safe)
############################################################
@st.cache_data(show_spinner=False)
def load_data(excel_bytes: bytes) -> pd.DataFrame | None:
    """Return a merged DataFrame or None if unrecoverable error."""
    try:
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    except Exception as exc:
        st.error(f"Unable to read Excel file: {exc}")
        return None

    # Helper to read a sheet safely
    def safe_read(name: str) -> pd.DataFrame:
        try:
            return pd.read_excel(xls, sheet_name=name)
        except ValueError:
            st.warning(f"Sheet “{name}” not found – using empty frame.")
            return pd.DataFrame()

    df_main  = safe_read(REQUIRED_SHEETS['main'])
    df_crit  = safe_read(REQUIRED_SHEETS['crit'])
    df_cirm  = safe_read(REQUIRED_SHEETS['cirm'])
    df_group = safe_read(REQUIRED_SHEETS['group'])

    # Guarantee PNOC ID col exists
    for df in (df_main, df_crit, df_cirm, df_group):
        if 'PNOC ID' not in df.columns:
            df['PNOC ID'] = []
        df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # Date parsing
    for col in ('Baseline','Actual'):
        if col in df_main.columns:
            df_main[col] = pd.to_datetime(df_main[col], errors='coerce')
        else:
            df_main[col] = pd.NaT

    # Ensure critical column
    if 'critical' not in df_crit.columns:
        df_crit['critical'] = np.nan

    # Ensure CI / RM / Contractor cols
    for col in ('CI','RM','Contractor'):
        if col not in df_cirm.columns:
            df_cirm[col] = 0 if col in ('CI','RM') else ''

    # Ensure Group col
    if 'Group' not in df_group.columns:
        df_group['Group'] = ''

    # Merge
    df = (
        df_main
        .merge(df_crit[['PNOC ID','critical']], on='PNOC ID', how='left')
        .merge(df_cirm[['PNOC ID','CI','RM','Contractor']], on='PNOC ID', how='left')
        .merge(df_group[['PNOC ID','Group']], on='PNOC ID', how='left')
    )

        # Fill NA & enforce numeric
    df['CI'] = pd.to_numeric(df['CI'], errors='coerce').fillna(0).astype(int)
    df['RM'] = pd.to_numeric(df['RM'], errors='coerce').fillna(0).astype(int)
    df['critical']   = df['critical'].fillna('N')
    df['Group']      = df['Group'].fillna('')
    df['Contractor'] = df['Contractor'].fillna('')

    # Compute FAN reviews
    df['Total FAN Reviews'] = 4 + df['RM'].apply(lambda x: 1 if x>0 else 0) + df['CI'].apply(lambda x: max(0, x-1))

    return df

############################################################
# Business‑day variance
############################################################
def bus_variance(baseline: pd.Series, actual: pd.Series) -> np.ndarray:
    valid = baseline.notna() & actual.notna()
    out = np.full(len(baseline), np.nan)
    if valid.any():
        b = baseline[valid].dt.normalize().values.astype('datetime64[D]')
        a = actual[valid].dt.normalize().values.astype('datetime64[D]')
        out[valid] = np.busday_count(a, b, holidays=US_HOL_2024)
    return out

############################################################
# Page: Home
############################################################

def page_home():
    st.image('logo.png', width=600)
    st.markdown("""
    ### Advanced Coordinated Analysis (ACA)
    Monitor PNOC performance across three KPI dashboards:
    1. **Baseline Schedule Analysis (BSA)** – schedule variance
    2. **Phase‑based Average Delay** – coming soon
    3. **Comment Resolution Time** – coming soon
    """)
    c1, c2, c3 = st.columns(3)
    if c1.button('🗓️ Baseline Schedule Analysis'):
        st.session_state.page = 'bsa'
    if c2.button('⏳ Phase‑based Average Delay'):
        st.info('Coming soon!')
    if c3.button('✉️ Comment Resolution Time'):
        st.info('Coming soon!')

############################################################
# Page: BSA
############################################################

def page_bsa(df: pd.DataFrame):
    st.header('Baseline Schedule Analysis')
    if st.button('⬅️ Home'): st.session_state.page = 'home'; return

    # ---------- Filters ----------
    st.subheader('Filter criteria')
    sel_groups = st.multiselect('Group', sorted(df['Group'].unique()), default=list(df['Group'].unique()))
    sel_types  = st.multiselect('PNOC Type', ['Routine','Critical'], default=['Routine','Critical'])
    min_ci     = st.number_input('Min Comment Issues (CI)', 0, step=1)
    min_rm     = st.number_input('Min Resolution Messages (RM)', 0, step=1)
    sel_con    = st.multiselect('Contractor(s)', sorted(df['Contractor'].unique()), default=list(df['Contractor'].unique()))
    over120opt = st.selectbox('120‑day Issuance Filter', ['All','>120 days','≤120 days'])

    # ---------- Apply filters ----------
    d = df.copy()
    if sel_groups:    d = d[d['Group'].isin(sel_groups)]
    if sel_types:
        d = d[d['critical'].apply(lambda x: ('Critical' in sel_types) if x=='Y' else ('Routine' in sel_types))]
    d = d[(d['CI']>=min_ci) & (d['RM']>=min_rm)]
    if sel_con:       d = d[d['Contractor'].isin(sel_con)]

    # ---------- Variance ----------
    d['Variance'] = bus_variance(d['Baseline'], d['Actual'])
    summary = d.groupby('PNOC ID')['Variance'].mean().dropna().rename('Variance').to_frame().reset_index()

    if over120opt=='>120 days': summary = summary[summary['Variance']<=-120]
    if over120opt=='≤120 days': summary = summary[summary['Variance']>-120]

    if summary.empty:
        st.warning('No PNOCs match your filters.')
        return

    # Bucket & chart
    summary['Bucket'] = summary['Variance'].apply(lambda v: 'Ahead' if v>=1 else ('On Time' if v>=-1 else 'Delayed'))
    import altair as alt
    chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X('PNOC ID:N', sort='-y'),
        y=alt.Y('Variance:Q', title='Variance (Bus Days)'),
        color=alt.Color('Variance:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0)),
        tooltip=['PNOC ID','Variance','Bucket']
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(summary.set_index('PNOC ID'))

############################################################
# Main app logic
############################################################

def main():
    st.set_page_config(page_title='ACA', layout='wide')
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Sidebar navigation
    page_choice = st.sidebar.radio('Navigation', ['Home','Baseline Schedule Analysis'])
    st.session_state.page = 'home' if page_choice=='Home' else 'bsa'

    if st.session_state.page=='home':
        page_home()
    else:
        upl = st.sidebar.file_uploader('Upload PNOCs Excel (.xls/.xlsx/.xlsm)', type=['xls','xlsx','xlsm'])
        if not upl:
            st.info('Please upload your Excel file to continue.')
            return
        df = load_data(upl.read())
        if df is None or df.empty:
            st.error('No valid data loaded. Check your file and sheet names.')
            return
        page_bsa(df)

if __name__ == '__main__':
    main()



