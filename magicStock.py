"""Streamlit Magic Formula screener backed by Turso financial_statements."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from russell import Turso, load_env


def credentials():
    load_env(Path(__file__).with_name('.env'))
    try:
        secrets = dict(st.secrets)
    except FileNotFoundError:
        secrets = {}
    return tuple(secrets.get(k) or os.getenv(k, '') for k in
                 ('TURSO_DATABASE_URL', 'TURSO_AUTH_TOKEN'))


@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
def statements(url, _token, industries=(), sectors=()):
    conditions, args = [], []
    for column, selected in [('industry', industries), ('sector', sectors)]:
        if selected:
            conditions.append(f'{column} IN ({",".join("?" for _ in selected)})')
            args.extend(selected)
    where = (' WHERE Ticker IN (SELECT ticker FROM russell3000 WHERE '
             + ' AND '.join(conditions) + ')') if conditions else ''
    # Transfer only columns consumed by the screener; preserve all required periods.
    fields = ('Ticker, Date, Period, OperatingIncome, EBIT, NetWorkingCapital, '
              'NetPPE, EV, EPS, ROE, TotalDebt, Cash, MarketCap, Price, Currency')
    result = Turso(url, _token).execute(
        f'SELECT {fields} FROM financial_statements{where} ORDER BY Ticker, Period, Date DESC', args)
    columns = [c['name'] for c in result['cols']]
    rows = [[None if cell['type'] == 'null' else cell.get('value')
             for cell in row] for row in result['rows']]
    frame = pd.DataFrame(rows, columns=columns)
    required = {'Ticker', 'Date', 'Period', 'OperatingIncome', 'EBIT',
                'NetWorkingCapital', 'NetPPE', 'EV', 'EPS', 'ROE',
                'TotalDebt', 'Cash', 'MarketCap', 'Price', 'Currency'}
    if not required.issubset(frame.columns):
        raise ValueError('financial_statements에 필요한 열이 없습니다: '
                         + ', '.join(sorted(required - set(frame.columns))))
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    for column in required - {'Ticker', 'Date', 'Period', 'Currency'}:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.dropna(subset=['Ticker', 'Date'])


@st.cache_data(ttl=900, show_spinner=False)
def companies(url, _token):
    result = Turso(url, _token).execute(
        'SELECT ticker AS "Ticker", "tickerName" AS "Ticker Name", '
        'industry AS "Industry", sector AS "Sector" FROM russell3000 ORDER BY ticker')
    return pd.DataFrame(
        [[None if cell['type'] == 'null' else cell.get('value') for cell in row]
         for row in result['rows']], columns=[c['name'] for c in result['cols']])


def yahoo_symbol(ticker):
    return ticker.replace('.', '-')


@st.cache_data(ttl=300, max_entries=6000, show_spinner=False)
def quote(ticker):
    try:
        info = yf.Ticker(yahoo_symbol(ticker)).fast_info
        price = float(info['last_price'])
        try:
            shares = float(info['shares'])
        except Exception:
            shares = np.nan
        return {'price': price if np.isfinite(price) and price > 0 else np.nan,
                'shares': shares, 'currency': info['currency']}
    except Exception:
        return {'price': np.nan, 'shares': np.nan, 'currency': None}


def quotes(tickers):
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=min(6, len(tickers))) as pool:
        return dict(zip(tickers, pool.map(quote, tickers)))


def ratio(a, b):
    return a / b * 100 if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def growth(current, previous):
    # Zero/negative base EPS has no comparable conventional growth rate.
    return (current / previous - 1) * 100 if pd.notna(current) and pd.notna(previous) and previous > 0 else np.nan


def previous_period(rows, row, yearly=False):
    days = (row['Date'] - rows['Date']).dt.days
    eligible = rows.loc[days.between(300, 430) if yearly else days.between(60, 120)]
    if eligible.empty:
        return None
    target = 365 if yearly else 91
    return eligible.loc[(days.loc[eligible.index] - target).abs().idxmin()]


def profit(rows, row, quarterly):
    periods = [row]
    if quarterly:
        for _ in range(3):
            earlier = previous_period(rows, periods[-1])
            if earlier is None:
                return np.nan
            periods.append(earlier)
    values = [r['OperatingIncome'] if pd.notna(r['OperatingIncome']) else r['EBIT'] for r in periods]
    return sum(values) if all(pd.notna(v) for v in values) else np.nan


@st.cache_data(ttl=900, max_entries=16, show_spinner=False)
def build_results(frame, market, company_data=None):
    records = []
    for ticker, group in frame.groupby('Ticker'):
        quote = market.get(ticker, {})
        record = {'Ticker': ticker, 'Current Price': quote.get('price', np.nan)}
        for period, label in [('quarterly', 'Q'), ('yearly', 'Y')]:
            rows = group[group['Period'] == period].sort_values('Date', ascending=False)
            recent = []
            if not rows.empty:
                recent.append(rows.iloc[0])
                for _ in range(2):
                    earlier = previous_period(rows, recent[-1], yearly=label == 'Y')
                    if earlier is None:
                        break
                    recent.append(earlier)
            for offset in range(3):
                suffix = label if offset == 0 else f'{label}-{offset}'
                roc = ey = eps_growth = np.nan
                if offset < len(recent):
                    row = recent[offset]
                    earnings = profit(rows, row, label == 'Q')
                    roc = ratio(earnings, row['NetWorkingCapital'] + row['NetPPE'])
                    ev = row['EV']
                    if label == 'Q' and offset == 0:
                        shares = quote.get('shares', np.nan)
                        if pd.isna(shares) and pd.notna(row['Price']) and row['Price'] > 0:
                            shares = row['MarketCap'] / row['Price']
                        ev = record['Current Price'] * shares + row['TotalDebt'] - row['Cash']
                        if not row['Currency'] or quote.get('currency') != row['Currency']:
                            ev = np.nan
                    ey = ratio(earnings, ev)
                    earlier = previous_period(rows, row, yearly=True)
                    if earlier is not None:
                        eps_growth = growth(row['EPS'], earlier['EPS'])
                record[f'ROC({suffix})'] = roc
                record[f'EY({suffix})'] = ey
                if label == 'Y' or offset == 0:
                    record[f'EPS 증가율({suffix})'] = eps_growth
            record[f'ROE({label})'] = recent[0]['ROE'] * 100 if recent else np.nan
        records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    if company_data is None:
        company_data = pd.DataFrame(columns=['Ticker', 'Ticker Name', 'Industry', 'Sector'])
    result = result.merge(company_data, on='Ticker', how='left', validate='many_to_one')
    valid = result[['ROC(Q)', 'EY(Q)', 'Current Price']].notna().all(axis=1)
    eligible = result.loc[valid]
    scores = eligible['ROC(Q)'].rank(ascending=False, method='min') + eligible['EY(Q)'].rank(ascending=False, method='min')
    result['Total Rank'] = scores.rank(method='min').reindex(result.index).astype('Int64')
    columns = ['Ticker', 'Ticker Name', 'Industry', 'Sector', 'Current Price', 'ROC(Q)', 'ROC(Q-1)', 'ROC(Q-2)',
               'EY(Q)', 'EY(Q-1)', 'EY(Q-2)', 'EPS 증가율(Q)',
               'ROC(Y)', 'ROC(Y-1)', 'ROC(Y-2)', 'EY(Y)', 'EY(Y-1)', 'EY(Y-2)',
               'EPS 증가율(Y)', 'EPS 증가율(Y-1)', 'EPS 증가율(Y-2)',
               'ROE(Y)', 'ROE(Q)', 'Total Rank']
    return result[columns].sort_values(['Total Rank', 'Ticker'], na_position='last').reset_index(drop=True)


def filter_results(result, filters):
    result = result.copy()
    for name, value in filters.items():
        if name in {'Industry', 'Sector'}:
            if value:
                result = result[result[name].isin(value)]
        elif name == 'Total Rank':
            result = result[result[name] <= value]
        elif name == 'EPS 증가율(Y)':
            result = result[(result[['EPS 증가율(Y)', 'EPS 증가율(Y-1)', 'EPS 증가율(Y-2)']] > value).all(axis=1)]
        else:
            result = result[result[name] > value]
    return result.reset_index(drop=True)


def chart(ticker):
    # Called only on selection; never download all tickers' one-year histories.
    history = yf.Ticker(yahoo_symbol(ticker)).history(period='1y', auto_adjust=False, actions=False)
    if history.empty:
        st.warning('최근 1년 주가를 가져오지 못했습니다.')
        return
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                           row_heights=[0.75, 0.25])
    figure.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'],
                                   low=history['Low'], close=history['Close'], name='주가'), row=1, col=1)
    volume_change = history['Volume'].diff()
    volume_colors = np.select(
        [volume_change > 0, volume_change < 0],
        ['#ef4444', '#3b82f6'], default='#9ca3af')
    figure.add_trace(go.Bar(x=history.index, y=history['Volume'], name='거래량',
                           marker_color=volume_colors), row=2, col=1)
    figure.update_layout(title=f'{ticker} · 최근 1년 주가 / 거래량', height=650,
                         xaxis_rangeslider_visible=False, showlegend=False)
    figure.update_yaxes(title_text='주가', row=1, col=1)
    figure.update_yaxes(title_text='거래량', row=2, col=1)
    st.plotly_chart(figure, width='stretch')
    st.caption('거래량 색상: 전 거래일 대비 증가 = 빨강, 감소 = 파랑, 동일 / 비교값 없음 = 회색')


def main():
    st.set_page_config(page_title='Magic Stock', layout='wide')
    st.title('Magic Stock')
    if st.sidebar.button('데이터 / 현재가 캐시 초기화'):
        statements.clear()
        companies.clear()
        quote.clear()
        build_results.clear()
        st.session_state.pop('screen', None)
    try:
        url, token = credentials()
        if not url or not token:
            st.error('TURSO_DATABASE_URL / TURSO_AUTH_TOKEN을 Secrets 또는 환경변수에 설정하세요.')
            return
        company_data = companies(url, token)
    except Exception as error:
        st.error(f'종목 분류 조회 실패 ({type(error).__name__}). russell3000 테이블과 DB 연결을 확인하세요.')
        return
    st.caption('Turso 재무제표 · Yahoo 최근 거래가격 · ROC / EY 순위')
    with st.sidebar:
        st.header('조회조건')
        filters = {}
        with st.form('filters'):
            for name in ['Industry', 'Sector']:
                options = sorted(v for v in company_data[name].dropna().unique() if str(v).strip())
                selected = st.multiselect(name, options,
                                         help='여러 항목을 선택할 수 있습니다. 미선택 시 전체를 조회합니다.')
                if not options:
                    st.caption(f'russell3000에 저장된 {name} 분류가 없습니다.')
                if selected:
                    filters[name] = selected
            for name, default in [('Total Rank', 100), ('EPS 증가율(Y)', 10.0), ('ROE(Y)', 10.0), ('ROE(Q)', 3.0)]:
                enabled = st.checkbox(f'{name} 적용', value=False)
                value = (st.number_input('Total Rank 상위 개수', min_value=1, value=default, step=1)
                         if name == 'Total Rank' else st.number_input(f'{name} 초과 (%)', value=default, step=0.5))
                if enabled:
                    filters[name] = value
            submitted = st.form_submit_button('조회', type='primary')
    if submitted:
        try:
            url, token = credentials()
            if not url or not token:
                st.error('TURSO_DATABASE_URL / TURSO_AUTH_TOKEN을 Secrets 또는 환경변수에 설정하세요.')
                return
            with st.spinner('재무제표와 현재가를 조회하고 있습니다. 최초 조회는 종목 수에 따라 수분 걸릴 수 있습니다.'):
                start = perf_counter()
                industries = tuple(sorted(filters.get('Industry', [])))
                sectors = tuple(sorted(filters.get('Sector', [])))
                frame = statements(url, token, industries, sectors)
                db_done = perf_counter()
                if frame.empty:
                    st.session_state.pop('screen', None)
                    st.info('financial_statements에 데이터가 없습니다.')
                    return
                market = quotes(tuple(sorted(frame['Ticker'].unique())))
                prices_done = perf_counter()
                st.session_state['screen'] = build_results(frame, market, company_data)
                calculated = perf_counter()
                st.session_state['timing'] = (db_done - start, prices_done - db_done,
                                            calculated - prices_done, len(frame))
                st.session_state['active_filters'] = filters
                st.session_state['screen_revision'] = st.session_state.get('screen_revision', 0) + 1
        except Exception as error:
            st.error(f'데이터 조회 실패 ({type(error).__name__}). DB 설정과 네트워크 연결을 확인하세요.')
            return
        st.rerun()
    if 'screen' not in st.session_state:
        st.info('조회조건을 선택하고 조회 버튼을 눌러 주세요.')
        return
    screen = st.session_state['screen']
    result = filter_results(screen, st.session_state.get('active_filters', {}))
    st.caption(f'선택 범위 {len(screen):,}종목 · 조회 {len(result):,}종목 · 순위 미산출 {screen["Total Rank"].isna().sum():,}종목 · Total Rank는 선택 범위 기준')
    if 'timing' in st.session_state:
        db_seconds, price_seconds, calculation_seconds, row_count = st.session_state['timing']
        st.caption(f'재무표 {row_count:,}행 · DB {db_seconds:.1f}초 · 현재가 {price_seconds:.1f}초 · 계산 {calculation_seconds:.1f}초 (캐시 사용 시 포함)')
    with st.expander('계산 기준 / 데이터 갱신'):
        st.markdown('''- ROC = 영업이익(없으면 EBIT) / (비현금 운전자본 + NetPPE). 분기는 최근 4분기 합계입니다.
- EY = 같은 영업이익 / EV. 당기 Q는 Yahoo 현재가 × 현재 주식 수 + DB 총부채 − DB 현금입니다. 주식 수 조회 실패 시 DB MarketCap / Price를 사용합니다.
- 과거 Q 및 Y의 EY는 해당 DB EV를 사용합니다. ROC는 주가와 무관합니다.
- Industry / Sector를 먼저 DB 조회에 적용하고 해당 종목의 현재가만 가져옵니다. Total Rank는 선택 범위 안에서 ROC(Q), EY(Q) 순위를 더해 매깁니다. 분류 미선택 시 전체 범위입니다. EPS / ROE / Rank 조건은 순위 계산 후 적용합니다.
- EPS 증가율은 전년 동기 대비입니다. 연간 조건은 최근 3년 모두 입력값 초과입니다. 전년 EPS가 0 이하이거나 데이터가 없으면 미산출합니다.
- ROE는 DB 비율 × 100이며 분기 ROE는 연율화하지 않습니다. 모든 비율은 %로 표시합니다.
- 재무 데이터는 15분, 현재가는 5분 캐시합니다. 현재가는 Yahoo 최근 거래가격이며 지연될 수 있습니다. 캐시 초기화 후 조회 버튼을 누르면 다시 가져옵니다.
- 통화가 다르거나 불명확하면 당기 EY 및 순위를 산출하지 않습니다. 결측치는 빈칸으로 표시합니다.''')
    if result.empty:
        st.info('선택한 조건을 만족하는 종목이 없습니다.')
        return
    config = {c: st.column_config.NumberColumn(c, format='%.1f%%') for c in result.columns
              if c not in {'Ticker', 'Ticker Name', 'Industry', 'Sector', 'Current Price', 'Total Rank'}}
    config['Current Price'] = st.column_config.NumberColumn('Current Price', format='%.2f')
    config['Total Rank'] = st.column_config.NumberColumn('Total Rank', format='%d')
    event = st.dataframe(result, hide_index=True, width='stretch', column_config=config,
                         on_select='rerun', selection_mode='single-row',
                         key=f'results_{st.session_state.get("screen_revision", 0)}')
    st.download_button('조회결과 CSV 다운로드', result.to_csv(index=False).encode('utf-8-sig'),
                       'magicStock.csv', 'text/csv')
    if event.selection.rows:
        ticker = result.iloc[event.selection.rows[0]]['Ticker']
        try:
            with st.spinner(f'{ticker} 최근 1년 주가 조회 중...'):
                chart(ticker)
        except Exception as error:
            st.warning(f'주가 차트 조회 실패 ({type(error).__name__}). 다시 선택해 주세요.')
    else:
        st.info('표의 종목 행을 선택하면 최근 1년 주가와 거래량 차트가 표시됩니다.')


if __name__ == '__main__':
    main()
