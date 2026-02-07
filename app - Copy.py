import pandas as pd
import yfinance as yf
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import io
import os

# ------------------------------
# App Initialization
# ------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), 'watchlist.csv')
HIST_CACHE = {}

# ------------------------------
# Utility Functions
# ------------------------------
def format_currency(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
    return df

def flatten_yf_data(data):
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [col[0] for col in data.columns]

    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    cols_to_keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
    data = data[cols_to_keep]
    for col in cols_to_keep:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

# ------------------------------
# Compute Signals
# ------------------------------
def compute_signals(symbols):
    results = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)

            # Use cache for historical data
            if symbol not in HIST_CACHE:
                hist = ticker.history(period="2y")
                hist = flatten_yf_data(hist)
                hist['9MA'] = hist['Close'].rolling(9).mean()
                hist['45MA'] = hist['Close'].rolling(45).mean()
                hist['200MA'] = hist['Close'].rolling(200).mean()
                HIST_CACHE[symbol] = hist
            else:
                hist = HIST_CACHE[symbol]

            latest = hist.iloc[-1]
            price = ticker.info.get('regularMarketPrice', latest['Close'])
            ma9, ma45, ma200 = latest['9MA'], latest['45MA'], latest['200MA']

            if price > ma9 and price > ma200 and ma9 > ma45:
                signal, icon = 'BUY', '⬆'
            elif price < ma45 or ma9 < ma45:
                signal, icon = 'SELL', '⬇'
            else:
                signal, icon = 'HOLD', '➖'

            results.append({
                'Symbol': symbol,
                'Price': price,
                '9MA': ma9,
                '45MA': ma45,
                '200MA': ma200,
                'Signal': signal,
                'Signal_Icon': icon
            })

        except Exception as e:
            print(f"Failed {symbol}: {e}")

    df = pd.DataFrame(results)
    return format_currency(df, ['Price','9MA','45MA','200MA'])

# ------------------------------
# Backtest
# ------------------------------
def backtest(symbol, start=None, end=None):
    data = yf.download(symbol, period="5y", progress=False)
    data = flatten_yf_data(data)
    data['9MA'] = data['Close'].rolling(9).mean()
    data['45MA'] = data['Close'].rolling(45).mean()
    data['200MA'] = data['Close'].rolling(200).mean()

    signals, icons = [], []
    for _, row in data.iterrows():
        if row['Close'] > row['9MA'] and row['Close'] > row['200MA'] and row['9MA'] > row['45MA']:
            signals.append('BUY'); icons.append('⬆')
        elif row['Close'] < row['45MA'] or row['9MA'] < row['45MA']:
            signals.append('SELL'); icons.append('⬇')
        else:
            signals.append('HOLD'); icons.append('➖')

    data['Signal'] = signals
    data['Signal_Icon'] = icons
    data = data.reset_index().rename(columns={'index':'Date'})
    data['Date_dt'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date_dt', ascending=False)
    data['Date'] = data['Date_dt'].dt.strftime('%m-%d-%Y')
    data = data.drop(columns=['Date_dt'])
    for col in ['Open','High','Low','Close','9MA','45MA','200MA']:
        if col in data.columns:
            data[col] = data[col].round(2)
    data = format_currency(data, ['Open','High','Low','Close','9MA','45MA','200MA'])
    if start and end:
        start_date, end_date = pd.to_datetime(start), pd.to_datetime(end)
        data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]
    return data[['Date','200MA','45MA','9MA','Open','High','Low','Close','Signal','Signal_Icon']]

# ------------------------------
# Excel download helper
# ------------------------------
def excel_download(table_data, filename):
    if not table_data:
        return None
    df = pd.DataFrame(table_data)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=filename.replace(".xlsx",""))
    buffer.seek(0)
    return dcc.send_bytes(buffer.read(), filename)

# ------------------------------
# Layout
# ------------------------------
app.layout = dbc.Container([
    html.H2("📈 Stock Signal Dashboard"),
    dcc.Interval(id='interval-update', interval=5*1000, n_intervals=0),

    dcc.Tabs([
        dcc.Tab(label="Dashboard", children=[
            html.Br(),
            html.H5("Stock Signals (Manual Input)"),
            dcc.Input(id='symbol-input', placeholder='Enter symbols comma separated', debounce=True, style={'width':'50%'}),
            html.Br(), html.Br(),
            dbc.Button("Download Signals Excel", id='download-btn', color='primary'),
            dcc.Download(id="download-dataframe-xlsx"),
            html.Br(), html.Br(),
            dash_table.DataTable(
                id='stock-table',
                style_header={'textAlign':'center','fontWeight':'bold'},
                style_table={'overflowX':'auto'},
                columns=[
                    {"name":"Symbol","id":"Symbol"},
                    {"name":"Price","id":"Price"},
                    {"name":"9MA","id":"9MA"},
                    {"name":"45MA","id":"45MA"},
                    {"name":"200MA","id":"200MA"},
                    {"name":"Signal","id":"Signal"},
                    {"name":"Signal_Icon","id":"Signal_Icon"}
                ]
            ),
            html.Br(),
            html.H5("Watchlist (CSV)"),
            dbc.Button("Download Watchlist Excel", id='download-watchlist-btn', color='secondary'),
            dcc.Download(id="download-watchlist-xlsx"),
            html.Br(),
            dash_table.DataTable(
                id='watchlist-table',
                style_header={'textAlign':'center','fontWeight':'bold'},
                style_table={'overflowX':'auto'},
                columns=[
                    {"name":"Symbol","id":"Symbol"},
                    {"name":"Price","id":"Price"},
                    {"name":"9MA","id":"9MA"},
                    {"name":"45MA","id":"45MA"},
                    {"name":"200MA","id":"200MA"},
                    {"name":"Signal","id":"Signal"},
                    {"name":"Signal_Icon","id":"Signal_Icon"}
                ]
            )
        ]),
        dcc.Tab(label="Backtest", children=[
            html.Br(),
            html.H5("Backtest (Enter US ticker)"),
            dcc.Input(id='backtest-symbol', placeholder='Enter US ticker', debounce=True, style={'width':'50%'}),
            html.Br(), html.Br(),
            dcc.DatePickerRange(id='date-range'),
            html.Br(), html.Br(),
            dbc.Button("Download Backtest Excel", id="download-backtest-btn", color='secondary', className="mb-2"),
            dcc.Download(id="download-backtest-xlsx"),
            dash_table.DataTable(
                id='backtest-table',
                style_header={'textAlign':'center','fontWeight':'bold'},
                style_table={'overflowX':'auto'}
            )
        ])
    ])
], fluid=True)

# ------------------------------
# Callbacks
# ------------------------------

# Stock-table (manual symbols + refresh)
@app.callback(
    Output('stock-table','data'),
    Output('stock-table','style_data_conditional'),
    Input('symbol-input','value'),
    Input('interval-update','n_intervals')
)
def update_stock_table(symbols, n_intervals):
    if not symbols:
        return [], []
    symbols = [s.strip().upper() for s in symbols.split(',')]
    df = compute_signals(symbols)
    data = df.to_dict('records')

    style_conditional = [
        {'if':{'column_id':'Symbol'}, 'textAlign':'left'},
        {'if':{'column_id':'Signal_Icon'}, 'textAlign':'center', 'fontSize':'20px'},
        {'if':{'column_id':'Signal'}, 'textAlign':'center'}
    ]
    for col in ['Price','9MA','45MA','200MA']:
        style_conditional.append({'if':{'column_id':col}, 'textAlign':'right'})
    style_conditional += [
        {'if': {'filter_query':'{Signal} = "BUY"'}, 'backgroundColor':'#d4edda'},
        {'if': {'filter_query':'{Signal} = "SELL"'}, 'backgroundColor':'#f8d7da'},
        {'if': {'filter_query':'{Signal} = "HOLD"'}, 'backgroundColor':'#fff3cd'}
    ]
    return data, style_conditional

# Watchlist-table (CSV auto-refresh)
@app.callback(
    Output('watchlist-table','data'),
    Output('watchlist-table','style_data_conditional'),
    Input('interval-update','n_intervals')
)
def update_watchlist_table(n_intervals):
    if not os.path.exists(WATCHLIST_FILE):
        return [], []
    df_csv = pd.read_csv(WATCHLIST_FILE, header=None)
    symbols = df_csv.iloc[:,0].dropna().tolist()
    if not symbols:
        return [], []
    df = compute_signals(symbols)
    data = df.to_dict('records')

    style_conditional = [
        {'if':{'column_id':'Symbol'}, 'textAlign':'left'},
        {'if':{'column_id':'Signal_Icon'}, 'textAlign':'center', 'fontSize':'20px'},
        {'if':{'column_id':'Signal'}, 'textAlign':'center'}
    ]
    for col in ['Price','9MA','45MA','200MA']:
        style_conditional.append({'if':{'column_id':col}, 'textAlign':'right'})
    style_conditional += [
        {'if': {'filter_query':'{Signal} = "BUY"'}, 'backgroundColor':'#d4edda'},
        {'if': {'filter_query':'{Signal} = "SELL"'}, 'backgroundColor':'#f8d7da'},
        {'if': {'filter_query':'{Signal} = "HOLD"'}, 'backgroundColor':'#fff3cd'}
    ]
    return data, style_conditional

# Backtest-table
@app.callback(
    Output('backtest-table','data'),
    Output('backtest-table','style_data_conditional'),
    Input('backtest-symbol','value'),
    Input('date-range','start_date'),
    Input('date-range','end_date')
)
def update_backtest(symbol_input,start,end):
    if not symbol_input:
        return [], []
    df = backtest(symbol_input.strip().upper(), start, end)
    data = df.to_dict('records')

    style_conditional = [
        {'if':{'column_id':'Signal_Icon'}, 'textAlign':'center','fontSize':'20px'},
        {'if':{'column_id':'Signal'}, 'textAlign':'center'}
    ]
    for col in ['Open','High','Low','Close','9MA','45MA','200MA']:
        style_conditional.append({'if':{'column_id':col}, 'textAlign':'right'})
    style_conditional += [
        {'if': {'filter_query':'{Signal} = "BUY"'}, 'backgroundColor':'#d4edda'},
        {'if': {'filter_query':'{Signal} = "SELL"'}, 'backgroundColor':'#f8d7da'},
        {'if': {'filter_query':'{Signal} = "HOLD"'}, 'backgroundColor':'#fff3cd'}
    ]
    return data, style_conditional

# Download Excel callbacks
@app.callback(Output("download-dataframe-xlsx","data"),
              Input("download-btn","n_clicks"),
              State('stock-table','data'),
              prevent_initial_call=True)
def download_stock_signals(n,data):
    return excel_download(data,"StockSignals.xlsx")

@app.callback(Output("download-watchlist-xlsx","data"),
              Input("download-watchlist-btn","n_clicks"),
              State('watchlist-table','data'),
              prevent_initial_call=True)
def download_watchlist(n,data):
    return excel_download(data,"Watchlist.xlsx")

@app.callback(Output("download-backtest-xlsx","data"),
              Input("download-backtest-btn","n_clicks"),
              State('backtest-table','data'),
              prevent_initial_call=True)
def download_backtest(n,data):
    return excel_download(data,"Backtest.xlsx")

# ------------------------------
# Run App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
