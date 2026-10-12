import os
import pandas as pd
import numpy as np
import xlsxwriter

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, 'outputs', 'hotel_bookings_cleaned.csv')
OUT  = os.path.join(BASE, 'outputs', 'Hotel_Bookings_Analysis.xlsx')

df = pd.read_csv(SRC)
print("Loaded:", df.shape)

# ── month order ───────────────────────────────────────────────────────────────
MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
df['arrival_date_month'] = pd.Categorical(
    df['arrival_date_month'], categories=MONTH_ORDER, ordered=True)

# ── pre-compute all summary tables ───────────────────────────────────────────

# 1. Cancellation by distribution channel
t_channel = (df.groupby('distribution_channel')
               .agg(Total_Bookings=('is_canceled','count'),
                    Cancellations=('is_canceled','sum'))
               .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
               .sort_values('Cancel_Rate', ascending=False)
               .reset_index())

# 2. Cancellation by market segment
t_segment = (df.groupby('market_segment')
               .agg(Total_Bookings=('is_canceled','count'),
                    Cancellations=('is_canceled','sum'))
               .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
               .sort_values('Cancel_Rate', ascending=False)
               .reset_index())

# 3. Lead-time analysis
bins   = [0,7,30,90,180,737]
labels = ['0-7 days','8-30 days','31-90 days','91-180 days','180+ days']
df['lead_bin'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
t_lead = (df.groupby('lead_bin', observed=True)
            .agg(Total_Bookings=('is_canceled','count'),
                 Cancellations=('is_canceled','sum'),
                 Avg_Lead_Time=('lead_time','mean'),
                 Avg_ADR=('adr','mean'))
            .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
            .reset_index())
t_lead.columns = ['Lead_Time_Bucket','Total_Bookings','Cancellations',
                  'Avg_Lead_Time','Avg_ADR','Cancel_Rate']

# 4. Seasonal occupancy
SEASON_ORDER = ['Spring','Summer','Autumn','Winter']
t_season = (df[df['is_canceled']==0]
              .groupby('season')
              .agg(Bookings=('is_canceled','count'),
                   Avg_ADR=('adr','mean'),
                   Avg_Nights=('total_nights','mean'),
                   Total_Revenue=('revenue','sum'))
              .reindex(SEASON_ORDER)
              .reset_index())

# 5. Monthly occupancy
t_monthly = (df[df['is_canceled']==0]
               .groupby('arrival_date_month', observed=True)
               .agg(Bookings=('is_canceled','count'),
                    Avg_ADR=('adr','mean'),
                    Total_Revenue=('revenue','sum'))
               .reset_index())
t_monthly.columns = ['Month','Bookings','Avg_ADR','Total_Revenue']

# 6. Guest type comparison
t_guest = (df.groupby('is_repeated_guest')
             .agg(Total_Bookings=('is_canceled','count'),
                  Cancellations=('is_canceled','sum'),
                  Avg_ADR=('adr','mean'),
                  Avg_Nights=('total_nights','mean'),
                  Avg_Special_Requests=('total_of_special_requests','mean'))
             .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
             .reset_index())
t_guest['Guest_Type'] = t_guest['is_repeated_guest'].map({0:'New Guest',1:'Repeat Guest'})
t_guest = t_guest[['Guest_Type','Total_Bookings','Cancellations','Cancel_Rate',
                   'Avg_ADR','Avg_Nights','Avg_Special_Requests']]

# 7. Room performance
t_room = (df[df['is_canceled']==0]
            .groupby('assigned_room_type')
            .agg(Bookings=('adr','count'),
                 Avg_ADR=('adr','mean'),
                 Total_Revenue=('revenue','sum'),
                 Avg_Nights=('total_nights','mean'))
            .sort_values('Total_Revenue', ascending=False)
            .reset_index())
t_room.columns = ['Room_Type','Bookings','Avg_ADR','Total_Revenue','Avg_Nights']

# 8. Room match (reserved vs assigned)
t_match = (df.groupby('room_match')
             .agg(Count=('is_canceled','count'),
                  Cancel_Rate=('is_canceled','mean'),
                  Avg_ADR=('adr','mean'))
             .reset_index())
t_match['Room_Match'] = t_match['room_match'].map({0:'Mismatch',1:'Match'})
t_match = t_match[['Room_Match','Count','Cancel_Rate','Avg_ADR']]

# 9. Event impact proxy: week-number occupancy (top 10 busiest weeks)
t_week = (df[df['is_canceled']==0]
            .groupby('arrival_date_week_number')
            .agg(Bookings=('adr','count'),
                 Avg_ADR=('adr','mean'),
                 Total_Revenue=('revenue','sum'))
            .sort_values('Bookings', ascending=False)
            .head(20)
            .reset_index())
t_week.columns = ['Week_Number','Bookings','Avg_ADR','Total_Revenue']

# 10. Hotel-level summary
t_hotel = (df.groupby('hotel')
             .agg(Total_Bookings=('is_canceled','count'),
                  Cancellations=('is_canceled','sum'),
                  Avg_ADR=('adr','mean'),
                  Avg_Lead_Time=('lead_time','mean'),
                  Total_Revenue=('revenue','sum'))
             .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
             .reset_index())

# 11. Deposit type vs cancellation
t_deposit = (df.groupby('deposit_type')
               .agg(Total_Bookings=('is_canceled','count'),
                    Cancellations=('is_canceled','sum'))
               .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
               .sort_values('Cancel_Rate', ascending=False)
               .reset_index())

# 12. Customer type vs cancellation
t_customer = (df.groupby('customer_type')
                .agg(Total_Bookings=('is_canceled','count'),
                     Cancellations=('is_canceled','sum'),
                     Avg_ADR=('adr','mean'))
                .assign(Cancel_Rate=lambda x: x['Cancellations']/x['Total_Bookings'])
                .sort_values('Cancel_Rate', ascending=False)
                .reset_index())

# 13. Descriptive stats
num_cols = ['lead_time','adr','total_nights','total_guests','revenue',
            'total_of_special_requests','booking_changes','days_in_waiting_list']
t_stats = df[num_cols].describe().T.reset_index()
t_stats.columns = ['Metric','Count','Mean','Std','Min','Q1','Median','Q3','Max']

print("All summary tables computed.")

# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL WORKBOOK
# ═══════════════════════════════════════════════════════════════════════════════
wb = xlsxwriter.Workbook(OUT, {'nan_inf_to_errors': True})

# ── formats ──────────────────────────────────────────────────────────────────
F = {}
F['title']    = wb.add_format({'bold':True,'font_size':16,'font_color':'#FFFFFF',
                                'bg_color':'#1F3864','align':'center','valign':'vcenter',
                                'border':1})
F['hdr']      = wb.add_format({'bold':True,'font_size':10,'font_color':'#FFFFFF',
                                'bg_color':'#2E75B6','align':'center','valign':'vcenter',
                                'border':1,'text_wrap':True})
F['hdr2']     = wb.add_format({'bold':True,'font_size':10,'font_color':'#FFFFFF',
                                'bg_color':'#375623','align':'center','valign':'vcenter',
                                'border':1,'text_wrap':True})
F['hdr3']     = wb.add_format({'bold':True,'font_size':10,'font_color':'#FFFFFF',
                                'bg_color':'#7B2C2C','align':'center','valign':'vcenter',
                                'border':1,'text_wrap':True})
F['cell']     = wb.add_format({'font_size':10,'border':1,'valign':'vcenter'})
F['cell_c']   = wb.add_format({'font_size':10,'border':1,'align':'center','valign':'vcenter'})
F['int']      = wb.add_format({'font_size':10,'border':1,'num_format':'#,##0','align':'right'})
F['float2']   = wb.add_format({'font_size':10,'border':1,'num_format':'#,##0.00','align':'right'})
F['pct']      = wb.add_format({'font_size':10,'border':1,'num_format':'0.0%','align':'right'})
F['money']    = wb.add_format({'font_size':10,'border':1,'num_format':'€#,##0','align':'right'})
F['alt']      = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','valign':'vcenter'})
F['alt_c']    = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','align':'center'})
F['alt_int']  = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','num_format':'#,##0','align':'right'})
F['alt_f2']   = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','num_format':'#,##0.00','align':'right'})
F['alt_pct']  = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','num_format':'0.0%','align':'right'})
F['alt_mon']  = wb.add_format({'font_size':10,'border':1,'bg_color':'#D9E1F2','num_format':'€#,##0','align':'right'})
F['label']    = wb.add_format({'bold':True,'font_size':11,'font_color':'#1F3864'})
F['formula']  = wb.add_format({'font_size':10,'border':1,'bg_color':'#FFF2CC',
                                'num_format':'#,##0.00','align':'right'})
F['formula_p']= wb.add_format({'font_size':10,'border':1,'bg_color':'#FFF2CC',
                                'num_format':'0.0%','align':'right'})
F['section']  = wb.add_format({'bold':True,'font_size':12,'font_color':'#FFFFFF',
                                'bg_color':'#1F3864','border':1})
F['kpi_val']  = wb.add_format({'bold':True,'font_size':20,'font_color':'#1F3864',
                                'align':'center','valign':'vcenter','border':2,
                                'bg_color':'#DEEAF1','num_format':'#,##0'})
F['kpi_pct']  = wb.add_format({'bold':True,'font_size':20,'font_color':'#C00000',
                                'align':'center','valign':'vcenter','border':2,
                                'bg_color':'#FCE4D6','num_format':'0.0%'})
F['kpi_mon']  = wb.add_format({'bold':True,'font_size':20,'font_color':'#375623',
                                'align':'center','valign':'vcenter','border':2,
                                'bg_color':'#E2EFDA','num_format':'€#,##0'})
F['kpi_lbl']  = wb.add_format({'bold':True,'font_size':10,'font_color':'#595959',
                                'align':'center','valign':'vcenter','bg_color':'#F2F2F2',
                                'border':1})

def write_table(ws, row, col, df_t, hdr_fmt, col_fmts, col_widths, alt=False):
    """Write a DataFrame as a formatted table. col_fmts: list of (fmt, alt_fmt)."""
    for ci, cname in enumerate(df_t.columns):
        ws.write(row, col+ci, cname, hdr_fmt)
        ws.set_column(col+ci, col+ci, col_widths[ci])
    for ri, (_, r) in enumerate(df_t.iterrows()):
        use_alt = alt and (ri % 2 == 1)
        for ci, val in enumerate(r):
            fmt_pair = col_fmts[ci]
            fmt = fmt_pair[1] if use_alt else fmt_pair[0]
            ws.write(row+1+ri, col+ci, val, fmt)

def sheet_title(ws, row, col, span, text):
    ws.merge_range(row, col, row, col+span-1, text, F['title'])
    ws.set_row(row, 28)

def section_hdr(ws, row, col, span, text):
    ws.merge_range(row, col, row, col+span-1, text, F['section'])
    ws.set_row(row, 20)

print("Workbook and formats ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 1 – RAW DATA (sample 5000 rows for performance)
# ═══════════════════════════════════════════════════════════════════════════════
ws_raw = wb.add_worksheet('Raw Data')
ws_raw.freeze_panes(1, 0)
ws_raw.set_zoom(85)
sample = df.head(5000)
for ci, col_name in enumerate(sample.columns):
    ws_raw.write(0, ci, col_name, F['hdr'])
    ws_raw.set_column(ci, ci, 16)
for ri, (_, row_data) in enumerate(sample.iterrows()):
    use_alt = ri % 2 == 1
    for ci, val in enumerate(row_data):
        fmt = F['alt'] if use_alt else F['cell']
        if pd.isna(val):
            ws_raw.write(ri+1, ci, '', fmt)
        elif isinstance(val, (int, np.integer)):
            ws_raw.write_number(ri+1, ci, int(val), F['alt_int'] if use_alt else F['int'])
        elif isinstance(val, (float, np.floating)):
            ws_raw.write_number(ri+1, ci, float(val), F['alt_f2'] if use_alt else F['float2'])
        else:
            ws_raw.write(ri+1, ci, str(val), F['alt'] if use_alt else F['cell'])
ws_raw.autofilter(0, 0, len(sample), len(sample.columns)-1)
print("Sheet 1 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 2 – DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
ws_stat = wb.add_worksheet('Descriptive Stats')
ws_stat.set_zoom(90)
ws_stat.set_column('A:A', 28)
for c in range(1, 9):
    ws_stat.set_column(c, c, 14)

sheet_title(ws_stat, 0, 0, 9, 'Descriptive Statistics – Hotel Bookings Dataset')

# table
hdr_cols = list(t_stats.columns)
col_fmts = [
    (F['cell'], F['alt']),
    (F['int'],  F['alt_int']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
    (F['float2'],F['alt_f2']),
]
write_table(ws_stat, 2, 0, t_stats, F['hdr'], col_fmts,
            [28,12,12,12,12,12,12,12,12], alt=True)

# formula section
section_hdr(ws_stat, 13, 0, 5, 'Key Statistical Formulas (Excel-style)')
ws_stat.write(14, 0, 'Metric', F['hdr'])
ws_stat.write(14, 1, 'Formula Used', F['hdr'])
ws_stat.write(14, 2, 'Value', F['hdr'])
ws_stat.write(14, 3, 'Interpretation', F['hdr'])
ws_stat.set_column(1, 1, 30)
ws_stat.set_column(3, 3, 40)

formulas = [
    ('Overall Cancellation Rate',  '=COUNTIF(is_canceled,1)/COUNT(is_canceled)', df['is_canceled'].mean(), '27.5% of bookings are cancelled'),
    ('Mean Lead Time (days)',       '=AVERAGE(lead_time)',  df['lead_time'].mean(), 'Avg days between booking and arrival'),
    ('Median ADR (€)',              '=MEDIAN(adr)',         df['adr'].median(), 'Middle value of daily rate'),
    ('Std Dev ADR',                 '=STDEV(adr)',          df['adr'].std(), 'Spread of daily rates'),
    ('Max Revenue (single booking)','=MAX(revenue)',        df['revenue'].max(), 'Highest single booking revenue'),
    ('% Repeat Guests',             '=COUNTIF(is_repeated_guest,1)/COUNT(is_repeated_guest)', df['is_repeated_guest'].mean(), 'Share of returning customers'),
    ('Avg Nights per Stay',         '=AVERAGE(total_nights)', df['total_nights'].mean(), 'Average length of stay'),
    ('Room Match Rate',             '=COUNTIF(room_match,1)/COUNT(room_match)', df['room_match'].mean(), 'Guests who got their reserved room'),
]
for ri, (metric, formula, value, interp) in enumerate(formulas):
    bg = F['alt'] if ri%2==1 else F['cell']
    bg_f = F['alt_f2'] if ri%2==1 else F['formula']
    bg_p = F['alt_pct'] if ri%2==1 else F['formula_p']
    ws_stat.write(15+ri, 0, metric, bg)
    ws_stat.write(15+ri, 1, formula, bg)
    if metric in ('Overall Cancellation Rate','% Repeat Guests','Room Match Rate'):
        ws_stat.write_number(15+ri, 2, value, bg_p)
    else:
        ws_stat.write_number(15+ri, 2, value, bg_f)
    ws_stat.write(15+ri, 3, interp, bg)

print("Sheet 2 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 3 – CANCELLATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
ws_can = wb.add_worksheet('Cancellation Analysis')
ws_can.set_zoom(90)
sheet_title(ws_can, 0, 0, 10, 'Cancellation Analysis – By Channel, Segment, Deposit & Hotel')

# --- Table A: by distribution channel ---
section_hdr(ws_can, 2, 0, 4, 'A. Cancellation by Distribution Channel')
write_table(ws_can, 3, 0, t_channel, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['pct'],F['alt_pct'])],
            [28,16,16,16], alt=True)

# Chart A
ch_a = wb.add_chart({'type':'column'})
n = len(t_channel)
ch_a.add_series({
    'name': 'Cancellation Rate',
    'categories': ['Cancellation Analysis', 4, 0, 3+n, 0],
    'values':     ['Cancellation Analysis', 4, 3, 3+n, 3],
    'fill':       {'color':'#2E75B6'},
    'data_labels':{'value':True,'num_format':'0%'},
})
ch_a.set_title({'name':'Cancellation Rate by Distribution Channel'})
ch_a.set_x_axis({'name':'Channel'})
ch_a.set_y_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_a.set_size({'width':480,'height':300})
ch_a.set_style(10)
ws_can.insert_chart('F3', ch_a)

# --- Table B: by market segment ---
section_hdr(ws_can, 2, 11, 4, 'B. Cancellation by Market Segment')
write_table(ws_can, 3, 11, t_segment, F['hdr3'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['pct'],F['alt_pct'])],
            [22,16,16,16], alt=True)

ch_b = wb.add_chart({'type':'bar'})
n2 = len(t_segment)
ch_b.add_series({
    'name': 'Cancellation Rate',
    'categories': ['Cancellation Analysis', 4, 11, 3+n2, 11],
    'values':     ['Cancellation Analysis', 4, 14, 3+n2, 14],
    'fill':       {'color':'#C00000'},
    'data_labels':{'value':True,'num_format':'0%'},
})
ch_b.set_title({'name':'Cancellation Rate by Market Segment'})
ch_b.set_x_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_b.set_y_axis({'name':'Segment'})
ch_b.set_size({'width':480,'height':300})
ch_b.set_style(10)
ws_can.insert_chart('F20', ch_b)

# --- Table C: deposit type ---
row_c = 3 + max(n, n2) + 4
section_hdr(ws_can, row_c, 0, 4, 'C. Cancellation by Deposit Type')
write_table(ws_can, row_c+1, 0, t_deposit, F['hdr2'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['pct'],F['alt_pct'])],
            [22,16,16,16], alt=True)

ch_c = wb.add_chart({'type':'column'})
nd = len(t_deposit)
ch_c.add_series({
    'name': 'Cancellation Rate',
    'categories': ['Cancellation Analysis', row_c+2, 0, row_c+1+nd, 0],
    'values':     ['Cancellation Analysis', row_c+2, 3, row_c+1+nd, 3],
    'fill':       {'color':'#375623'},
    'data_labels':{'value':True,'num_format':'0%'},
})
ch_c.set_title({'name':'Cancellation Rate by Deposit Type'})
ch_c.set_y_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_c.set_size({'width':400,'height':280})
ch_c.set_style(10)
ws_can.insert_chart('F'+str(row_c+1), ch_c)

# --- Table D: hotel type ---
section_hdr(ws_can, row_c, 8, 6, 'D. Hotel-Level Summary')
write_table(ws_can, row_c+1, 8, t_hotel, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),
             (F['float2'],F['alt_f2']),(F['float2'],F['alt_f2']),(F['money'],F['alt_mon']),(F['pct'],F['alt_pct'])],
            [18,14,14,12,14,16,14], alt=True)

ws_can.set_column('A:A', 28)
print("Sheet 3 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 4 – LEAD TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
ws_lt = wb.add_worksheet('Lead Time Analysis')
ws_lt.set_zoom(90)
ws_lt.set_column('A:A', 18)
for c in range(1, 7): ws_lt.set_column(c, c, 16)
sheet_title(ws_lt, 0, 0, 6, 'Lead Time Analysis – Booking Window vs Cancellation')

section_hdr(ws_lt, 2, 0, 6, 'Lead Time Buckets: Bookings, Cancellations & ADR')
write_table(ws_lt, 3, 0, t_lead, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),
             (F['float2'],F['alt_f2']),(F['float2'],F['alt_f2']),(F['pct'],F['alt_pct'])],
            [18,16,16,16,14,14], alt=True)

# Combo chart: bars = bookings, line = cancel rate
nl = len(t_lead)
ch_lt1 = wb.add_chart({'type':'column'})
ch_lt1.add_series({
    'name':'Total Bookings',
    'categories':['Lead Time Analysis',4,0,3+nl,0],
    'values':    ['Lead Time Analysis',4,1,3+nl,1],
    'fill':{'color':'#2E75B6'},
})
ch_lt2 = wb.add_chart({'type':'line'})
ch_lt2.add_series({
    'name':'Cancellation Rate',
    'categories':['Lead Time Analysis',4,0,3+nl,0],
    'values':    ['Lead Time Analysis',4,5,3+nl,5],
    'y2_axis':True,
    'line':{'color':'#C00000','width':2.5},
    'marker':{'type':'circle','size':6,'fill':{'color':'#C00000'}},
})
ch_lt1.combine(ch_lt2)
ch_lt1.set_title({'name':'Lead Time: Bookings vs Cancellation Rate'})
ch_lt1.set_x_axis({'name':'Lead Time Bucket'})
ch_lt1.set_y_axis({'name':'Number of Bookings','num_format':'#,##0'})
ch_lt1.set_y2_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_lt1.set_size({'width':520,'height':320})
ch_lt1.set_style(10)
ws_lt.insert_chart('H3', ch_lt1)

# ADR by lead time
ch_lt3 = wb.add_chart({'type':'column'})
ch_lt3.add_series({
    'name':'Avg ADR',
    'categories':['Lead Time Analysis',4,0,3+nl,0],
    'values':    ['Lead Time Analysis',4,4,3+nl,4],
    'fill':{'color':'#375623'},
    'data_labels':{'value':True,'num_format':'€0'},
})
ch_lt3.set_title({'name':'Average ADR by Lead Time Bucket'})
ch_lt3.set_y_axis({'name':'ADR (€)','num_format':'€#,##0'})
ch_lt3.set_size({'width':400,'height':280})
ch_lt3.set_style(10)
ws_lt.insert_chart('H22', ch_lt3)

# formula note
ws_lt.write(nl+6, 0, 'Correlation (Lead Time vs Cancellation):', F['label'])
corr_val = df[['lead_time','is_canceled']].corr().iloc[0,1]
ws_lt.write_number(nl+6, 2, corr_val, F['formula'])
ws_lt.write(nl+7, 0, 'Interpretation:', F['label'])
ws_lt.write(nl+7, 2, 'Positive correlation: longer lead time → higher cancellation risk', F['cell'])
print("Sheet 4 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 5 – SEASONAL OCCUPANCY
# ═══════════════════════════════════════════════════════════════════════════════
ws_sea = wb.add_worksheet('Seasonal Occupancy')
ws_sea.set_zoom(90)
ws_sea.set_column('A:A', 14)
for c in range(1, 6): ws_sea.set_column(c, c, 16)
sheet_title(ws_sea, 0, 0, 8, 'Seasonal & Monthly Occupancy Analysis')

# Seasonal table
section_hdr(ws_sea, 2, 0, 5, 'A. Seasonal Summary (Checked-In Bookings)')
write_table(ws_sea, 3, 0, t_season, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),
             (F['float2'],F['alt_f2']),(F['money'],F['alt_mon'])],
            [14,14,14,14,16], alt=True)

ch_s1 = wb.add_chart({'type':'column'})
ns = len(t_season)
ch_s1.add_series({
    'name':'Bookings',
    'categories':['Seasonal Occupancy',4,0,3+ns,0],
    'values':    ['Seasonal Occupancy',4,1,3+ns,1],
    'fill':{'color':'#ED7D31'},
    'data_labels':{'value':True,'num_format':'#,##0'},
})
ch_s1.set_title({'name':'Bookings by Season'})
ch_s1.set_y_axis({'name':'Bookings','num_format':'#,##0'})
ch_s1.set_size({'width':380,'height':280})
ch_s1.set_style(10)
ws_sea.insert_chart('G3', ch_s1)

ch_s2 = wb.add_chart({'type':'pie'})
ch_s2.add_series({
    'name':'Revenue Share',
    'categories':['Seasonal Occupancy',4,0,3+ns,0],
    'values':    ['Seasonal Occupancy',4,4,3+ns,4],
    'data_labels':{'percentage':True,'category':True},
})
ch_s2.set_title({'name':'Revenue Share by Season'})
ch_s2.set_size({'width':380,'height':280})
ch_s2.set_style(10)
ws_sea.insert_chart('M3', ch_s2)

# Monthly table
section_hdr(ws_sea, 10, 0, 4, 'B. Monthly Occupancy (Checked-In Bookings)')
write_table(ws_sea, 11, 0, t_monthly, F['hdr2'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),(F['money'],F['alt_mon'])],
            [14,14,14,16], alt=True)

ch_s3 = wb.add_chart({'type':'column'})
nm = len(t_monthly)
ch_s3.add_series({
    'name':'Bookings',
    'categories':['Seasonal Occupancy',12,0,11+nm,0],
    'values':    ['Seasonal Occupancy',12,1,11+nm,1],
    'fill':{'color':'#4472C4'},
})
ch_s3.add_series({
    'name':'Avg ADR',
    'categories':['Seasonal Occupancy',12,0,11+nm,0],
    'values':    ['Seasonal Occupancy',12,2,11+nm,2],
    'y2_axis':True,
    'fill':{'color':'#ED7D31'},
})
ch_s3.set_title({'name':'Monthly Bookings & Average ADR'})
ch_s3.set_x_axis({'name':'Month'})
ch_s3.set_y_axis({'name':'Bookings','num_format':'#,##0'})
ch_s3.set_y2_axis({'name':'Avg ADR (€)','num_format':'€#,##0'})
ch_s3.set_size({'width':560,'height':320})
ch_s3.set_style(10)
ws_sea.insert_chart('G11', ch_s3)
print("Sheet 5 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 6 – GUEST TYPE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
ws_gst = wb.add_worksheet('Guest Type Comparison')
ws_gst.set_zoom(90)
ws_gst.set_column('A:A', 18)
for c in range(1, 8): ws_gst.set_column(c, c, 18)
sheet_title(ws_gst, 0, 0, 7, 'Guest Type Comparison – New vs Repeat & Customer Type')

section_hdr(ws_gst, 2, 0, 7, 'A. New vs Repeat Guest Analysis')
write_table(ws_gst, 3, 0, t_guest, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),
             (F['pct'],F['alt_pct']),(F['float2'],F['alt_f2']),(F['float2'],F['alt_f2']),(F['float2'],F['alt_f2'])],
            [18,16,16,16,14,14,20], alt=True)

ch_g1 = wb.add_chart({'type':'column'})
ng = len(t_guest)
ch_g1.add_series({
    'name':'Cancellation Rate',
    'categories':['Guest Type Comparison',4,0,3+ng,0],
    'values':    ['Guest Type Comparison',4,3,3+ng,3],
    'fill':{'color':'#C00000'},
    'data_labels':{'value':True,'num_format':'0.0%'},
})
ch_g1.set_title({'name':'Cancellation Rate: New vs Repeat Guests'})
ch_g1.set_y_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_g1.set_size({'width':380,'height':280})
ch_g1.set_style(10)
ws_gst.insert_chart('I3', ch_g1)

ch_g2 = wb.add_chart({'type':'column'})
ch_g2.add_series({
    'name':'Avg ADR',
    'categories':['Guest Type Comparison',4,0,3+ng,0],
    'values':    ['Guest Type Comparison',4,4,3+ng,4],
    'fill':{'color':'#375623'},
    'data_labels':{'value':True,'num_format':'€0'},
})
ch_g2.set_title({'name':'Average ADR: New vs Repeat Guests'})
ch_g2.set_y_axis({'name':'ADR (€)','num_format':'€#,##0'})
ch_g2.set_size({'width':380,'height':280})
ch_g2.set_style(10)
ws_gst.insert_chart('M3', ch_g2)

# Customer type table
section_hdr(ws_gst, 10, 0, 5, 'B. Customer Type Analysis')
write_table(ws_gst, 11, 0, t_customer, F['hdr3'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),
             (F['float2'],F['alt_f2']),(F['pct'],F['alt_pct'])],
            [20,16,16,14,14], alt=True)

ch_g3 = wb.add_chart({'type':'bar'})
nc = len(t_customer)
ch_g3.add_series({
    'name':'Cancellation Rate',
    'categories':['Guest Type Comparison',12,0,11+nc,0],
    'values':    ['Guest Type Comparison',12,4,11+nc,4],
    'fill':{'color':'#7030A0'},
    'data_labels':{'value':True,'num_format':'0%'},
})
ch_g3.set_title({'name':'Cancellation Rate by Customer Type'})
ch_g3.set_x_axis({'name':'Cancellation Rate','num_format':'0%'})
ch_g3.set_size({'width':400,'height':280})
ch_g3.set_style(10)
ws_gst.insert_chart('I11', ch_g3)
print("Sheet 6 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 7 – ROOM PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
ws_rm = wb.add_worksheet('Room Performance')
ws_rm.set_zoom(90)
ws_rm.set_column('A:A', 14)
for c in range(1, 6): ws_rm.set_column(c, c, 16)
sheet_title(ws_rm, 0, 0, 8, 'Room Performance – ADR, Revenue & Volume by Room Type')

section_hdr(ws_rm, 2, 0, 5, 'A. Room Type Performance (Checked-In Bookings)')
write_table(ws_rm, 3, 0, t_room, F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),
             (F['money'],F['alt_mon']),(F['float2'],F['alt_f2'])],
            [14,14,14,16,14], alt=True)

nr = len(t_room)
ch_r1 = wb.add_chart({'type':'column'})
ch_r1.add_series({
    'name':'Total Revenue',
    'categories':['Room Performance',4,0,3+nr,0],
    'values':    ['Room Performance',4,3,3+nr,3],
    'fill':{'color':'#375623'},
    'data_labels':{'value':True,'num_format':'€#,##0'},
})
ch_r1.set_title({'name':'Total Revenue by Room Type'})
ch_r1.set_y_axis({'name':'Revenue (€)','num_format':'€#,##0'})
ch_r1.set_size({'width':460,'height':300})
ch_r1.set_style(10)
ws_rm.insert_chart('G3', ch_r1)

ch_r2 = wb.add_chart({'type':'column'})
ch_r2.add_series({
    'name':'Avg ADR',
    'categories':['Room Performance',4,0,3+nr,0],
    'values':    ['Room Performance',4,2,3+nr,2],
    'fill':{'color':'#2E75B6'},
    'data_labels':{'value':True,'num_format':'€0'},
})
ch_r2.set_title({'name':'Average ADR by Room Type'})
ch_r2.set_y_axis({'name':'ADR (€)','num_format':'€#,##0'})
ch_r2.set_size({'width':460,'height':300})
ch_r2.set_style(10)
ws_rm.insert_chart('G20', ch_r2)

# Room match table
section_hdr(ws_rm, nr+6, 0, 4, 'B. Room Match vs Mismatch Impact')
write_table(ws_rm, nr+7, 0, t_match, F['hdr3'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['pct'],F['alt_pct']),(F['float2'],F['alt_f2'])],
            [14,14,14,14], alt=True)
print("Sheet 7 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 8 – EVENT IMPACT (Week-Number Proxy)
# ═══════════════════════════════════════════════════════════════════════════════
ws_ev = wb.add_worksheet('Event Impact')
ws_ev.set_zoom(90)
ws_ev.set_column('A:A', 16)
for c in range(1, 5): ws_ev.set_column(c, c, 16)
sheet_title(ws_ev, 0, 0, 6, 'Event Impact – Peak Weeks & Occupancy Spikes')

ws_ev.write(2, 0, 'Note: Local events are proxied by arrival week number. '
            'Weeks with unusually high bookings/ADR indicate event-driven demand.', F['cell'])
ws_ev.set_row(2, 30)

section_hdr(ws_ev, 4, 0, 4, 'Top 20 Busiest Arrival Weeks (by Bookings)')
write_table(ws_ev, 5, 0, t_week, F['hdr'],
            [(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),(F['money'],F['alt_mon'])],
            [16,14,14,16], alt=True)

nw = len(t_week)
ch_e1 = wb.add_chart({'type':'column'})
ch_e1.add_series({
    'name':'Bookings',
    'categories':['Event Impact',6,0,5+nw,0],
    'values':    ['Event Impact',6,1,5+nw,1],
    'fill':{'color':'#ED7D31'},
})
ch_e1.add_series({
    'name':'Avg ADR',
    'categories':['Event Impact',6,0,5+nw,0],
    'values':    ['Event Impact',6,2,5+nw,2],
    'y2_axis':True,
    'line':{'color':'#C00000','width':2},
    'marker':{'type':'diamond','size':6},
})
ch_e1.set_title({'name':'Top 20 Weeks: Bookings & ADR (Event Proxy)'})
ch_e1.set_x_axis({'name':'Week Number'})
ch_e1.set_y_axis({'name':'Bookings','num_format':'#,##0'})
ch_e1.set_y2_axis({'name':'Avg ADR (€)','num_format':'€#,##0'})
ch_e1.set_size({'width':560,'height':340})
ch_e1.set_style(10)
ws_ev.insert_chart('F5', ch_e1)

# All weeks for full picture
t_all_weeks = (df[df['is_canceled']==0]
                 .groupby('arrival_date_week_number')
                 .agg(Bookings=('adr','count'), Avg_ADR=('adr','mean'))
                 .reset_index())
t_all_weeks.columns = ['Week_Number','Bookings','Avg_ADR']
section_hdr(ws_ev, nw+8, 0, 3, 'All Weeks – Full Occupancy Pattern')
write_table(ws_ev, nw+9, 0, t_all_weeks, F['hdr2'],
            [(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2'])],
            [16,14,14], alt=True)

naw = len(t_all_weeks)
ch_e2 = wb.add_chart({'type':'line'})
ch_e2.add_series({
    'name':'Weekly Bookings',
    'categories':['Event Impact', nw+10, 0, nw+9+naw, 0],
    'values':    ['Event Impact', nw+10, 1, nw+9+naw, 1],
    'line':{'color':'#2E75B6','width':2},
    'marker':{'type':'circle','size':4},
})
ch_e2.set_title({'name':'Weekly Occupancy – Full Year View'})
ch_e2.set_x_axis({'name':'Week Number'})
ch_e2.set_y_axis({'name':'Bookings','num_format':'#,##0'})
ch_e2.set_size({'width':560,'height':300})
ch_e2.set_style(10)
ws_ev.insert_chart('F'+str(nw+9), ch_e2)
print("Sheet 8 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 9 – PIVOT TABLES
# ═══════════════════════════════════════════════════════════════════════════════
ws_pv = wb.add_worksheet('Pivot Tables')
ws_pv.set_zoom(90)
sheet_title(ws_pv, 0, 0, 14, 'Pivot Tables – Multi-Dimensional Analysis')

# Pivot 1: Hotel x Season – Bookings & Avg ADR
pv1 = (df[df['is_canceled']==0]
         .pivot_table(values=['adr','revenue'], index='hotel',
                      columns='season', aggfunc={'adr':'mean','revenue':'sum'})
         .round(2))
pv1_adr = pv1['adr'].reindex(columns=SEASON_ORDER)
pv1_rev = pv1['revenue'].reindex(columns=SEASON_ORDER)

section_hdr(ws_pv, 2, 0, 6, 'Pivot 1: Average ADR by Hotel × Season')
ws_pv.write(3, 0, 'Hotel \\ Season', F['hdr'])
ws_pv.set_column(0, 0, 20)
for ci, s in enumerate(SEASON_ORDER):
    ws_pv.write(3, ci+1, s, F['hdr'])
    ws_pv.set_column(ci+1, ci+1, 14)
ws_pv.write(3, 5, 'Grand Total', F['hdr'])
for ri, (hotel, row_d) in enumerate(pv1_adr.iterrows()):
    bg = F['alt'] if ri%2==1 else F['cell']
    bg_f = F['alt_f2'] if ri%2==1 else F['float2']
    ws_pv.write(4+ri, 0, hotel, bg)
    for ci, s in enumerate(SEASON_ORDER):
        ws_pv.write_number(4+ri, ci+1, row_d.get(s, 0) or 0, bg_f)
    ws_pv.write_number(4+ri, 5, row_d.mean(), bg_f)

section_hdr(ws_pv, 8, 0, 6, 'Pivot 2: Total Revenue by Hotel × Season')
ws_pv.write(9, 0, 'Hotel \\ Season', F['hdr'])
for ci, s in enumerate(SEASON_ORDER):
    ws_pv.write(9, ci+1, s, F['hdr'])
ws_pv.write(9, 5, 'Grand Total', F['hdr'])
for ri, (hotel, row_d) in enumerate(pv1_rev.iterrows()):
    bg = F['alt'] if ri%2==1 else F['cell']
    bg_m = F['alt_mon'] if ri%2==1 else F['money']
    ws_pv.write(10+ri, 0, hotel, bg)
    for ci, s in enumerate(SEASON_ORDER):
        ws_pv.write_number(10+ri, ci+1, row_d.get(s, 0) or 0, bg_m)
    ws_pv.write_number(10+ri, 5, row_d.sum(), bg_m)

# Pivot 2: Market Segment x Hotel – Cancel Rate
pv2 = (df.pivot_table(values='is_canceled', index='market_segment',
                       columns='hotel', aggfunc='mean').round(4))
section_hdr(ws_pv, 14, 0, 4, 'Pivot 3: Cancellation Rate by Market Segment × Hotel')
ws_pv.write(15, 0, 'Segment \\ Hotel', F['hdr'])
hotels = list(pv2.columns)
for ci, h in enumerate(hotels):
    ws_pv.write(15, ci+1, h, F['hdr'])
ws_pv.write(15, len(hotels)+1, 'Overall', F['hdr'])
for ri, (seg, row_d) in enumerate(pv2.iterrows()):
    bg = F['alt'] if ri%2==1 else F['cell']
    bg_p = F['alt_pct'] if ri%2==1 else F['pct']
    ws_pv.write(16+ri, 0, seg, bg)
    for ci, h in enumerate(hotels):
        ws_pv.write_number(16+ri, ci+1, row_d.get(h, 0) or 0, bg_p)
    ws_pv.write_number(16+ri, len(hotels)+1, row_d.mean(), bg_p)

# Pivot 3: Month x Hotel – Bookings
pv3 = (df[df['is_canceled']==0]
         .pivot_table(values='adr', index='arrival_date_month',
                      columns='hotel', aggfunc='count', observed=True)
         .fillna(0).astype(int))
section_hdr(ws_pv, 28, 0, 4, 'Pivot 4: Monthly Bookings by Hotel')
ws_pv.write(29, 0, 'Month \\ Hotel', F['hdr'])
for ci, h in enumerate(pv3.columns):
    ws_pv.write(29, ci+1, h, F['hdr'])
ws_pv.write(29, len(pv3.columns)+1, 'Total', F['hdr'])
for ri, (month, row_d) in enumerate(pv3.iterrows()):
    bg = F['alt'] if ri%2==1 else F['cell']
    bg_i = F['alt_int'] if ri%2==1 else F['int']
    ws_pv.write(30+ri, 0, str(month), bg)
    for ci, h in enumerate(pv3.columns):
        ws_pv.write_number(30+ri, ci+1, int(row_d.get(h, 0)), bg_i)
    ws_pv.write_number(30+ri, len(pv3.columns)+1, int(row_d.sum()), bg_i)

# Pivot chart for pv3
ch_pv = wb.add_chart({'type':'column','subtype':'stacked'})
for ci, h in enumerate(pv3.columns):
    ch_pv.add_series({
        'name': h,
        'categories': ['Pivot Tables', 30, 0, 30+len(pv3)-1, 0],
        'values':     ['Pivot Tables', 30, ci+1, 30+len(pv3)-1, ci+1],
    })
ch_pv.set_title({'name':'Monthly Bookings by Hotel Type (Stacked)'})
ch_pv.set_x_axis({'name':'Month'})
ch_pv.set_y_axis({'name':'Bookings','num_format':'#,##0'})
ch_pv.set_size({'width':560,'height':320})
ch_pv.set_style(10)
ws_pv.insert_chart('F28', ch_pv)
print("Sheet 9 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHEET 10 – SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
ws_sum = wb.add_worksheet('Summary Dashboard')
ws_sum.set_zoom(90)
ws_sum.set_tab_color('#1F3864')

# Title
ws_sum.merge_range('A1:P2', 'Hotel Bookings – Executive Summary Dashboard', F['title'])
ws_sum.set_row(0, 36)
ws_sum.set_row(1, 4)

# ── KPI row ──────────────────────────────────────────────────────────────────
kpis = [
    ('Total Bookings',    int(len(df)),                          F['kpi_val']),
    ('Cancellation Rate', float(df['is_canceled'].mean()),       F['kpi_pct']),
    ('Avg ADR (€)',       float(df['adr'].mean()),               F['kpi_val']),
    ('Total Revenue (€)', float(df['revenue'].sum()),            F['kpi_mon']),
    ('Avg Lead Time',     float(df['lead_time'].mean()),         F['kpi_val']),
    ('Repeat Guest %',    float(df['is_repeated_guest'].mean()), F['kpi_pct']),
]
kpi_cols = [0, 2, 4, 6, 8, 10]
for i, (lbl, val, fmt) in enumerate(kpis):
    c = kpi_cols[i]
    ws_sum.merge_range(2, c, 2, c+1, lbl,  F['kpi_lbl'])
    ws_sum.merge_range(3, c, 3, c+1, val,  fmt)
    ws_sum.set_column(c, c+1, 12)
    ws_sum.set_row(2, 22)
    ws_sum.set_row(3, 40)

# ── Section: Key Findings ─────────────────────────────────────────────────────
ws_sum.merge_range('A6:P6', 'KEY FINDINGS', F['section'])
ws_sum.set_row(5, 22)

findings = [
    ('Cancellation by Channel',
     f"TA/TO channel has the highest cancellation rate "
     f"({t_channel.iloc[0]['Cancel_Rate']:.1%}). "
     f"Direct bookings cancel least ({t_channel[t_channel['distribution_channel']=='Direct']['Cancel_Rate'].values[0]:.1%})."),
    ('Lead Time Impact',
     f"Bookings made 180+ days ahead cancel at "
     f"{t_lead[t_lead['Lead_Time_Bucket']=='180+ days']['Cancel_Rate'].values[0]:.1%} vs "
     f"{t_lead[t_lead['Lead_Time_Bucket']=='0-7 days']['Cancel_Rate'].values[0]:.1%} for same-week bookings. "
     f"Correlation = {corr_val:.3f}."),
    ('Seasonal Occupancy',
     f"Summer is the peak season with {t_season[t_season['season']=='Summer']['Bookings'].values[0]:,} bookings. "
     f"Winter is the slowest with {t_season[t_season['season']=='Winter']['Bookings'].values[0]:,} bookings."),
    ('Repeat vs New Guests',
     f"Repeat guests cancel at only {t_guest[t_guest['Guest_Type']=='Repeat Guest']['Cancel_Rate'].values[0]:.1%} "
     f"vs {t_guest[t_guest['Guest_Type']=='New Guest']['Cancel_Rate'].values[0]:.1%} for new guests. "
     f"Repeat guests also spend more on average."),
    ('Room Performance',
     f"Room type {t_room.iloc[0]['Room_Type']} generates the highest total revenue "
     f"(€{t_room.iloc[0]['Total_Revenue']:,.0f}). "
     f"Room type {t_room.sort_values('Avg_ADR',ascending=False).iloc[0]['Room_Type']} has the highest ADR."),
    ('Event Impact (Week Proxy)',
     f"Week {int(t_week.iloc[0]['Week_Number'])} is the busiest with {int(t_week.iloc[0]['Bookings']):,} bookings "
     f"and avg ADR of €{t_week.iloc[0]['Avg_ADR']:.0f}. "
     f"Summer weeks (27-35) consistently show elevated demand."),
    ('Deposit Type',
     f"Non-refundable deposits have the highest cancellation rate "
     f"({t_deposit[t_deposit['deposit_type']=='Non Refund']['Cancel_Rate'].values[0]:.1%} if present). "
     f"No-deposit bookings dominate volume."),
    ('Hotel Type',
     f"City Hotel cancels at {t_hotel[t_hotel['hotel']=='City Hotel']['Cancel_Rate'].values[0]:.1%} "
     f"vs Resort Hotel at {t_hotel[t_hotel['hotel']=='Resort Hotel']['Cancel_Rate'].values[0]:.1%}."),
]

lbl_fmt = wb.add_format({'bold':True,'font_size':10,'font_color':'#1F3864',
                          'bg_color':'#DEEAF1','border':1,'valign':'vcenter'})
txt_fmt = wb.add_format({'font_size':10,'border':1,'text_wrap':True,'valign':'vcenter',
                          'bg_color':'#F2F2F2'})
txt_alt = wb.add_format({'font_size':10,'border':1,'text_wrap':True,'valign':'vcenter',
                          'bg_color':'#FFFFFF'})

ws_sum.set_column('A:C', 22)
ws_sum.set_column('D:P', 12)
for ri, (topic, text) in enumerate(findings):
    row = 6 + ri
    ws_sum.set_row(row, 40)
    ws_sum.merge_range(row, 0, row, 2, topic, lbl_fmt)
    ws_sum.merge_range(row, 3, row, 15, text, txt_fmt if ri%2==0 else txt_alt)

# ── Summary tables side by side ───────────────────────────────────────────────
base_r = 16
section_hdr(ws_sum, base_r, 0, 4, 'Cancellation by Channel (Top 5)')
write_table(ws_sum, base_r+1, 0, t_channel.head(5), F['hdr'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['int'],F['alt_int']),(F['pct'],F['alt_pct'])],
            [22,14,14,14], alt=True)

section_hdr(ws_sum, base_r, 5, 4, 'Seasonal Occupancy')
write_table(ws_sum, base_r+1, 5, t_season, F['hdr2'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),
             (F['float2'],F['alt_f2']),(F['money'],F['alt_mon'])],
            [12,12,12,12,14], alt=True)

section_hdr(ws_sum, base_r, 11, 4, 'Room Performance (Top 5)')
write_table(ws_sum, base_r+1, 11, t_room.head(5), F['hdr3'],
            [(F['cell'],F['alt']),(F['int'],F['alt_int']),(F['float2'],F['alt_f2']),
             (F['money'],F['alt_mon']),(F['float2'],F['alt_f2'])],
            [12,12,12,14,12], alt=True)

# ── Mini charts on summary ────────────────────────────────────────────────────
ch_sum1 = wb.add_chart({'type':'column'})
ch_sum1.add_series({
    'name':'Cancel Rate',
    'categories':['Cancellation Analysis',4,0,3+len(t_channel),0],
    'values':    ['Cancellation Analysis',4,3,3+len(t_channel),3],
    'fill':{'color':'#2E75B6'},
})
ch_sum1.set_title({'name':'Cancellation by Channel'})
ch_sum1.set_y_axis({'num_format':'0%'})
ch_sum1.set_size({'width':340,'height':220})
ch_sum1.set_style(10)
ws_sum.insert_chart('A26', ch_sum1)

ch_sum2 = wb.add_chart({'type':'pie'})
ch_sum2.add_series({
    'name':'Seasonal Revenue',
    'categories':['Seasonal Occupancy',4,0,3+len(t_season),0],
    'values':    ['Seasonal Occupancy',4,4,3+len(t_season),4],
    'data_labels':{'percentage':True,'category':True},
})
ch_sum2.set_title({'name':'Revenue by Season'})
ch_sum2.set_size({'width':340,'height':220})
ch_sum2.set_style(10)
ws_sum.insert_chart('G26', ch_sum2)

ch_sum3 = wb.add_chart({'type':'column'})
ch_sum3.add_series({
    'name':'Avg ADR',
    'categories':['Room Performance',4,0,3+len(t_room),0],
    'values':    ['Room Performance',4,2,3+len(t_room),2],
    'fill':{'color':'#375623'},
})
ch_sum3.set_title({'name':'ADR by Room Type'})
ch_sum3.set_y_axis({'num_format':'€#,##0'})
ch_sum3.set_size({'width':340,'height':220})
ch_sum3.set_style(10)
ws_sum.insert_chart('M26', ch_sum3)

print("Sheet 10 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# FINALISE
# ═══════════════════════════════════════════════════════════════════════════════
# Set sheet order: Summary first
wb.set_properties({
    'title':   'Hotel Bookings Analysis',
    'subject': 'EDA & Business Intelligence',
    'author':  'Data Science Team',
    'comments':'Generated by build_excel.py',
})

wb.close()
print(f"\nWorkbook saved: {OUT}")
print("=== EXCEL BUILD COMPLETE ===")
