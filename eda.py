import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
os_path = os.path.join(BASE, 'outputs') + os.sep
if not os.path.exists(os_path.rstrip(os.sep)):
    os.mkdir(os_path.rstrip(os.sep))

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'hotel_bookings.csv'))
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)

# ── 2. DATA QUALITY ──────────────────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDuplicates:", df.duplicated().sum())

# Unusual: adr < 0, guests == 0
print("\nNegative ADR:", (df['adr'] < 0).sum())
print("Zero guests:", ((df['adults'] + df['children'].fillna(0) + df['babies']) == 0).sum())

# ── 3. CLEAN ─────────────────────────────────────────────────────────────────
# Fill missing
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Replace NULL string
df.replace('NULL', np.nan, inplace=True)

# Remove rows with adr < 0 or no guests
df = df[df['adr'] >= 0]
df = df[(df['adults'] + df['children'] + df['babies']) > 0]

# Remove duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print("\nCleaned shape:", df.shape)

# ── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
season_map = {'December':'Winter','January':'Winter','February':'Winter',
              'March':'Spring','April':'Spring','May':'Spring',
              'June':'Summer','July':'Summer','August':'Summer',
              'September':'Autumn','October':'Autumn','November':'Autumn'}

df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['revenue'] = df['adr'] * df['total_nights']
df['season'] = df['arrival_date_month'].map(season_map)
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
df['room_match'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
df['has_special_request'] = (df['total_of_special_requests'] > 0).astype(int)
df['lead_time_bin'] = pd.cut(df['lead_time'], bins=[0,7,30,90,180,737],
                              labels=['0-7d','8-30d','31-90d','91-180d','180+d'])

print("\nEngineered features added.")
print(df[['total_nights','total_guests','revenue','season','room_match']].describe())

# ── 5. ENCODE CATEGORICALS ───────────────────────────────────────────────────
cat_cols = ['hotel','meal','market_segment','distribution_channel',
            'reserved_room_type','assigned_room_type','deposit_type',
            'customer_type','season']
le = LabelEncoder()
df_enc = df.copy()
for c in cat_cols:
    df_enc[c + '_enc'] = le.fit_transform(df_enc[c].astype(str))

print("\nEncoded columns:", [c+'_enc' for c in cat_cols])

# ── 6. SAVE CLEANED DATA ─────────────────────────────────────────────────────
df.to_csv(os.path.join(BASE, 'outputs', 'hotel_bookings_cleaned.csv'), index=False)
print("\nCleaned dataset saved.")

# ── 7. DESCRIPTIVE STATS ─────────────────────────────────────────────────────
print("\nDescriptive stats:\n", df[['lead_time','adr','total_nights','total_guests','revenue']].describe().round(2))
print("\nCancellation rate:", df['is_canceled'].mean().round(3))

# ── 8. VISUALISATIONS ────────────────────────────────────────────────────────
# 8a. Cancellation by channel
fig, ax = plt.subplots()
cancel_ch = df.groupby('distribution_channel')['is_canceled'].mean().sort_values(ascending=False)
cancel_ch.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Cancellation Rate by Distribution Channel')
ax.set_ylabel('Cancellation Rate')
ax.set_xlabel('')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os_path + 'fig1_cancel_by_channel.png', dpi=120)
plt.close()

# 8b. Cancellation by market segment
fig, ax = plt.subplots()
cancel_seg = df.groupby('market_segment')['is_canceled'].mean().sort_values(ascending=False)
cancel_seg.plot(kind='bar', ax=ax, color='coral', edgecolor='white')
ax.set_title('Cancellation Rate by Market Segment')
ax.set_ylabel('Cancellation Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os_path + 'fig2_cancel_by_segment.png', dpi=120)
plt.close()

# 8c. Lead time vs cancellation
fig, ax = plt.subplots()
cancel_lt = df.groupby('lead_time_bin')['is_canceled'].mean()
cancel_lt.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='white')
ax.set_title('Cancellation Rate by Lead Time Bucket')
ax.set_ylabel('Cancellation Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig3_cancel_by_leadtime.png', dpi=120)
plt.close()

# 8d. Lead time distribution (canceled vs not)
fig, ax = plt.subplots()
df[df['is_canceled']==0]['lead_time'].clip(upper=400).plot(kind='hist', bins=50, alpha=0.6, label='Not Canceled', ax=ax)
df[df['is_canceled']==1]['lead_time'].clip(upper=400).plot(kind='hist', bins=50, alpha=0.6, label='Canceled', ax=ax)
ax.set_title('Lead Time Distribution: Canceled vs Not Canceled')
ax.set_xlabel('Lead Time (days)')
ax.legend()
plt.tight_layout()
plt.savefig(os_path + 'fig4_leadtime_dist.png', dpi=120)
plt.close()

# 8e. Monthly occupancy (non-canceled)
fig, ax = plt.subplots()
occ = df[df['is_canceled']==0].groupby('arrival_date_month').size()
occ.plot(kind='bar', ax=ax, color='teal', edgecolor='white')
ax.set_title('Monthly Occupancy (Checked-In Bookings)')
ax.set_ylabel('Number of Bookings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os_path + 'fig5_monthly_occupancy.png', dpi=120)
plt.close()

# 8f. Seasonal occupancy
fig, ax = plt.subplots()
season_order = ['Spring','Summer','Autumn','Winter']
occ_s = df[df['is_canceled']==0].groupby('season').size().reindex(season_order)
occ_s.plot(kind='bar', ax=ax, color='goldenrod', edgecolor='white')
ax.set_title('Seasonal Occupancy')
ax.set_ylabel('Number of Bookings')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig6_seasonal_occupancy.png', dpi=120)
plt.close()

# 8g. Repeat guests cancellation
fig, ax = plt.subplots()
rg = df.groupby('is_repeated_guest')['is_canceled'].mean()
rg.index = ['New Guest', 'Repeat Guest']
rg.plot(kind='bar', ax=ax, color=['salmon','seagreen'], edgecolor='white')
ax.set_title('Cancellation Rate: New vs Repeat Guests')
ax.set_ylabel('Cancellation Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig7_repeat_guest_cancel.png', dpi=120)
plt.close()

# 8h. Room type performance (avg ADR, non-canceled)
fig, ax = plt.subplots()
rt = df[df['is_canceled']==0].groupby('assigned_room_type')['adr'].mean().sort_values(ascending=False)
rt.plot(kind='bar', ax=ax, color='dodgerblue', edgecolor='white')
ax.set_title('Average ADR by Assigned Room Type (Checked-In)')
ax.set_ylabel('Average Daily Rate (€)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig8_room_type_adr.png', dpi=120)
plt.close()

# 8i. Room type booking volume
fig, ax = plt.subplots()
rt_vol = df[df['is_canceled']==0].groupby('assigned_room_type').size().sort_values(ascending=False)
rt_vol.plot(kind='bar', ax=ax, color='slateblue', edgecolor='white')
ax.set_title('Booking Volume by Assigned Room Type (Checked-In)')
ax.set_ylabel('Number of Bookings')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig9_room_type_volume.png', dpi=120)
plt.close()

# 8j. ADR over months (heatmap hotel x month)
fig, ax = plt.subplots(figsize=(12, 4))
pivot = df[df['is_canceled']==0].pivot_table(values='adr', index='hotel',
            columns='arrival_date_month', aggfunc='mean')
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', fmt='.0f', annot=True, linewidths=0.5)
ax.set_title('Average ADR by Hotel and Month')
plt.tight_layout()
plt.savefig(os_path + 'fig10_adr_heatmap.png', dpi=120)
plt.close()

# 8k. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
num_cols = ['lead_time','adr','total_nights','total_guests','is_canceled',
            'previous_cancellations','booking_changes','total_of_special_requests',
            'days_in_waiting_list','required_car_parking_spaces','room_match']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os_path + 'fig11_correlation.png', dpi=120)
plt.close()

# 8l. Hotel type cancellation
fig, ax = plt.subplots()
df.groupby('hotel')['is_canceled'].mean().plot(kind='bar', ax=ax, color=['steelblue','coral'], edgecolor='white')
ax.set_title('Cancellation Rate by Hotel Type')
ax.set_ylabel('Cancellation Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os_path + 'fig12_cancel_by_hotel.png', dpi=120)
plt.close()

print("\nAll figures saved to outputs/")

# ── 9. FEATURE SELECTION ─────────────────────────────────────────────────────
feature_cols = [c for c in df_enc.columns if c.endswith('_enc')] + \
               ['lead_time','adr','total_nights','total_guests','room_match',
                'has_special_request','previous_cancellations','booking_changes',
                'days_in_waiting_list','required_car_parking_spaces',
                'total_of_special_requests','is_repeated_guest']

X = df_enc[feature_cols].fillna(0)
y = df_enc['is_canceled']

selector = SelectKBest(f_classif, k=10)
selector.fit(X, y)
scores = pd.Series(selector.scores_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features (ANOVA F-score):\n", scores.head(10).round(2))

fig, ax = plt.subplots()
scores.head(10).plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Top 10 Features by ANOVA F-Score')
ax.set_xlabel('F-Score')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os_path + 'fig13_feature_importance.png', dpi=120)
plt.close()

print("\nFeature selection figure saved.")
print("\n=== EDA COMPLETE ===")
