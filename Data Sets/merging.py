# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# print("CLIMATE COLUMNS:")
# print(climate.columns.tolist())
# print(climate.head(3))

# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")
# print("\nWILDFIRE COLUMNS:")
# print(wildfire.columns.tolist())
# print(wildfire.head(3))

# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# print("CLIMATE REGIONS:")
# print(sorted(climate['Region'].unique()))

# print("\nWILDFIRE FIRE CENTRES:")
# print(sorted(wildfire['FRCNTR'].dropna().unique()))
# import pandas as pd

# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# # Look at fire number prefix vs fire centre vs location
# wildfire['CENTRE_LETTER'] = wildfire['FIRE_NO'].str[0]

# print(wildfire[['FIRE_NO', 'FRCNTR', 'CENTRE_LETTER', 'GEO_DESC', 'LATITUDE', 'LONGITUDE']].drop_duplicates('FRCNTR').sort_values('FRCNTR'))
# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# # Map fire centre numbers to climate region names
# frcntr_to_region = {
#     2.0: 'Vancouver Island',
#     3.0: 'Prince George',
#     4.0: 'Prince George',
#     5.0: 'Kamloops',
#     6.0: 'Okanagan',
#     7.0: 'Cariboo'
# }

# wildfire['Region'] = wildfire['FRCNTR'].map(frcntr_to_region)

# # Drop rows where region couldn't be mapped
# wildfire = wildfire.dropna(subset=['Region'])

# print(wildfire['Region'].value_counts())
# print(f"Unmapped rows dropped: {wildfire['Region'].isna().sum()}")
# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# # Map fire centres to regions
# frcntr_to_region = {
#     2.0: 'Vancouver Island',
#     3.0: 'Prince George',
#     4.0: 'Prince George',
#     5.0: 'Kamloops',
#     6.0: 'Okanagan',
#     7.0: 'Cariboo'
# }
# wildfire['Region'] = wildfire['FRCNTR'].map(frcntr_to_region)
# wildfire = wildfire.dropna(subset=['Region'])

# # Create risk labels from fire size
# def assign_risk(size):
#     if size < 0.1:
#         return 'Low'
#     elif size < 10:
#         return 'Moderate'
#     elif size < 1000:
#         return 'High'
#     else:
#         return 'Extreme'

# wildfire['RISK_LABEL'] = wildfire['SIZE_HA'].apply(assign_risk)

# # Extract month and season from ignition date
# wildfire['IGN_DATE'] = pd.to_datetime(wildfire['IGN_DATE'], errors='coerce')
# wildfire['Month'] = wildfire['IGN_DATE'].dt.month

# def get_season(month):
#     if month in [12, 1, 2]: return 'Winter'
#     elif month in [3, 4, 5]: return 'Spring'
#     elif month in [6, 7, 8]: return 'Summer'
#     else: return 'Fall'

# wildfire['Season'] = wildfire['Month'].apply(get_season)

# # Aggregate wildfire by region + year + season
# fire_grouped = wildfire.groupby(['Region', 'FIRE_YEAR', 'Season']).agg(
#     num_fires=('FIRE_NO', 'count'),
#     avg_size_ha=('SIZE_HA', 'mean'),
#     dominant_risk=('RISK_LABEL', lambda x: x.mode()[0])
# ).reset_index()
# fire_grouped.rename(columns={'FIRE_YEAR': 'Year'}, inplace=True)

# # Merge with climate data
# merged = pd.merge(climate, fire_grouped, on=['Year', 'Region', 'Season'], how='inner')

# print(f"Merged shape: {merged.shape}")
# print(merged.columns.tolist())
# print(merged.head())

# # Save
# merged.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
# print("Saved!")

# import pandas as pd

# df = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv")

# print("Missing values per column:")
# print(df.isnull().sum())
# print(f"\nTotal rows: {len(df)}")

# import pandas as pd

# df = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv")

# # Fill missing avg_size_ha with mean for that region
# df['avg_size_ha'] = df.groupby('Region')['avg_size_ha'].transform(lambda x: x.fillna(x.mean()))

# print("Missing values after fix:")
# print(df.isnull().sum())

# # Save updated dataset
# df.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
# print("Saved!")



# import pandas as pd

# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# # Convert IGN_DATE properly
# wildfire['IGN_DATE'] = pd.to_datetime(wildfire['IGN_DATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
# wildfire['Month'] = wildfire['IGN_DATE'].dt.month

# def get_season(month):
#     if month in [12, 1, 2]: return 'Winter'
#     elif month in [3, 4, 5]: return 'Spring'
#     elif month in [6, 7, 8]: return 'Summer'
#     else: return 'Fall'

# wildfire['Season'] = wildfire['Month'].apply(get_season)

# print("Seasons in wildfire data:")
# print(wildfire['Season'].value_counts())

# print("\nSample IGN_DATE values:")
# print(wildfire['IGN_DATE'].head(10))

# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# wildfire = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")

# # Fix date and extract season
# wildfire['IGN_DATE'] = pd.to_datetime(wildfire['IGN_DATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
# wildfire['Month'] = wildfire['IGN_DATE'].dt.month

# def get_season(month):
#     if month in [12, 1, 2]: return 'Winter'
#     elif month in [3, 4, 5]: return 'Spring'
#     elif month in [6, 7, 8]: return 'Summer'
#     else: return 'Fall'

# wildfire['Season'] = wildfire['Month'].apply(get_season)

# # Map fire centres to regions
# frcntr_to_region = {
#     2.0: 'Vancouver Island',
#     3.0: 'Prince George',
#     4.0: 'Prince George',
#     5.0: 'Kamloops',
#     6.0: 'Okanagan',
#     7.0: 'Cariboo'
# }
# wildfire['Region'] = wildfire['FRCNTR'].map(frcntr_to_region)
# wildfire = wildfire.dropna(subset=['Region'])

# # Create risk labels
# def assign_risk(size):
#     if size < 0.1: return 'Low'
#     elif size < 10: return 'Moderate'
#     elif size < 1000: return 'High'
#     else: return 'Extreme'

# wildfire['RISK_LABEL'] = wildfire['SIZE_HA'].apply(assign_risk)

# # Aggregate by region + year + season
# fire_grouped = wildfire.groupby(['Region', 'FIRE_YEAR', 'Season']).agg(
#     num_fires=('FIRE_NO', 'count'),
#     avg_size_ha=('SIZE_HA', 'mean'),
#     dominant_risk=('RISK_LABEL', lambda x: x.mode()[0])
# ).reset_index()
# fire_grouped.rename(columns={'FIRE_YEAR': 'Year'}, inplace=True)

# # Merge with climate
# merged = pd.merge(climate, fire_grouped, on=['Year', 'Region', 'Season'], how='inner')

# # Fill missing avg_size_ha
# merged['avg_size_ha'] = merged.groupby('Region')['avg_size_ha'].transform(lambda x: x.fillna(x.mean()))

# print(f"Merged shape: {merged.shape}")
# print(merged['Season'].value_counts())
# print(merged['Year'].min(), "to", merged['Year'].max())

# # Save
# merged.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
# print("Saved!")

# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# historical = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")
# current = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire\wildfire.csv")

# # Rename FIRE_CENTR to FRCNTR so they match
# current = current.rename(columns={'FIRE_CENTR': 'FRCNTR'})

# # Keep only columns that exist in both
# common_cols = ['FIRE_NO', 'FIRE_YEAR', 'IGN_DATE', 'FIRE_CAUSE', 'FRCNTR', 'ZONE', 'FIRE_ID', 'FIRE_TYPE', 'GEO_DESC', 'LATITUDE', 'LONGITUDE', 'SIZE_HA']
# historical = historical[common_cols]
# current = current[common_cols]

# # Combine both
# wildfire = pd.concat([historical, current], ignore_index=True)
# print(f"Total wildfire rows: {len(wildfire)}")
# print(wildfire['FIRE_YEAR'].value_counts().sort_index().tail(5))

# import pandas as pd

# climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
# historical = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")
# current = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire\wildfire.csv")

# # Rename and combine
# current = current.rename(columns={'FIRE_CENTR': 'FRCNTR'})
# common_cols = ['FIRE_NO', 'FIRE_YEAR', 'IGN_DATE', 'FIRE_CAUSE', 'FRCNTR', 'ZONE', 'FIRE_ID', 'FIRE_TYPE', 'GEO_DESC', 'LATITUDE', 'LONGITUDE', 'SIZE_HA']
# wildfire = pd.concat([historical[common_cols], current[common_cols]], ignore_index=True)

# # Fix dates
# wildfire['IGN_DATE'] = pd.to_datetime(wildfire['IGN_DATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
# wildfire['Month'] = wildfire['IGN_DATE'].dt.month

# def get_season(month):
#     if month in [12, 1, 2]: return 'Winter'
#     elif month in [3, 4, 5]: return 'Spring'
#     elif month in [6, 7, 8]: return 'Summer'
#     else: return 'Fall'

# wildfire['Season'] = wildfire['Month'].apply(get_season)

# # Map fire centres to regions
# frcntr_to_region = {
#     2.0: 'Vancouver Island',
#     3.0: 'Prince George',
#     4.0: 'Prince George',
#     5.0: 'Kamloops',
#     6.0: 'Okanagan',
#     7.0: 'Cariboo'
# }
# wildfire['Region'] = wildfire['FRCNTR'].map(frcntr_to_region)
# wildfire = wildfire.dropna(subset=['Region'])

# # Create risk labels
# def assign_risk(size):
#     if size < 0.1: return 'Low'
#     elif size < 10: return 'Moderate'
#     elif size < 1000: return 'High'
#     else: return 'Extreme'

# wildfire['RISK_LABEL'] = wildfire['SIZE_HA'].apply(assign_risk)

# # Aggregate by region + year + season
# fire_grouped = wildfire.groupby(['Region', 'FIRE_YEAR', 'Season']).agg(
#     num_fires=('FIRE_NO', 'count'),
#     avg_size_ha=('SIZE_HA', 'mean'),
#     dominant_risk=('RISK_LABEL', lambda x: x.mode()[0])
# ).reset_index()
# fire_grouped.rename(columns={'FIRE_YEAR': 'Year'}, inplace=True)

# # Merge with climate
# merged = pd.merge(climate, fire_grouped, on=['Year', 'Region', 'Season'], how='inner')

# # Fill missing avg_size_ha
# merged['avg_size_ha'] = merged.groupby('Region')['avg_size_ha'].transform(lambda x: x.fillna(x.mean()))

# print(f"Merged shape: {merged.shape}")
# print(merged['Season'].value_counts())
# print(merged['Year'].min(), "to", merged['Year'].max())
# print(merged.isnull().sum())

# # Save
# merged.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
# print("Saved!")



# NEW CODE
import pandas as pd

climate = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\climate data merge.xlsx")
historical = pd.read_excel(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire_historical merge.xlsx")
current = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\wildfire\wildfire.csv")

# Rename and combine
current = current.rename(columns={'FIRE_CENTR': 'FRCNTR'})
common_cols = ['FIRE_NO', 'FIRE_YEAR', 'IGN_DATE', 'FIRE_CAUSE', 'FRCNTR', 'SIZE_HA']
wildfire = pd.concat([historical[common_cols], current[common_cols]], ignore_index=True)

# Fix dates
wildfire['IGN_DATE'] = pd.to_datetime(wildfire['IGN_DATE'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
wildfire['Month'] = wildfire['IGN_DATE'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

wildfire['Season'] = wildfire['Month'].apply(get_season)

# Map fire centres to regions
frcntr_to_region = {
    2.0: 'Vancouver Island',
    3.0: 'Prince George',
    4.0: 'Prince George',
    5.0: 'Kamloops',
    6.0: 'Okanagan',
    7.0: 'Cariboo'
}
wildfire['Region'] = wildfire['FRCNTR'].map(frcntr_to_region)
wildfire = wildfire.dropna(subset=['Region'])
wildfire = wildfire.rename(columns={'FIRE_YEAR': 'Year'})

# Create risk label per individual fire
def assign_risk(size):
    if size < 0.1: return 'Low'
    elif size < 10: return 'Moderate'
    elif size < 1000: return 'High'
    else: return 'Extreme'

wildfire['RISK_LABEL'] = wildfire['SIZE_HA'].apply(assign_risk)

# Merge on Year + Region + Month (exact month match)
merged = pd.merge(wildfire, climate, on=['Year', 'Region', 'Month'], how='inner')

print(f"Merged shape: {merged.shape}")
print(merged['RISK_LABEL'].value_counts())
print(merged.isnull().sum())

# Save
merged.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
print("Saved!")

df = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv")

# Drop duplicate season column and unnecessary columns
df = df.drop(columns=['Season_y', 'FRCNTR'])

# Rename Season_x to Season
df = df.rename(columns={'Season_x': 'Season'})

print(df.shape)
print(df.columns.tolist())
print(df['RISK_LABEL'].value_counts())

df.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
print("Saved!")

df = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv")

# Drop rows with missing SIZE_HA
df = df.dropna(subset=['SIZE_HA'])

print(f"Rows after dropping missing SIZE_HA: {len(df)}")
print(df['RISK_LABEL'].value_counts())
print(df.isnull().sum())

df.to_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv", index=False)
print("Saved!")


df = pd.read_csv(r"C:\Users\Eluma\OneDrive\Desktop\Desktop\SFU\Spring 2026\CMPT310\proj\merged_dataset.csv")

print(f"Total rows: {len(df)}")
print(f"\nRisk label distribution:")
print(df['RISK_LABEL'].value_counts())
print(f"\nYear range: {df['Year'].min()} to {df['Year'].max()}")
print(f"\nRegions: {df['Region'].unique()}")
print(f"\nMissing values:")
print(df.isnull().sum())