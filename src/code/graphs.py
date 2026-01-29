# -*- coding: utf-8 -*-
"""
@author: pco30
"""
# Import relevant libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to get the creation time of a file
def get_file_creation_time(file_path):
    return os.path.getctime(file_path)

# Function to extract files from a folder path
def extract_files(folder_path):
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return excel_files

# Iterate over each file path and alter the cycle index with addition of more files
def process_files(folder_path, sheet_names_to_import):
    file_paths = extract_files(folder_path)
    file_paths.sort(key=get_file_creation_time)
    dfs = []
    last_cycle_index = 0
    for file_path in file_paths:
        # Get sheet names from the Excel file
        sheet_names = pd.ExcelFile(file_path).sheet_names
        for sheet_name in sheet_names:
            if sheet_name in sheet_names_to_import:
                # Read the Excel file with the current sheet name
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df['Cycle_Index'] += last_cycle_index
                last_cycle_index = df['Cycle_Index'].max()
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Function to Calculate the State of health of the battery
def calculate_soh(df, c_rate):
    # Convert 'Test_Time(s)' to hours
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    # Identify the start of each discharge cycle
    discharge_start = (df['Current(A)'] < 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_discharge_start_time = df.loc[discharge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_discharge_start_time_series = df['Cycle_Index'].map(first_discharge_start_time)
    # Calculate test time discharged for each row
    df['Test_Time_Discharged(h)'] = df['Test_Time(h)'] - first_discharge_start_time_series
    # Calculate battery capacity
    df['Battery_Capacity_Per_Point(mAh)'] = df['Test_Time_Discharged(h)'] * abs(df['Current(A)']) * 1000
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].transform('max')
    max_soh = df['Battery_Capacity(mAh)'].max()
    # Calculate the SOH values by dividing each battery capacity by the maximum value and multiplying by 100
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_soh) * 100
    df['C_rate'] = c_rate
    # Identify the start of each charge cycle
    charge_start = (df['Current(A)'] >= 0) & (df['Cycle_Index'] == df['Cycle_Index'].shift())
    first_charge_start_time = df.loc[charge_start, ['Cycle_Index', 'Test_Time(h)']].groupby('Cycle_Index')['Test_Time(h)'].first()
    first_charge_start_time_series = df['Cycle_Index'].map(first_charge_start_time)
    # Calculate test time charged for each row
    df['Test_Time_charged(h)'] = df['Test_Time(h)'] - first_charge_start_time_series
    return df

# Initial function to plot the SOH against the number of cycles
def plot_soh(df):
    SOH = df.groupby('Cycle_Index')['Battery_Capacity_Per_Point(mAh)'].max().tolist()
    max_soh = max(SOH)
    Actual_soh = [(value / max_soh) * 100 for value in SOH]
    x = df['Cycle_Index'].unique()
    y = Actual_soh
    plt.xlabel('Cycles')
    plt.ylabel('Battery State of Health (%)')
    plt.plot(x, y)
    plt.show()

# Function to create a heatmap correlation graph
def plot_heatmap(df, discharge_boundary):
    discharge_data = df[df['Current(A)'] < -(discharge_boundary)]
    features = ['Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)',
                'Internal_Resistance(Ohm)', 'Test_Time_Discharged(h)', 'Battery_Capacity(mAh)', 'Calculated_SOH(%)', 'C_rate']#, 'C_rate']
    X = pd.get_dummies(discharge_data[features], drop_first=True)
    corr_matrix = X.corr()
    corr_target = corr_matrix[['Calculated_SOH(%)']].drop(labels=['Calculated_SOH(%)'])
    # plt.figure()
    sns.heatmap(corr_target, annot=True, cmap='RdBu_r')
    plt.show()
    plt.close()

# Function for selecting valid features (necessary to visualize features)
def selection(df, c_rate):
    # Filter rows where 'Test_Time_Discharged(h)' is greater than 0 and assign 0 to 'Test_Time_charged(h)'
    df.loc[df['Test_Time_Discharged(h)'] > 0, 'Test_Time_charged(h)'] = 0
    
    # Select specific columns
    df = df[['Cycle_Index', 'Internal_Resistance(Ohm)', 'Voltage(V)', 
              'Test_Time_Discharged(h)', 'Test_Time_charged(h)', 'Calculated_SOH(%)', 'Battery_Capacity(mAh)',
              'Charge_Capacity(Ah)', 'Charge_Energy(Wh)', 'dV/dt(V/s)', 'Step_Time(s)']]
    
    # Group by cycle
    cycle_groups = df.groupby('Cycle_Index')

    # Initialize the new column in the original DataFrame
    df['avg_abs_dvdt'] = np.nan
    
    # Iterate through each cycle group
    for cycle, group in cycle_groups:
        # Calculate the average absolute dV/dt for the current cycle
        dvdt = group['dV/dt(V/s)'].dropna()
        if len(dvdt) > 0:
            avg_abs_dvdt_val = np.mean(np.abs(dvdt))

            # Update the 'avg_abs_dvdt' column for rows belonging to the current cycle
            df.loc[df['Cycle_Index'] == cycle, 'avg_abs_dvdt'] = avg_abs_dvdt_val

    # Group the DataFrame by 'Cycle_Index' and apply aggregation functions
    df = df.groupby('Cycle_Index').agg({
        'Internal_Resistance(Ohm)': 'max',
        'Battery_Capacity(mAh)': 'max',
        'Step_Time(s)': 'max',
        'dV/dt(V/s)': 'mean',
        'Charge_Energy(Wh)': 'mean',
        'Charge_Capacity(Ah)': 'mean',
        'Voltage(V)': 'mean',
        'Test_Time_Discharged(h)': 'max',
        'Test_Time_charged(h)': lambda x: x.max() - x.min(), # Corrected code for the calculation
        'Calculated_SOH(%)' : 'max',
        'avg_abs_dvdt': 'max'
    }).reset_index()    
    # Dropping rows with missing or NaN values
    df.dropna(inplace=True)
    df['C_rate'] = c_rate
    return df


folder_path_1 = "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35"
sheet_name_1 = ["Channel_1-008"]
CS2_35 = process_files(folder_path_1, sheet_name_1)
print(CS2_35.columns)
print(CS2_35.dtypes)
CS2_35 = calculate_soh(CS2_35, 1)
discharge = CS2_35[CS2_35['Current(A)'] < -0.5]
discharge.reset_index(inplace = True)
#CS2_35  = selection(CS2_35,1)
# plot_soh(CS2_35)
# plot_heatmap(CS2_35, 0.5)

plt.figure(figsize=(10, 6))
plt.plot(discharge['Cycle_Index'], discharge['Internal_Resistance(Ohm)'])
plt.xlabel('Cycles)')
plt.ylabel('Internal_Resistance(Ohm)')
plt.title('Incremental Increase of the Internal Resistance with multiple Discharge Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Get data for specific cycle indices (1st, 20th, 60th, and 200th)
cycle_indices = [1, 20, 60, 200, 300, 400, 500, 800]
cycle_labels = ['1', '20', '60', '200', '300', '400', '500', '800']

plt.figure(figsize=(10, 6))
for cycle_index, label in zip(cycle_indices, cycle_labels):
    cycle_data = discharge[discharge['Cycle_Index'] == cycle_index]
    plt.plot(cycle_data['Test_Time_Discharged(h)'], cycle_data['Voltage(V)'], label=f'Cycle {label}')

plt.xlabel('Test Time(h)')
plt.ylabel('Voltage(h)')
plt.title('Voltage vs Test Time for Specific Cycle Indices in the Discharge Cycle')
plt.legend()
plt.grid(True)
plt.show()

charge = CS2_35[CS2_35['Test_Time_Discharged(h)'] < 0]
charge.reset_index(inplace = True)

cycle_indice = [1, 20, 60, 200, 300, 400, 500, 800]
cycle_label = ['1', '20', '60', '200', '300', '400', '500', '800']

plt.figure(figsize=(10, 6))
for cycle_index, label in zip(cycle_indice, cycle_label):
    cycle_data = charge[charge['Cycle_Index'] == cycle_index]
    plt.plot(cycle_data['Test_Time_charged(h)'], cycle_data['Voltage(V)'], label=f'Cycle {label}')

plt.xlabel('Test Time(h)')
plt.ylabel('Voltage(h)')
# plt.title('Voltage vs Test Time for Specific Cycle Indices in the charge Cycle')
plt.legend()
plt.grid(True)
plt.show()


# Group by cycle
cycle_groups = CS2_35.groupby('Cycle_Index')

soh = []
avg_abs_dvdt = []

for cycle, group in cycle_groups:
    dvdt = group['dV/dt(V/s)'].dropna()
    soh_val = group['Calculated_SOH(%)'].mean()

    if len(dvdt) > 0:
        soh.append(soh_val)
        avg_abs_dvdt.append(np.mean(np.abs(dvdt)))

# Plotting
plt.figure(figsize=(10,6))
plt.plot(soh, avg_abs_dvdt, 'o')
plt.xlabel('State of Health (%)')
plt.ylabel('Avg |dV/dt| (V/s)')
plt.title('Relationship between |dV/dt| and SoH')
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10,6))
plt.plot(CS2_35["Calculated_SOH(%)"], CS2_35['avg_abs_dvdt'], 'o')
plt.xlabel('State of Health (%)')
plt.ylabel('Avg |dV/dt| (V/s)')
plt.title('Relationship between |dV/dt| and SoH')
plt.grid(True)
plt.show()

### Reminder: Everything below can be copy-pasted after running model.py
# SOH CURVES against cycle index for all batteries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# #  Clean random SOH drops (per battery) 
# def clean_soh_drops_per_battery(
#     df,
#     battery_col="battery_id",
#     cycle_col="Cycle_Index",
#     soh_col="Calculated_SOH(%)",
#     jump_thresh=5.0,          # how big a jump counts as "spiky"
#     window=5,                 # rolling median window 
#     replace_with="rolling_median"  
# ):
#     d = df.copy()
#     d[cycle_col] = pd.to_numeric(d[cycle_col], errors="coerce")
#     d[soh_col] = pd.to_numeric(d[soh_col], errors="coerce")
#     d = d.dropna(subset=[battery_col, cycle_col, soh_col]).sort_values([battery_col, cycle_col])

#     d["SOH_clean"] = d[soh_col]

#     # identify spikes: big change vs prev AND vs next (your earlier logic), then replace
#     for bid, g in d.groupby(battery_col, sort=False):
#         idx = g.index
#         s = g[soh_col].astype(float)

#         prev_diff = (s - s.shift(1)).abs()
#         next_diff = (s - s.shift(-1)).abs()
#         spike = (prev_diff > jump_thresh) & (next_diff > jump_thresh)

#         if replace_with == "nan_drop":
#             d.loc[idx[spike], "SOH_clean"] = np.nan
#             continue

#         # rolling median as replacement (robust smoother)
#         med = s.rolling(window=window, center=True, min_periods=1).median()
#         d.loc[idx[spike], "SOH_clean"] = med.loc[idx[spike]].to_numpy()

#     return d

# cleaned = clean_soh_drops_per_battery(
#     combined,
#     jump_thresh=5.0,     # tweak if needed (e.g., 3.0 or 8.0)
#     window=7,            # smoother replacement
#     replace_with="rolling_median"
# )

# # Replot (all batteries on one chart)
# plt.figure(figsize=(12, 7))

# for bid, g in cleaned.groupby("battery_id", sort=True):
#     g = g.sort_values("Cycle_Index")
#     plt.plot(
#         g["Cycle_Index"].to_numpy(),
#         g["SOH_clean"].to_numpy(),
#         label=str(bid),
#         linewidth=1.5
#     )

# plt.xlabel("Cycle Index")
# plt.ylabel("Calculated SOH (%)")
# plt.title("Cleaned SOH vs Cycle Index (All Batteries)")
# plt.grid(True, alpha=0.3)
# plt.legend(title="battery_id", bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()
# plt.show()


##############################################
# ## DECREASING FEATURE SET PLOT
# import numpy as np
# import matplotlib.pyplot as plt

# # ---- Your values (from the plot) ----
# stages = ["All\nfeatures", "All +\nprune", "Final\nshortlist", "Stability\nselected"]
# mean_mse = np.array([0.379, 1.027, 0.231, 1.388])
# k_text = ["k≈108", "k≈79", "k≈68", "k≈9"]

# # Publication defaults 
# plt.rcParams.update({
#     "font.size": 15,
#     "axes.titlesize": 15,
#     "axes.labelsize": 15,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "figure.dpi": 300,
# })

# x = np.arange(len(stages))

# -
# fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

# # Line plot
# ax.plot(x, mean_mse, marker="o", linewidth=1.6, markersize=4.5)

# # Axes labels
# ax.set_xticks(x)
# ax.set_xticklabels(stages)
# ax.set_ylabel("Mean holdout MSE")
# ax.set_xlabel("Decreasing feature set")

# # Clean look
# ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)
# ax.grid(False, axis="x")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # Y-limits with headroom for annotations
# ax.set_ylim(0, mean_mse.max() * 1.18)

# #  Manual annotation offsets (dx, dy) in points 
# # Tuned individually for each stage
# offsets = [
#     (0,-35),    # All features
#     (0, 10),    # All + prune
#     (0, -40),    # Final shortlist
#     (-5, 12),   # Stability selected
# ]

# #  Annotations 
# for i, (m, k) in enumerate(zip(mean_mse, k_text)):
#     dx, dy = offsets[i]

#     ax.annotate(
#         f"{k}\n{m:.3f}",
#         (x[i], m),
#         textcoords="offset points",
#         xytext=(dx, dy),
#         ha="left" if dx >= 0 else "right",
#         va="bottom",
#         clip_on=True
#     )

# plt.show()

