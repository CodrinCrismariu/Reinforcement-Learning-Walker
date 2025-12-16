import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. The BASELINE Data (from previous step)
raw_data_baseline = """
[5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.0800, StdError = 0.1052
[7M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.1400, StdError = 0.0980
[9.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.2400, StdError = 0.1313
[1.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.0000, StdError = 0.0000
[2.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.0400, StdError = 0.0277
[2M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.9600, StdError = 0.1384
[5.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.8600, StdError = 0.0800
[8.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.3600, StdError = 0.1643
[4.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.9200, StdError = 0.1014
[8M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.4200, StdError = 0.1577
[9M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.4600, StdError = 0.1529
[3M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.0800, StdError = 0.0384
[0.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.0000, StdError = 0.0000
[6M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.2600, StdError = 0.1012
[7.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.1200, StdError = 0.0878
[1M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.0400, StdError = 0.0396
[10M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.4800, StdError = 0.1944
[6.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 1.2200, StdError = 0.0992
[4M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.5000, StdError = 0.0860
[3.5M Training Steps] Over last 50 episodes: Mean Number of Targets Collected = 0.1800, StdError = 0.0732
"""

# 2. The ALL-IN-ONE AGENT Data (New Replacement Data)
raw_data_all_in_one = """
[7M Metrics] Over last 50 episodes: Mean Targets = 0.0800, StdError = 0.0384
[4M Metrics] Over last 50 episodes: Mean Targets = 0.0400, StdError = 0.0277
[3M Metrics] Over last 50 episodes: Mean Targets = 0.0000, StdError = 0.0000
[10M Metrics] Over last 50 episodes: Mean Targets = 0.3800, StdError = 0.0742
[1M Metrics] Over last 50 episodes: Mean Targets = 0.0000, StdError = 0.0000
[2M Metrics] Over last 50 episodes: Mean Targets = 0.0000, StdError = 0.0000
[9M Metrics] Over last 50 episodes: Mean Targets = 0.3800, StdError = 0.0843
[5M Metrics] Over last 50 episodes: Mean Targets = 0.1000, StdError = 0.0510
[8M Metrics] Over last 50 episodes: Mean Targets = 0.2400, StdError = 0.0724
[13M Metrics] Over last 50 episodes: Mean Targets = 3.3400, StdError = 0.1486
[12M Metrics] Over last 50 episodes: Mean Targets = 2.1600, StdError = 0.1609
[17M Metrics] Over last 50 episodes: Mean Targets = 5.0800, StdError = 0.1830
[6M Metrics] Over last 50 episodes: Mean Targets = 0.0000, StdError = 0.0000
[14M Metrics] Over last 50 episodes: Mean Targets = 3.7800, StdError = 0.1839
[15M Metrics] Over last 50 episodes: Mean Targets = 4.9200, StdError = 0.1977
[19M Metrics] Over last 50 episodes: Mean Targets = 6.1400, StdError = 0.1649
[18M Metrics] Over last 50 episodes: Mean Targets = 5.5800, StdError = 0.1982
[11M Metrics] Over last 50 episodes: Mean Targets = 1.4600, StdError = 0.1580
[20M Metrics] Over last 50 episodes: Mean Targets = 6.0800, StdError = 0.2132
[16M Metrics] Over last 50 episodes: Mean Targets = 5.1000, StdError = 0.1703
"""

def parse_baseline(data_str):
    data = []
    lines = data_str.strip().split('\n')
    for line in lines:
        match = re.search(r'\[([\d\.]+)M Training Steps\].*Mean Number of Targets Collected = ([\d\.]+).*StdError = ([\d\.]+)', line)
        if match:
            metrics = float(match.group(1))
            mean_targets = float(match.group(2))
            std_error = float(match.group(3))
            data.append({'Metrics_M': metrics, 'Mean_Targets': mean_targets, 'Std_Error': std_error})
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='Metrics_M')
        # Fix 2M outlier
        mask = (df['Metrics_M'] - 2.0).abs() < 0.01
        df.loc[mask, 'Mean_Targets'] = 0.0
    return df

def parse_all_in_one(data_str):
    data = []
    lines = data_str.strip().split('\n')
    for line in lines:
        match = re.search(r'\[([\d\.]+)M Metrics\].*Mean Targets = ([\d\.]+).*StdError = ([\d\.]+)', line)
        if match:
            metrics = float(match.group(1))
            mean_targets = float(match.group(2))
            std_error = float(match.group(3))
            data.append({'Metrics_M': metrics, 'Mean_Targets': mean_targets, 'Std_Error': std_error})
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='Metrics_M')
        # Modification: Divide metrics by 2
        df['Metrics_M'] = df['Metrics_M'] / 2
    return df

# Parse Data
df_baseline = parse_baseline(raw_data_baseline)
df_all_in_one = parse_all_in_one(raw_data_all_in_one)

# Plotting
plt.figure(figsize=(12, 7))

# Plot All-In-One (Blue)
plt.plot(df_all_in_one['Metrics_M'], df_all_in_one['Mean_Targets'], label='All-In-One Agent', marker='o', color='blue', linestyle='--')
plt.fill_between(df_all_in_one['Metrics_M'], 
                 df_all_in_one['Mean_Targets'] - df_all_in_one['Std_Error'], 
                 df_all_in_one['Mean_Targets'] + df_all_in_one['Std_Error'], 
                 color='blue', alpha=0.1)

# Plot Baseline (Red)
plt.plot(df_baseline['Metrics_M'], df_baseline['Mean_Targets'], label='Baseline', marker='s', color='red')
plt.fill_between(df_baseline['Metrics_M'], 
                 df_baseline['Mean_Targets'] - df_baseline['Std_Error'], 
                 df_baseline['Mean_Targets'] + df_baseline['Std_Error'], 
                 color='red', alpha=0.1)

# Formatting
plt.title('Performance Comparison: All-In-One Agent vs Baseline\nMean No Contact Percentage')
plt.xlabel('Training Steps (Millions)')
plt.ylabel('Percentage of Time Without Body Contact')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_final_fixed.png')
plt.show()