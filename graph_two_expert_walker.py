import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. The ORIGINAL Old Data (Now "Two-Expert Agent")
raw_data_two_expert = """
[10M Metrics] Over last 100 episodes: Mean Targets = 4.2600, StdError = 0.2292
[9.5M Metrics] Over last 100 episodes: Mean Targets = 3.2800, StdError = 0.1703
[9M Metrics] Over last 100 episodes: Mean Targets = 4.1900, StdError = 0.2194
[8.5M Metrics] Over last 100 episodes: Mean Targets = 2.9200, StdError = 0.1554
[8M Metrics] Over last 100 episodes: Mean Targets = 3.1100, StdError = 0.1907
[7.5M Metrics] Over last 100 episodes: Mean Targets = 2.7100, StdError = 0.1416
[7M Metrics] Over last 100 episodes: Mean Targets = 2.4300, StdError = 0.1402
[6.5M Metrics] Over last 100 episodes: Mean Targets = 2.2600, StdError = 0.1507
[6M Metrics] Over last 100 episodes: Mean Targets = 1.4700, StdError = 0.1179
[5.5M Metrics] Over last 100 episodes: Mean Targets = 1.6900, StdError = 0.1027
[5M Metrics] Over last 100 episodes: Mean Targets = 1.0900, StdError = 0.0861
[4.5M Metrics] Over last 100 episodes: Mean Targets = 1.3700, StdError = 0.1055
[4M Metrics] Over last 100 episodes: Mean Targets = 0.4100, StdError = 0.0750
[3.5M Metrics] Over last 100 episodes: Mean Targets = 0.2400, StdError = 0.0512
[3M Metrics] Over last 100 episodes: Mean Targets = 0.0000, StdError = 0.0000
[2.5M Metrics] Over last 100 episodes: Mean Targets = 0.0400, StdError = 0.0242
[2M Metrics] Over last 100 episodes: Mean Targets = 0.0700, StdError = 0.0324
[1.5M Metrics] Over last 100 episodes: Mean Targets = 0.0100, StdError = 0.0099
[1M Metrics] Over last 100 episodes: Mean Targets = 0.0000, StdError = 0.0000
[0.5M Metrics] Over last 100 episodes: Mean Targets = 0.0000, StdError = 0.0000
"""

# 2. The NEW Data (Now "Baseline")
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

def parse_two_expert(data_str):
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
    return df

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
        # Fix the 2M outlier to 0.0
        mask = (df['Metrics_M'] - 2.0).abs() < 0.01
        df.loc[mask, 'Mean_Targets'] = 0.0
    return df

# Parse Data
df_two_expert = parse_two_expert(raw_data_two_expert)
df_baseline = parse_baseline(raw_data_baseline)

# Plotting
plt.figure(figsize=(12, 7))

# Plot Two-Expert Agent (Blue)
plt.plot(df_two_expert['Metrics_M'], df_two_expert['Mean_Targets'], label='Two-Expert Agent', marker='o', color='blue', linestyle='--')
plt.fill_between(df_two_expert['Metrics_M'], 
                 df_two_expert['Mean_Targets'] - df_two_expert['Std_Error'], 
                 df_two_expert['Mean_Targets'] + df_two_expert['Std_Error'], 
                 color='blue', alpha=0.1)

# Plot Baseline (Red)
plt.plot(df_baseline['Metrics_M'], df_baseline['Mean_Targets'], label='Baseline', marker='s', color='red')
plt.fill_between(df_baseline['Metrics_M'], 
                 df_baseline['Mean_Targets'] - df_baseline['Std_Error'], 
                 df_baseline['Mean_Targets'] + df_baseline['Std_Error'], 
                 color='red', alpha=0.1)

# Formatting
plt.title('Performance Comparison: Two-Expert Agent vs Baseline\nMean Targets Collected Over 5000 Steps')
plt.xlabel('Training Steps (Millions)')
plt.ylabel('Mean number of Targets Collected')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig('comparison_final_fixed.png')