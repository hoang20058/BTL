import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import math

# ========== Thiết lập thư mục lưu kết quả ==========
BASE_DIR = os.path.join('REPORT_BTL', 'BTL2_File')  # Thay đổi đường dẫn đến thư mục mong muốn
os.makedirs(BASE_DIR, exist_ok=True)

# ========== Đọc và xử lý dữ liệu từ file CSV ==========
def read_and_process_data(file_path):
    df = pd.read_csv(file_path, header=2)
    df.replace('N/a', pd.NA, inplace=True)

    cols_to_exclude = ['Name', 'Age', 'Team', 'Nation', 'Position']
    cols_to_convert = [col for col in df.columns if col not in cols_to_exclude]

    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Minutes'] = df['Minutes'].replace({',': ''}, regex=True)
    df['Minutes'] = pd.to_numeric(df['Minutes'], errors='coerce')
    return df

# ========== Tính toán Top 3 và Bottom 3 ==========
def get_top_and_bottom_3(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    rows = []

    for col in numeric_df.columns:
        if numeric_df[col].nunique() < 3:
            continue

        # Top 3
        for i, idx in enumerate(numeric_df[col].nlargest(3).index, 1):
            name = df.loc[idx, 'Name']
            value = df.loc[idx, col]
            rows.append((col, f'Top {i}', name, value))

        # Bottom 3
        for i, idx in enumerate(numeric_df[col].nsmallest(3).index, 1):
            name = df.loc[idx, 'Name']
            value = df.loc[idx, col]
            rows.append((col, f'Bot {i}', name, value))

    result_df = pd.DataFrame(rows, columns=['Name_col', 'Rank', 'Player Name', 'Value'])
    return result_df

# ========== Lưu kết quả Top 3 và Bottom 3 ==========
def save_top_bottom_3(result_df):
    folder = os.path.join(BASE_DIR, 'Top_Bottom')
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, 'top_3.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        for stat in result_df['Name_col'].unique():
            f.write(f"\n==============\n")
            stat_df = result_df[result_df['Name_col'] == stat]
            f.write(stat_df.to_string(index=False))
            f.write('\n')
    print(f"✅ Saved Top/Bottom 3 to {filepath}")

# ========== Tính toán Mean, Median, Std theo đội và toàn giải ==========
def calculate_statistics(df):
    folder = os.path.join(BASE_DIR, 'Statistics')
    os.makedirs(folder, exist_ok=True)

    def smart_float(val):
        if pd.isna(val):
            return 'N/a'
        return int(val) if val == int(val) else round(val, 8)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    teams = sorted(df['Team'].unique(), key=lambda x: x[0])

    results = []

    # Toàn giải
    row = {'Team': 'All'}
    for col in numeric_df.columns:
        row[f'Mean of {col}'] = smart_float(df[col].mean())
        row[f'Median of {col}'] = smart_float(df[col].median())
        row[f'Std of {col}'] = smart_float(df[col].std())
    results.append(row)

    # Theo đội
    for team in teams:
        team_df = df[df['Team'] == team]
        row = {'Team': team}
        for col in numeric_df.columns:
            row[f'Mean of {col}'] = smart_float(team_df[col].mean())
            row[f'Median of {col}'] = smart_float(team_df[col].median())
            row[f'Std of {col}'] = smart_float(team_df[col].std())
        results.append(row)

    results_df = pd.DataFrame(results)
    ordered_columns = ['Team'] + [f'{stat} of {col}' for col in numeric_df.columns for stat in ['Mean', 'Median', 'Std']]
    results_df = results_df[ordered_columns]

    filepath = os.path.join(folder, 'result_statistics.csv')
    results_df.to_csv(filepath, index=False)
    print(f"✅ Saved statistics to {filepath}")

# ========== Vẽ và lưu biểu đồ Histogram ==========
def plot_histograms(df, attack_stats, defense_stats):
    folder = os.path.join(BASE_DIR, 'Histograms')
    os.makedirs(folder, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    def plot_his(data, stats, title, prefix, check):
        for stat in stats:
            plt.figure(figsize=(10, 6))
            if check:
                sns.histplot(data[stat].dropna(), bins=30, kde=True , color='skyblue')
            else:
                sns.histplot(data[stat].dropna(), bins=30, kde=True , color='red')      
            plt.title(f'{title} of {stat}')
            plt.xlabel(stat)
            plt.ylabel('Frequency')
            plt.tight_layout()
            filename = f"{prefix}_{stat}.png"
            plt.savefig(os.path.join(folder, filename))
            plt.close()

    def plot_stat_per_team_subplots(df, stats, title_prefix, prefix, check):
        for stat in stats:
            teams = df['Team'].dropna().unique()
            num_teams = len(teams)
            cols = 5 # số cột subplot
            rows = math.ceil(num_teams / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
            axes = axes.flatten()

            for i, team in enumerate(teams):
                ax = axes[i]
                team_data = df[df['Team'] == team][stat].dropna()
                if check:  sns.histplot(team_data, bins=20, kde=True, color='skyblue', ax=ax)
                else: sns.histplot(team_data, bins=20, kde=True, color='red', ax=ax)
                ax.set_title(team)
                ax.set_xlabel(stat)
                ax.set_ylabel('Frequency')

            # Ẩn các subplot còn dư
            for j in range(len(teams), len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f'{title_prefix} - {stat}', fontsize=16)
            filename = f"{prefix}_teams_grid_{stat}.png"
            plt.savefig(os.path.join(folder, filename))
            plt.close()

    # Plot tổng cho tất cả cầu thủ
    plot_his(df, attack_stats, 'Attack Stats for All Players', 'All_Att', True)
    plot_his(df, defense_stats, 'Defense Stats for All Players', 'All_Def', False)

    # Plot gộp tất cả đội vào 1 ảnh cho mỗi chỉ số
    plot_stat_per_team_subplots(df, attack_stats, 'Attack Stats per Team', 'All_Teams_Attack', True)
    plot_stat_per_team_subplots(df, defense_stats, 'Defense Stats per Team', 'All_Teams_Defense', False)

    print(f"✅ Saved histograms to {folder}")
# ========== Tìm đội bóng tốt nhất ==========
def find_best_team(df):
    folder = os.path.join(BASE_DIR, 'THE_BEST_TEAM')
    os.makedirs(folder, exist_ok=True)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    stat_summary, team_scores = [], {}

    for col in numeric_df.columns:
        if col not in df.columns or df[col].dropna().empty:
            continue

        team_means = df.groupby('Team')[col].mean().dropna()
        if team_means.empty:
            continue

        top_team = team_means.idxmax()
        top_score = team_means.loc[top_team]

        stat_summary.append({
            'Statistic': col,
            'Best Team': top_team,
            'Average Score': round(top_score, 4)
        })

        team_scores[top_team] = team_scores.get(top_team, 0) + 1

    best_team = max(team_scores.items(), key=lambda x: x[1])[0]
    top_count = team_scores[best_team]

    stat_summary.append({
        'Statistic': 'The best Team',
        'Best Team': best_team,
        'Average Score': f'Top in {top_count} statistics'
    })

    filepath = os.path.join(folder, 'Team_of_the_season.txt')
    pd.DataFrame(stat_summary).to_csv(filepath, index=False, sep='\t')
    print(f"✅ Best team: {best_team} (Top in {top_count} stats) → saved to {filepath}")

# ===================== Main =====================

# Đọc và xử lý dữ liệu
df = read_and_process_data('Table_EPL.csv')

# Top/Bottom 3
result_df = get_top_and_bottom_3(df)
save_top_bottom_3(result_df)

# Tính thống kê và lưu
calculate_statistics(df)

# Vẽ biểu đồ histogram
attack_stats = ['Goals', 'Assists', 'SoT%']
defense_stats = ['Tkl', 'Blocks', 'Interceptions']
plot_histograms(df, attack_stats, defense_stats)

# Tìm đội bóng tốt nhất
find_best_team(df)