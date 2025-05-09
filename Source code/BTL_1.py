from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time
import os


# ========== Thiết lập thư mục lưu kết quả ========== 
BASE_DIR = os.path.join('REPORT_BTL', 'BTL1_File')  # Thay đổi đường dẫn đến thư mục mong muốn
os.makedirs(BASE_DIR, exist_ok=True)

# ==== Cấu hình trình duyệt ====
def init_driver():
    return webdriver.Chrome()

# ==== Lấy BeautifulSoup từ URL ====
def get_soup(driver, url):
    driver.get(url)
    time.sleep(0.5)
    return BeautifulSoup(driver.page_source, 'html.parser')

# ==== Trích xuất dữ liệu theo fields từ một <tr> ====
def extract_data(tr, fields):
    row = []
    for field in fields:
        td = tr.find("td", {"data-stat": field})
        value = td.text.strip() if td else 'N/a'
        row.append(value if value != '' else 'N/a')
    return row

# ==== Lấy dữ liệu chính của cầu thủ (tối thiểu 90 phút) ====
def get_main_player_data(driver, url_base, info_fields, columns):
    soup = get_soup(driver, url_base)
    table = soup.find('table', {'class': 'min_width sortable stats_table shade_zero now_sortable sticky_table eq2 re2 le2'})
    body = table.find('tbody')

    players_dict = {}

    for tr in body.find_all('tr'):
        if 'thead' in tr.get('class', []):
            continue

        minutes = tr.find('td', {'data-stat': 'minutes'}).get_text(strip=True).replace(',', '')
        if not minutes or int(minutes) < 90:
            continue

        player_th = tr.find('td', {'data-append-csv': True})
        if not player_th:
            continue

        data = extract_data(tr, info_fields)
        nation = tr.find('td', {'data-stat': 'nationality'}).find('a').find('span').contents[-1].strip()
        player_id = player_th['data-append-csv'].strip()

        players_dict[player_id] = [data[0] , nation] + data[2:] + ['N/a'] * (len(columns) - len(data))

    print(f"Done: {url_base}")
    return players_dict

# ==== Hàm cập nhật dữ liệu từ các bảng phụ ====
def update_sections(driver, players_dict, sections, start_index):
    x = start_index

    for url, fields in sections:
        soup = get_soup(driver, url)
        table = soup.find('table', {'class': 'min_width sortable stats_table shade_zero now_sortable sticky_table eq2 re2 le2'})
        body = table.find('tbody')

        for tr in body.find_all('tr'):
            player_th = tr.find('td', {'data-append-csv': True})
            if not player_th:
                continue

            player_id = player_th['data-append-csv'].strip()
            if player_id not in players_dict:
                continue

            stats = extract_data(tr, fields)
            players_dict[player_id][x:x + len(fields)] = stats

        x += len(fields)
        print(f"Done: {url}")

    return players_dict

# ==== Hàm xuất DataFrame ra file CSV ==== 
def export_to_csv(players_dict, columns, filename):
    players_sorted = sorted(players_dict.values(), key=lambda x: x[0])
    multi_cols = pd.MultiIndex.from_tuples(columns)
    df = pd.DataFrame(players_sorted, columns=multi_cols)
    df.replace(['', None], 'N/a', inplace=True)
    df.fillna('N/a', inplace=True)
    file_path = os.path.join(BASE_DIR, filename)  # Lưu vào thư mục REPORT_BTL/BTL1_File
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Successfully exported to {file_path} file")


# ==== Main chương trình ====
def main():
    driver = init_driver()

    # ==== URL và cấu hình ==== 
    url_base = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'
    info_fields = [
        'player', 'nationality', 'team', 'position', 'age',
        'games', 'games_starts', 'minutes', 'goals', 'assists',
        'cards_yellow', 'cards_red', 'xg', 'xg_assist',
        'progressive_carries', 'progressive_passes', 'progressive_passes_received',
        'goals_per90', 'assists_per90', 'xg_per90', 'xg_assist_per90'
    ]

    sections = [
        ('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', 
        ['gk_goals_against_per90', 'gk_save_pct', 'gk_clean_sheets_pct', 'gk_pens_save_pct']),
        ('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', 
        ['shots_on_target_pct', 'shots_on_target_per90', 'goals_per_shot', 'average_shot_distance']),
        ('https://fbref.com/en/comps/9/passing/Premier-League-Stats', 
        ['passes_completed', 'passes_pct', 'passes_total_distance',
        'passes_pct_short', 'passes_pct_medium', 'passes_pct_long',
        'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area',
        'crosses_into_penalty_area', 'progressive_passes']),
        ('https://fbref.com/en/comps/9/gca/Premier-League-Stats', 
        ['sca', 'sca_per90', 'gca', 'gca_per90']),
        ('https://fbref.com/en/comps/9/defense/Premier-League-Stats', 
        ['tackles', 'tackles_won', 'challenges', 'challenges_lost',
         'blocks', 'blocked_shots', 'blocked_passes', 'interceptions']),
        ('https://fbref.com/en/comps/9/possession/Premier-League-Stats', 
        ['touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd',
        'touches_att_3rd', 'touches_att_pen_area', 'take_ons', 'take_ons_won',
        'take_ons_tackled', 'carries', 'carries_progressive_distance',
        'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area',
        'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received']),
        ('https://fbref.com/en/comps/9/misc/Premier-League-Stats', 
        ['fouls', 'fouled', 'offsides', 'crosses',
         'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']),
    ]

    # ==== Cột MultiIndex ====
    columns = [
    ('','' ,'Name'),
    ('', '','Nation' ),
    ('', '', 'Team'),
    ('', '', 'Position' ),
    ('', '', 'Age'),
    ('', 'Playing Time', 'Matches played'),
    ('', '', 'Starts'),
    ('', '', 'Minutes'),
    ('', 'Performance', 'Goals'),
    ('', '', 'Assists'),
    ('', '', 'Yellow Cards'),
    ('', '', 'Red Cards'),
    ('', 'Expected', 'xG'),
    ('', '', 'xAG'),
    ('', 'Progression', 'Prg Carries'),
    ('', '', 'Prg Passes'),
    ('', '', 'Prg Passes Rec'),
    ('', 'Per 90 Minutes', 'Goals/90'),
    ('', '', 'Assists/90'),
    ('', '', 'xG/90'),
    ('', '', 'xGA/90'),
    ('Goalkeeping', 'Performance', 'GA90'),
    ('', '', 'Save%'),
    ('', '', 'CS%'),
    ('', 'Penalty Kicks', 'PK_Save%'),
    ('Shooting','Standard', 'SoT%'),
    ('','','Sot/90'),
    ('','', 'G/sh'),
    ('','', 'Shoot_Dist'),
    ('Passing','Total', 'Passes Completed (Cmp)'),
    ('','', 'Cmp%'),
    ('','', 'Total Dist'),
    ('','Short', 'Short_Cmp%'),
    ('','Medium', 'Medium_Cmp%'),
    ('','Long', 'Long_Cmp%'),
    ('','Expected', 'Key Passes'),
    ('','', 'Pass 1/3'),
    ('','', 'Pass PA'),
    ('','', 'Cross PA'),
    ('','', 'Prg Passes (Pass)'),
    ('Goal and Shot Creation','SCA', 'Shot-Creating Actions (SCA)'),
    ('','', 'SCA90'),
    ('','GCA', 'Goal-Creating Actions (GCA)'),
    ('','', 'GCA90'),
    ('Defensive','Tackles', 'Tkl'),
    ('','', 'Tkl Won'),
    ('','Challenges', 'Dribbles Challenged (Att)'),
    ('','', 'Challenges Lost'),
    ('','Blocks', 'Blocks'),
    ('','', 'Shots Blocked'),
    ('','', 'Pass Blocked'),
    ('','', 'Interceptions'),
    ('Possession','Touches', 'Touches'),
    ('','', 'Def_Pen'),
    ('','', 'Def_3rd'),
    ('','', 'Mid_3rd'),
    ('','', 'Att_3rd'),
    ('','', 'Att_Pen'),
    ('','Take-Ons', 'Attempted'),
    ('','', 'Successful %'),
    ('','', 'Tkld%'),
    ('','Carries', 'Carries'),
    ('','', 'PrgDist'),
    ('','', 'PrgC (Poss)'),
    ('','', 'Carries_1/3'),
    ('','', 'Carries_CPA'),
    ('','', 'Miscontrols'),
    ('','', 'Dispossessed'),
    ('','Receiving', 'Passes Rec'),
    ('','', 'Prg Rec (Poss)'),
    (' Miscellaneous','Performance', 'Fouls Committed'),
    ('','', 'Fouls Drawn'),
    ('','', 'Offsides'),
    ('','', 'Crosses'),
    ('','', 'Recov'),
    ('','Aerial Duels', 'Aerials Won'),
    ('','', 'Aerials Lost'),
    ('','', 'Aerials Won%')
    ]

    # ==== Lấy dữ liệu ====
    players_dict = get_main_player_data(driver, url_base, info_fields, columns)
    players_dict = update_sections(driver, players_dict, sections, len(info_fields))
    export_to_csv(players_dict, columns, 'Table_EPL.csv')

    driver.quit()

if __name__ == "__main__":
    main()