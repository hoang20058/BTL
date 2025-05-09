import time
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from bs4 import BeautifulSoup

# Hàm chuẩn hóa tuổi từ chuỗi 'tuổi-ngày' thành dạng số thực
def standardize_age(age_string):
    """
    Chuẩn hóa tuổi từ định dạng 'tuổi-ngày' thành số thực.
    """
    try:
        age, days = map(int, age_string.split('-'))
        age += days / 365  # Chuyển số ngày thành phần của tuổi
        return round(age, 2)  # Làm tròn đến 2 chữ số thập phân
    except:
        return None  # Trả về None nếu không thể chuẩn hóa

# Hàm đọc và lọc dữ liệu từ CSV, chuẩn hóa tuổi và loại bỏ dữ liệu không hợp lệ
def load_and_filter_data(input_path):
    """
    Đọc dữ liệu từ CSV, chuẩn hóa tuổi và lọc các cầu thủ có hơn 900 phút thi đấu.
    """
    df = pd.read_csv(input_path, header=[2])

    # Kiểm tra và chuyển dữ liệu không hợp lệ thành NaN cho cột 'Minutes'
    df['Minutes'] = df['Minutes'].str.replace(',', '').apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['Minutes'])  # Loại bỏ các hàng có giá trị Minutes không hợp lệ
    df['Minutes'] = df['Minutes'].astype(int)  # Chuyển cột 'Minutes' thành kiểu int

    # Chuẩn hóa cột 'Age' và loại bỏ các hàng không hợp lệ
    df['Age'] = df['Age'].apply(standardize_age)
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)

    # Chỉ giữ lại các hàng có 'Minutes' > 900
    return df[df['Minutes'] > 900]

# Hàm crawl dữ liệu giá trị chuyển nhượng từ trang footballtransfers.com
def crawl_transfer_values(player_names, max_pages=22):
    """
    Crawl dữ liệu giá trị chuyển nhượng của các cầu thủ từ trang web footballtransfers.com.
    """
    values = {name: None for name in player_names}
    collected_data = {}

    driver = webdriver.Chrome()
    driver.get("https://www.footballtransfers.com/us/players/uk-premier-league/")

    for _ in range(max_pages):
        time.sleep(2)  # Chờ để tải dữ liệu
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='table-hover')
        if not table:
            break

        name_tags = table.find_all('div', class_='text')
        price_tags = table.find_all('span', class_='player-tag')

        # Lấy tên và giá trị chuyển nhượng cho các cầu thủ cần tìm
        for name_tag, price_tag in zip(name_tags, price_tags):
            a_tag = name_tag.find('a')
            if not a_tag:
                continue
            player_name = a_tag.get('title')
            if player_name in player_names:
                collected_data[player_name] = price_tag.text.strip()

        try:
            driver.find_element(By.CLASS_NAME, 'pagination_next_button').click()  # Chuyển trang
        except:
            break

    driver.quit()
    values.update(collected_data)  # Cập nhật giá trị chuyển nhượng vào từ điển
    return values

# Hàm chuyển đổi giá trị tiền tệ từ chuỗi sang số thực
def convert_price(price_str):
    """
    Chuyển giá trị tiền tệ từ chuỗi dạng '€1M', '€1K',... thành số thực.
    """
    if not isinstance(price_str, str):
        return None
    try:
        if 'M' in price_str:
            return int(float(price_str.replace('€', '').replace('M', '').strip()) * 1_000_000)
        elif 'K' in price_str:
            return int(float(price_str.replace('€', '').replace('K', '').strip()) * 1_000)
        else:
            return int(float(price_str.replace('€', '').strip()))
    except:
        return None

# Hàm chuyển giá trị tiền tệ từ số thực thành chuỗi có định dạng đẹp
def convert_price_to_string(price):
    """
    Chuyển giá trị chuyển nhượng từ số thành chuỗi có định dạng '€1M' hoặc '€1K'.
    """
    if pd.isna(price):
        return None
    if price >= 1_000_000:
        return f"€{price / 1_000_000:.1f}M"
    elif price >= 1_000:
        return f"€{price / 1_000:.1f}K"
    else:
        return f"€{price}"

# Hàm lưu dữ liệu đã lọc và cập nhật giá trị chuyển nhượng vào file CSV
def save_filtered_data(df, transfer_values_dict, output_path):
    """
    Lưu dữ liệu đã lọc và thêm giá trị chuyển nhượng vào file CSV.
    """
    # Thêm cột 'Transfer values (numeric)' từ từ điển giá trị chuyển nhượng
    df['Transfer values (numeric)'] = df['Name'].map(transfer_values_dict).map(convert_price)

    # Xoá các hàng không có giá trị hợp lệ
    df = df.dropna(subset=['Transfer values (numeric)']).copy()

    # Chuyển sang định dạng đẹp cho cột 'Transfer values'
    df['Transfer values'] = df['Transfer values (numeric)'].apply(convert_price_to_string)

    # Lưu dữ liệu vào CSV
    df_to_save = df.drop(columns=['Transfer values (numeric)'])
    df_to_save.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved updated data to {output_path}")

    return df  # Trả về dataframe đầy đủ để tiếp tục sử dụng

# Hàm huấn luyện mô hình Random Forest và dự đoán giá trị chuyển nhượng
def train_and_predict_with_pipeline(df):
    """
    Huấn luyện mô hình Random Forest và dự đoán giá trị chuyển nhượng.
    """
    features = ['Age', 'Position', 'Minutes', 'Goals', 'Assists', 'Goal-Creating Actions (GCA)', 'Prg Passes Rec', 'Tkl']
    target = 'Transfer values (numeric)'

    X = df[features]
    y = df[target]

    # Tiền xử lý cho dữ liệu số và phân loại
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Age', 'Minutes', 'Goals', 'Assists', 'Goal-Creating Actions (GCA)', 'Prg Passes Rec', 'Tkl']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Position'])  # Chỉ một cột phân loại 'Position'
    ])

    # Xây dựng pipeline với mô hình Random Forest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Dự đoán và tính toán lỗi tuyệt đối trung bình
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {convert_price_to_string(mae)} €")

# Hàm chính thực hiện toàn bộ quy trình
def main():
    """
    Thực hiện toàn bộ quy trình từ việc tải, lọc dữ liệu, crawl giá trị chuyển nhượng, huấn luyện mô hình đến dự đoán giá trị cầu thủ.
    """
    # Đặt đường dẫn lưu vào thư mục REPORT_BTL/BTL4_File
    report_btl_path = "REPORT_BTL"
    btl4_file_path = os.path.join(report_btl_path, "BTL4_File")
    
    # Tạo thư mục REPORT_BTL và BTL4_File nếu chưa tồn tại
    os.makedirs(btl4_file_path, exist_ok=True)

    input_path = "Table_EPL.csv"
    output_path = os.path.join(btl4_file_path, "Players_900mins.csv")

    # Bước 1: Đọc và lọc dữ liệu
    filtered_df = load_and_filter_data(input_path)

    # Bước 2: Crawl dữ liệu giá trị chuyển nhượng
    transfer_values = crawl_transfer_values(set(filtered_df['Name']))

    # Bước 3: Lưu dữ liệu đã cập nhật giá trị chuyển nhượng
    df_updated = save_filtered_data(filtered_df, transfer_values, output_path)

    # Bước 4: Huấn luyện mô hình và dự đoán giá trị chuyển nhượng
    train_and_predict_with_pipeline(df_updated)

    print("✅ Successful")


if __name__ == '__main__':
    main()  # Chạy hàm chính khi file này được thực thi
