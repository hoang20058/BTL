import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Tạo thư mục lưu kết quả
BASE_DIR = os.path.join('REPORT_BTL', 'BTL3_File')  # Thay đổi đường dẫn đến thư mục mong muốn
os.makedirs(BASE_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Đọc và chuẩn hóa dữ liệu số từ file CSV.
    """
    df = pd.read_csv(file_path, header=2)

    # Chỉ giữ lại các cột số, bỏ cột 'Age' nếu có
    numeric_df = df.select_dtypes(include='number').drop(columns=['Age'], errors='ignore')

    # Thay thế giá trị thiếu bằng 0
    numeric_df = numeric_df.fillna(0)

    # Chuẩn hóa
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    return df, numeric_df, scaled_data


def find_optimal_k(data):
    """
    Tìm số cụm tối ưu bằng Elbow Method và lưu biểu đồ.
    """
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data) 
        wcss.append(kmeans.inertia_)


    # Vẽ biểu đồ Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title(f"Elbow Method - Chọn số cụm tối ưu k ")
    plt.xlabel("Số cụm (k)")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "Elbow_Method.png"), dpi=300)
    plt.close()
    optimal_k = 4
    return optimal_k

def perform_kmeans(data, k):
    """
    Thực hiện phân cụm KMeans.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    # Chỉnh lại nhãn để bắt đầu từ 1 thay vì 0
    cluster_labels += 1
    
    return cluster_labels


def reduce_with_pca(data):
    """
    Giảm chiều dữ liệu bằng PCA (2 thành phần chính).
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

def plot_clusters(pca_data, cluster_labels):
    """
    Vẽ biểu đồ phân cụm sau PCA.
    """
    df_plot = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set2', data=df_plot, s=100)
    plt.title("Biểu đồ phân cụm bằng PCA (2D)")
    plt.xlabel("Thành phần 1")
    plt.ylabel("Thành phần 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "KMeans_PCA_Clusters.png"), dpi=300)
    plt.close()
    print(f"Successful plot Kmean PCA")

def save_cluster_analysis(result_df, output_file):
    """
    Lưu thông tin phân tích các cụm vào file TXT.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("--- PHÂN TÍCH CỤM CẦU THỦ ---\n")
        for c in sorted(result_df['Cluster'].unique()):
            group = result_df[result_df['Cluster'] == c]
            f.write(f"\n=============================\n")
            f.write(f"✅ Cụm {c} - Số cầu thủ: {len(group)}\n")

            if 'Name' in group.columns:
                f.write("🔹 Tên cầu thủ: " + ", ".join(group['Name'].astype(str).tolist()) + "\n")
            else:
                f.write("Không có cột 'Name' để hiển thị tên.\n")

            # Tính trung bình cho các chỉ số số học
            stats = group.select_dtypes(include='number').mean().round(2)
            f.write("🔍 Trung bình chỉ số cụm:\n")
            f.write(stats.to_string())
            f.write("\n")

        print(f"Successful Saved to {output_file}")

def main():
    file_path = "Table_EPL.csv"

    # Bước 1: Load và chuẩn hóa dữ liệu
    df_raw, df_numeric, scaled_data = load_and_preprocess_data(file_path)

    # Bước 2: Tìm số cụm tối ưu
    optimal_k = find_optimal_k(scaled_data)

    # Bước 3: Phân cụm KMeans
    clusters = perform_kmeans(scaled_data, optimal_k)

    # Bước 4: Giảm chiều bằng PCA
    pca_data = reduce_with_pca(scaled_data)

    # Bước 5: Vẽ biểu đồ phân cụm
    plot_clusters(pca_data, clusters)

    # Bước 6: Gắn nhãn cụm vào DataFrame gốc và lưu
    result_df = df_raw.loc[df_numeric.index].copy()
    result_df['Cluster'] = clusters
    result_df.to_csv(os.path.join(BASE_DIR, "Table_EPL_Clustered.csv"), index=False, encoding='utf-8-sig')
    print(f"Successful save to csv file")

    # Bước 7: Ghi phân tích cụm vào file
    analysis_file = os.path.join(BASE_DIR, "Cluster_Analysis.txt")
    save_cluster_analysis(result_df, analysis_file)

if __name__ == "__main__":
    main()

