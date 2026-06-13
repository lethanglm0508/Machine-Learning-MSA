# Machine Learning: E-commerce Product Recommendation System (Transition to Linear/Ridge Regression)

Dự án này là một hệ thống gợi ý sản phẩm thương mại điện tử phục vụ cho môn học **Machine Learning**. Dự án thực hiện chuyển đổi mô hình cốt lõi từ phương pháp lọc cộng tác **SVD (Matrix Factorization)** sang phương pháp **Hồi quy Tuyến tính (Linear Regression)** và **Hồi quy Ridge (Ridge Regression)** có giám sát trên bộ dữ liệu **Magazine Subscriptions** của Amazon.

---

## 🌟 Tính Năng Chính
- **Hồi quy Giám sát (Supervised Regression):** Dự đoán mức độ yêu thích (Rating) từ $1.0$ đến $5.0$ cho mỗi người dùng đối với các tạp chí.
- **Tích hợp Đặc trưng Đa nguồn (Multi-source Feature Engineering):** Kết hợp các chỉ số tương tác lịch sử (User Average, Item Average),Platform metadata (Amazon average rating, rating count), transactional indicators (`verified_purchase`, `helpful_vote`), đặc trưng nội dung (TF-IDF PCA/SVD) và độ tương đồng giữa sở thích người dùng và sản phẩm.
- **Trực quan hóa Dữ liệu Không gian Thấp (Data Visualization):** Sử dụng PCA và t-SNE để giảm chiều dữ liệu tiêu đề tạp chí TF-IDF và trực quan hóa trong không gian 2D.
- **Đánh giá Đa chiều:** Đánh giá độ chính xác điểm số (RMSE, MAE) và khả năng xếp hạng gợi ý Top-10 thực tế (Precision@10, Recall@10, NDCG@10) so với các mô hình baseline.

---

## 📂 Cấu Trúc Thư Mục
```text
Machine-Learning-MSA/
├── input/
│   ├── Magazine_Subscriptions.jsonl        # Dữ liệu đánh giá (User Ratings)
│   └── meta_Magazine_Subscriptions.jsonl   # Dữ liệu Metadata sản phẩm
├── plots/
│   ├── rating_distribution.png              # Biểu đồ phân phối rating
│   ├── pca_visualization.png                # Trực quan hóa sản phẩm bằng PCA (2D)
│   └── tsne_visualization.png               # Trực quan hóa sản phẩm bằng t-SNE (2D)
├── model_based_collaborative_filtering_recommender.ipynb  # Notebook chính của dự án (Linear Regression)
├── report_recommendation_system.md         # Báo cáo học thuật chi tiết (Tiếng Việt)
└── README.md                               # Hướng dẫn dự án
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Cài đặt các thư viện cần thiết
Dự án yêu cầu cài đặt Python 3.8+ và các thư viện học máy cơ bản sau:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Chuẩn bị dữ liệu
Đặt hai tệp dữ liệu vào thư mục `input/`:
- `input/Magazine_Subscriptions.jsonl`
- `input/meta_Magazine_Subscriptions.jsonl`

### 3. Chạy Notebook
Bạn có thể mở tệp `model_based_collaborative_filtering_recommender.ipynb` bằng VS Code hoặc Jupyter Lab và thực thi tất cả các cell để tái lập kết quả thực nghiệm.

---

## 📊 Kết Quả Thực Nghiệm

### 1. Độ chính xác dự đoán điểm (Prediction Metrics)
Đánh giá trên tập kiểm thử (Test Set) sau khi phân chia User-based Split:

| Mô hình | RMSE | MAE |
| :--- | :---: | :---: |
| **Global Mean Baseline** | 1.0896 | 0.8391 |
| **Linear Regression** | 0.9756 | 0.6476 |
| **Ridge Regression (alpha=10)** | **0.9724** | **0.6434** |
| *User Mean Baseline* | *0.9678* | *0.5506* |

### 2. Chất lượng gợi ý xếp hạng Top-10 (Ranking Metrics)
Đánh giá khả năng đề xuất danh sách Top-10 cho các người dùng:

| Mô hình gợi ý | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (alpha=10)** | 0.0015 | 0.0142 | 0.0068 |
| **Popularity Baseline** | **0.0205** | **0.1846** | **0.0910** |

*Lưu ý và thảo luận chi tiết về sự phân bố đuôi dài (long-tail) của dữ liệu thương mại điện tử và lý do Popularity Baseline đạt kết quả tốt hơn được trình bày rõ trong [Báo Cáo Dự Án](report_recommendation_system.md).*

---

## 📉 Trực Quan Hóa Không Gian Sản Phẩm (t-SNE)
Dưới đây là không gian biểu diễn văn bản của các sản phẩm tạp chí được giảm chiều bằng t-SNE từ TF-IDF của tiêu đề:

![t-SNE Visualization](plots/tsne_visualization.png)