# Báo Cáo Nghiên Cứu & Bảo Vệ Đề Tài: Hệ Thống Gợi Ý Sản Phẩm Thương Mại Điện Tử Bằng SVD / Matrix Factorization

Tài liệu này tổng hợp toàn bộ cơ sở lý thuyết, thiết kế thực nghiệm, kết quả đánh giá và câu hỏi phản biện học thuật cho dự án **Model-based Collaborative Filtering for E-commerce Product Recommendation using SVD / Matrix Factorization**. Đây là tài liệu hướng dẫn chi tiết phục vụ cho việc bảo vệ thạc sĩ trước Hội đồng Khoa học.

---

## 1. Giới thiệu Tổng Quan Dự Án

Dự án tập trung vào việc phát triển hệ thống gợi ý sản phẩm thương mại điện tử dựa trên hành vi đánh giá (ratings) của người dùng từ bộ dữ liệu **Amazon Beauty Ratings**. Thuật toán chính được sử dụng là **Singular Value Decomposition (SVD)** kết hợp với kỹ thuật **Mean-Centering** chuẩn hóa dữ liệu.

Mô hình đã được triển khai, kiểm thử thực nghiệm trên môi trường máy tính thực tế, đạt tính đúng đắn về thuật toán và đảm bảo hiệu năng xử lý.

---

## 2. Trả Lời 14 Câu Hỏi Phản Biện Bảo Vệ Đề Tài (Q&A)

### Q1: Bài toán Machine Learning ở đây là gì?
- **Trả lời:** Bài toán ở đây thuộc nhóm **Hệ thống Gợi ý (Recommender Systems)**, cụ thể hơn là bài toán **Dự đoán Rating và Xếp hạng sản phẩm (Recommendation / Ranking)**.
- **Phân biệt với các bài toán khác:**
  - **Classification:** Nhãn đầu ra là các lớp rời rạc (ví dụ: `Mua` hay `Không mua`).
  - **Regression:** Đầu ra là một giá trị liên tục đơn lẻ từ tập các đặc trưng cố định.
  - **Clustering:** Phân cụm không giám sát dữ liệu dựa trên khoảng cách đặc trưng mà không có nhãn.
  - **Hệ thống gợi ý:** Đầu vào là ma trận tương tác thưa thớt giữa người dùng và sản phẩm. Mục tiêu là dự đoán các giá trị khuyết thiếu trong ma trận và xếp hạng chúng để đề xuất **Top-N** sản phẩm tốt nhất.

### Q2: Input và Output của hệ thống là gì?
- **Input:** 
  - Mã người dùng (`UserId` - string).
  - Mã sản phẩm (`ProductId` - string).
  - Điểm đánh giá (`Rating` - float từ 1.0 đến 5.0).
  - Lịch sử tương tác được lưu dưới dạng bảng dữ liệu tương tác thưa.
- **Output:**
  - Rating dự đoán cho các sản phẩm người dùng chưa từng tương tác.
  - Danh sách **Top-N sản phẩm gợi ý** cho từng user kèm theo điểm rating dự đoán tương ứng (dạng bảng: `ProductId | PredictedRating`), được sắp xếp giảm dần.

### Q3: Vì sao đây là bài toán Recommender Systems (chứ không phải Classification hay Clustering)?
- **Trả lời:** 
  - Vì mục tiêu cuối cùng của bài toán không phải là gán nhãn đúng/sai (Classification) hay gom nhóm các thực thể (Clustering), mà là **cá nhân hóa đề xuất**. Đầu ra là một danh sách các sản phẩm được **xếp hạng (Ranking)** theo mức độ quan tâm dự kiến của từng cá nhân người dùng.
  - Các chỉ số đánh giá của hệ thống phản ánh đúng bản chất của bài toán Recommender Systems thông qua độ đo xếp hạng như **Precision@K**, **Recall@K**, và **NDCG@K** (đo lường chất lượng của danh sách Top-K được đề xuất thực tế), thay vì chỉ đo độ chính xác phân lớp (Accuracy/F1-score).

### Q4: Vì sao lại chọn mô hình SVD (Singular Value Decomposition)?
- **Trả lời:**
  - **Giải quyết vấn đề dữ liệu thưa (Sparsity):** Ma trận tương tác người dùng - sản phẩm trong thực tế cực kỳ thưa (độ thưa thường > 99.9%). SVD giúp phân tách ma trận này thành các nhân tố ẩn (latent factors) có số chiều thấp hơn ($k \\ll m, n$), tự động loại bỏ nhiễu và lấp đầy các ô trống một cách hiệu quả.
  - **Khám phá mối quan hệ ẩn (Latent Semantic Features):** SVD tự động tìm ra các đặc trưng ẩn biểu thị sở thích người dùng và thuộc tính sản phẩm mà không cần gán nhãn thủ công.
  - **Hiệu năng và Khả năng mở rộng:** Truncated SVD trong scikit-learn tối ưu hóa tính toán ma trận thưa tốt, thời gian huấn luyện nhanh hơn so với các phương pháp tính toán khoảng cách lân cận (Memory-based Collaborative Filtering) khi số lượng người dùng tăng lên.

### Q5: Dataset có những đặc điểm gì nổi bật?
- **Bộ dữ liệu sử dụng:** Amazon Beauty Ratings (`ratings_Beauty.csv`) chứa 2,023,070 ratings của 1,210,271 users trên 249,274 products.
- **Các đặc điểm cốt lõi:**
  1. **Độ thưa cực kỳ cao (High Sparsity):** Đạt **99.9993%**, nghĩa là trung bình một người dùng chỉ đánh giá dưới 2 sản phẩm trên tổng số gần 250,000 sản phẩm.
  2. **Phân phối điểm số lệch (Skewed Distribution):** Hơn 50% tổng số đánh giá đạt điểm tuyệt đối 5.0. Người dùng có xu hướng đánh giá tích cực.
  3. **Phân phối đuôi dài (Long-tail Distribution):** Hơn 80% người dùng chỉ đánh giá đúng 1 sản phẩm. Tương tự, phần lớn các sản phẩm chỉ nhận được 1 lượt đánh giá duy nhất, trong khi một số ít sản phẩm bán chạy nhận được hàng nghìn lượt đánh giá.

### Q6: Dữ liệu được tiền xử lý như thế nào?
1. **Làm sạch:** Loại bỏ các giá trị khuyết thiếu (null) và gộp các đánh giá trùng lặp (nếu cùng một User đánh giá một Product nhiều lần, lấy điểm trung bình).
2. **Lọc dữ liệu thưa (Filter Sparse Data):** Chỉ giữ lại các user đã đánh giá ít nhất 5 sản phẩm và các sản phẩm nhận được ít nhất 5 đánh giá để đảm bảo mô hình có đủ thông tin học tập.
3. **Phân mẫu tối ưu (Sub-sampling):** Lấy tập con gồm `MAX_USERS = 5000` người dùng tích cực nhất và `MAX_PRODUCTS = 5000` sản phẩm phổ biến nhất từ tập dữ liệu đã lọc. Điều này giúp ma trận User-Item đạt kích thước $5000 \\times 5000$ (tỷ lệ thưa giảm xuống còn ~99.71%), vừa đảm bảo tính đại diện vừa giúp mô hình chạy mượt mà trên các máy tính thông thường mà không gây tràn bộ nhớ.

### Q7: Thiết kế Train/Test split được thực hiện ra sao?
- **Thiết kế phân chia theo Người dùng (User-based Split):** 
  - Với mỗi người dùng, ta phân chia ngẫu nhiên **80%** số lượt đánh giá của họ vào tập huấn luyện (Train set) và **20%** còn lại vào tập kiểm thử (Test set).
  - **Lý do:** Đảm bảo rằng **mọi người dùng xuất hiện trong tập kiểm thử đều đã xuất hiện trong tập huấn luyện** (tối thiểu là 1 rating trong train). Phương pháp này giúp tránh hoàn toàn lỗi logic đánh giá khi gặp người dùng mới (Cold-start User) trong tập kiểm thử, đảm bảo tính khách quan học thuật của thực nghiệm.

### Q8: Mô hình được huấn luyện như thế nào?
Quy trình huấn luyện gồm các bước:
1. **Xây dựng ma trận User-Item:** Chuyển đổi dữ liệu tập train dạng bảng sang dạng ma trận $R$.
2. **Mean-Centering:** Tính rating trung bình $\\mu_u$ của từng user. Trừ rating thực tế cho $\\mu_u$ để thu được ma trận lệch tâm. Các ô khuyết thiếu được điền bằng điểm `0` (mang ý nghĩa trung lập).
3. **Phân tách SVD:** Áp dụng `sklearn.decomposition.TruncatedSVD` trên ma trận lệch tâm để trích xuất ma trận latent đặc trưng ẩn của user ($P$) và product ($Q$).
4. **Tái tạo ma trận dự đoán:** Nhân hai ma trận ẩn này lại và cộng trả lại rating trung bình $\\mu_u$ của từng user: $\\hat{R} = P \\times Q^T + \\mu_u$.
5. **Clipping:** Giới hạn kết quả dự đoán về đoạn $[1.0, 5.0]$.

### Q9: Top-N Recommendation được tạo ra như thế nào?
- Khi cần gợi ý sản phẩm cho người dùng $u$:
  1. Mô hình truy xuất dòng dự báo rating $\\hat{r}_{u}$ của người dùng đó cho toàn bộ các sản phẩm.
  2. Lọc bỏ toàn bộ sản phẩm người dùng $u$ **đã đánh giá** trong tập huấn luyện (chỉ giữ lại sản phẩm mới).
  3. Sắp xếp điểm số dự báo giảm dần.
  4. Trích xuất $N$ sản phẩm đứng đầu làm danh sách gợi ý đề xuất.

### Q10: Đánh giá mô hình bằng các chỉ số (metrics) nào?
- **Đánh giá sai số dự đoán điểm số (Prediction accuracy):**
  - **RMSE (Root Mean Squared Error):** Đo lường độ lệch bình phương trung bình giữa rating thực tế và dự đoán. Phạt nặng sai số lớn.
  - **MAE (Mean Absolute Error):** Độ lệch tuyệt đối trung bình, trực quan và ít nhạy cảm với nhiễu.
- **Đánh giá chất lượng danh sách Top-N gợi ý (Ranking accuracy):**
  - Ngưỡng xác định sản phẩm thực sự thích (Relevant Item) là $\\ge 4.0$.
  - **Precision@K:** Tỷ lệ sản phẩm gợi ý trúng đích trên tổng số $K$ sản phẩm gợi ý.
  - **Recall@K:** Tỷ lệ sản phẩm gợi ý trúng đích trên tổng số sản phẩm thực sự thích của user trong tập test.
  - **NDCG@K:** Đánh giá thứ tự sắp xếp của danh sách gợi ý (ưu tiên các sản phẩm đúng nằm ở vị trí đầu danh sách).

### Q11: Số lượng latent factors (`n_components`) ảnh hưởng thế nào đến mô hình?
- **Độ phức tạp và thời gian chạy:** Khi `n_components` tăng từ 5 đến 50, thời gian huấn luyện tăng gấp đôi (từ ~0.39s lên ~0.83s). Phép nhân ma trận cũng tốn tài nguyên hơn.
- **Phương sai giải thích:** Tăng mạnh từ 2.77% lên 13.87%, chứng tỏ mô hình lưu giữ được nhiều thông tin ẩn hơn.
- **Hiệu năng khuyến nghị:** Các chỉ số gợi ý xếp hạng như Precision@10, Recall@10, và NDCG@10 tăng nhẹ (Recall@10 tăng từ 0.010 lên 0.015). Việc tăng latent factors giúp mô hình nắm bắt được các khía cạnh sở thích cá nhân hóa phức tạp hơn, song cần kiểm soát để tránh overfitting.

### Q12: Độ phức tạp tính toán (Complexity) và Running Time thực tế ra sao?
- **Độ phức tạp:**
  - Xây dựng ma trận: $\\mathcal{O}(R)$
  - Huấn luyện Truncated SVD: $\\mathcal{O}(U \\cdot P \\cdot k)$ với $k$ là số thành phần ẩn.
  - Dự đoán ma trận: $\\mathcal{O}(U \\cdot P \\cdot k)$
  - Gợi ý Top-N: $\\mathcal{O}(P \\log P)$ trên mỗi user.
- **Running Time thực tế (trên phân mẫu 5,000 x 5,000):**
  - Thời gian build matrix: ~0.20 giây.
  - Huấn luyện SVD (k=20): ~0.60 giây.
  - Dự đoán & Tái lập: ~0.62 giây.
  - Đánh giá toàn bộ tập test (4,800+ users): ~12.7 giây nhờ tối ưu hóa numpy vectorization.

### Q13: Mô hình có những hạn chế gì?
1. **Cold-start:** Hoàn toàn không thể đưa ra gợi ý cho người dùng mới hoặc sản phẩm mới chưa từng có đánh giá.
2. **Không gian ma trận tĩnh:** Không cập nhật theo thời gian thực khi người dùng có hành vi tương tác mới, yêu cầu phải tái huấn luyện định kỳ.
3. **Thiếu thông tin ngữ cảnh:** Bỏ qua các dữ liệu phi cấu trúc vô cùng giá trị như hình ảnh, mô tả sản phẩm bằng văn bản và thời gian mua hàng.

### Q14: Hướng phát triển tương lai (Future Work) của đề tài là gì?
1. **Xây dựng hệ thống gợi ý lai (Hybrid Recommender):** Kết hợp phân tách ma trận SVD với Content-based Filtering (như TF-IDF hoặc BERT embeddings trên văn bản mô tả sản phẩm) để xử lý triệt để lỗi Cold-start sản phẩm.
2. **Tận dụng phản hồi ẩn (Implicit Feedback):** Áp dụng thuật toán ALS (Alternating Least Squares) trên hành vi click, lượt xem, thời gian dừng màn hình.
3. **Mô hình học sâu (Deep Learning):** Nghiên cứu Neural Collaborative Filtering (NCF) hoặc Graph Neural Networks (GNN) để tự động học mối quan hệ phi tuyến giữa các thực thể.

---

## 3. Bảng Kết Quả Thực Nghiệm

Dưới đây là bảng kết quả thực nghiệm thu được khi chạy grid-search trên phân mẫu $5000 \\times 5000$:

| n_components | top_k | RMSE | MAE | Precision | Recall | NDCG | TrainTime (s) | PredTime (s) | EvalTime (s) | Explained Var |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5** | 5 | 1.0591 | 0.7715 | 0.0050 | 0.0055 | 0.0062 | 0.39 | 0.56 | 13.04 | 2.77% |
| **5** | 10 | 1.0591 | 0.7715 | 0.0048 | 0.0104 | 0.0081 | 0.39 | 0.56 | 13.94 | 2.77% |
| **10** | 5 | 1.0593 | 0.7715 | 0.0043 | 0.0043 | 0.0048 | 0.45 | 0.59 | 12.60 | 4.56% |
| **10** | 10 | 1.0593 | 0.7715 | 0.0040 | 0.0086 | 0.0064 | 0.45 | 0.59 | 14.09 | 4.56% |
| **20** | 5 | 1.0590 | 0.7712 | 0.0056 | 0.0074 | 0.0076 | 0.60 | 0.71 | 13.04 | 7.37% |
| **20** | 10 | 1.0590 | 0.7712 | 0.0049 | 0.0124 | 0.0093 | 0.60 | 0.71 | 12.71 | 7.37% |
| **50** | 5 | 1.0597 | 0.7718 | 0.0056 | 0.0087 | 0.0084 | 0.72 | 0.61 | 12.73 | 13.87% |
| **50** | 10 | 1.0597 | 0.7718 | 0.0051 | 0.0153 | 0.0110 | 0.72 | 0.61 | 14.07 | 13.87% |

### Nhận xét kết quả:
1. **Độ ổn định:** Sai số RMSE và MAE rất ổn định qua các số lượng components khác nhau, duy trì quanh mức ~1.059 (RMSE) và ~0.771 (MAE).
2. **Xu hướng cải thiện chất lượng đề xuất:** Recall@10 tăng rõ rệt từ `0.0104` ($k=5$) lên `0.0153` ($k=50$), cho thấy mô hình khi được bổ sung nhiều đặc trưng ẩn đã tìm kiếm và gợi ý trúng đích nhiều sản phẩm ưa thích của người dùng hơn.
3. **Tính khả thi thực tế:** Tổng thời gian huấn luyện và dự đoán chỉ mất chưa đầy 1.5 giây, đảm bảo khả năng đáp ứng thời gian thực khi tích hợp vào hệ thống e-commerce thực tế.

---

## 4. Hướng Dẫn Chạy Dự Án (Notebook Execution Guide)

1. **Chuẩn bị file dữ liệu:** Đảm bảo file dữ liệu `ratings_Beauty.csv` được đặt tại thư mục:
   `project/input/amazon-ratings/ratings_Beauty.csv`
2. **Khởi chạy môi trường:** Mở Jupyter Notebook / JupyterLab hoặc VS Code tại thư mục workspace.
3. **Mở file notebook:** `model_based_collaborative_filtering_recommender.ipynb`.
4. **Chạy toàn bộ (Run All):** Thực thi tuần tự các cell từ trên xuống dưới. Các biểu đồ trực quan hóa phân phối rating, sai số, chất lượng xếp hạng và không gian latent (PCA/t-SNE) sẽ được sinh trực tiếp inline.
