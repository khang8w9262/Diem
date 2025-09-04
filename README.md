# 🎓 Hệ thống Dự báo Ngành học

## 📊 Tổng quan
Hệ thống so sánh hiệu suất dự báo ngành học giữa **Major Prediction** (phương pháp truyền thống) và **LightGBM** (Machine Learning).

## 🎯 Kết quả chính
- **5 ngành học:** CNPM, Mạng, An toàn, Hệ thống, Máy học
- **LightGBM accuracy:** 86.4% (vượt trội +21.4% so với phương pháp cơ bản)
- **Tỷ lệ trùng khớp:** 93.1%
- **Dữ liệu:** 5,000 sinh viên, 42 môn học

## 🚀 Cài đặt và Chạy

### 1. Clone repository
```bash
git clone <repository-url>
cd Diem
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy Streamlit Dashboard
```bash
streamlit run streamlit_dashboard_updated.py
```

### 4. Truy cập Dashboard
Mở browser tại: `http://localhost:8501`

## 📁 Cấu trúc Project

```
Diem/
├── Data/                                    # Dữ liệu
│   ├── 1.xlsx                              # Ma trận hệ số
│   ├── DiemDaw.csv                         # Dữ liệu gốc
│   ├── student_grades_5000.csv             # Điểm 5000 sinh viên
│   ├── student_major_predictions_5000.csv  # Dự báo ngành
│   └── comparison_lgbm_vs_major_with_analysis.csv  # So sánh
├── streamlit_dashboard_updated.py          # Dashboard chính
├── enhanced_major_prediction.py            # Script nâng cấp hệ thống
├── lgbm_major_prediction.ipynb            # Notebook huấn luyện model
├── duabao.py                              # Utilities
├── requirements.txt                        # Dependencies
└── README.md                              # Documentation
```

## 🔧 Các Chức năng

### 1. **Streamlit Dashboard** (`streamlit_dashboard_updated.py`)
- **Tổng quan:** So sánh tổng thể 2 phương pháp
- **Phân bố ngành:** Biểu đồ pie charts
- **Độ chính xác:** So sánh chi tiết với biểu đồ
- **Phân tích GPA:** Correlation analysis
- **Ma trận nhầm lẫn:** Heatmap comparison
- **Phân bố điểm:** Multi-dimensional analysis

### 2. **Enhanced Major Prediction** (`enhanced_major_prediction.py`)
- Thuật toán Major Prediction cải tiến
- Huấn luyện LightGBM với hyperparameters tối ưu
- Tạo file so sánh và analysis

### 3. **LightGBM Training** (`lgbm_major_prediction.ipynb`)
- Sinh dữ liệu 5000 sinh viên
- Feature engineering
- Model training và evaluation
- Export model và encoders

## 📈 Kết quả So sánh

| Phương pháp | Độ chính xác | Ưu điểm | Nhược điểm |
|-------------|--------------|---------|------------|
| **Major Prediction** | 65.0% | Đơn giản, nhanh | Không học từ dữ liệu |
| **LightGBM** | **86.4%** | **Học từ dữ liệu, accuracy cao** | Phức tạp hơn |

## 🎯 Highlights

- ✅ **Cải thiện +21.4%** accuracy với LightGBM
- ✅ **Đầy đủ 5 ngành** thay vì 4 ngành trước đó
- ✅ **Tỷ lệ trùng khớp 93.1%** giữa 2 phương pháp
- ✅ **Interactive dashboard** với Streamlit
- ✅ **Visualizations** chuyên nghiệp với Plotly

## 🛠 Technology Stack

- **Python 3.13**
- **Streamlit:** Web dashboard
- **LightGBM:** Machine learning
- **Pandas/NumPy:** Data processing
- **Plotly:** Interactive visualizations
- **Scikit-learn:** ML utilities

## 📝 Usage

1. **Khởi động dashboard:**
   ```bash
   streamlit run streamlit_dashboard_updated.py
   ```

2. **Train model mới:**
   ```bash
   python enhanced_major_prediction.py
   ```

3. **Xem notebook training:**
   ```bash
   jupyter notebook lgbm_major_prediction.ipynb
   ```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License.

## 👥 Authors

- **Research Team** - *Initial work*

## 🙏 Acknowledgments

- LightGBM team for the excellent ML framework
- Streamlit for the amazing dashboard framework
- Contributors and testers
# 🎓 Hệ thống Dự báo Ngành học

## 📊 Tổng quan
Hệ thống so sánh hiệu suất dự báo ngành học giữa **Major Prediction** (phương pháp truyền thống) và **LightGBM** (Machine Learning).

## 🎯 Kết quả chính
- **5 ngành học:** CNPM, Mạng, An toàn, Hệ thống, Máy học
- **LightGBM accuracy:** 86.4% (vượt trội +21.4% so với phương pháp cơ bản)
- **Tỷ lệ trùng khớp:** 93.1%
- **Dữ liệu:** 5,000 sinh viên, 42 môn học

## 🚀 Cài đặt và Chạy

### 1. Clone repository
```bash
git clone <repository-url>
cd Diem
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Chạy Streamlit Dashboard
```bash
streamlit run streamlit_dashboard_updated.py
```

### 4. Truy cập Dashboard
Mở browser tại: `http://localhost:8501`

## 📁 Cấu trúc Project

```
Diem/
├── Data/                                    # Dữ liệu
│   ├── 1.xlsx                              # Ma trận hệ số
│   ├── DiemDaw.csv                         # Dữ liệu gốc
│   ├── student_grades_5000.csv             # Điểm 5000 sinh viên
│   ├── student_major_predictions_5000.csv  # Dự báo ngành
│   └── comparison_lgbm_vs_major_with_analysis.csv  # So sánh
├── streamlit_dashboard_updated.py          # Dashboard chính
├── enhanced_major_prediction.py            # Script nâng cấp hệ thống
├── lgbm_major_prediction.ipynb            # Notebook huấn luyện model
├── duabao.py                              # Utilities
├── requirements.txt                        # Dependencies
└── README.md                              # Documentation
```

## 🔧 Các Chức năng

### 1. **Streamlit Dashboard** (`streamlit_dashboard_updated.py`)
- **Tổng quan:** So sánh tổng thể 2 phương pháp
- **Phân bố ngành:** Biểu đồ pie charts
- **Độ chính xác:** So sánh chi tiết với biểu đồ
- **Phân tích GPA:** Correlation analysis
- **Ma trận nhầm lẫn:** Heatmap comparison
- **Phân bố điểm:** Multi-dimensional analysis

### 2. **Enhanced Major Prediction** (`enhanced_major_prediction.py`)
- Thuật toán Major Prediction cải tiến
- Huấn luyện LightGBM với hyperparameters tối ưu
- Tạo file so sánh và analysis

### 3. **LightGBM Training** (`lgbm_major_prediction.ipynb`)
- Sinh dữ liệu 5000 sinh viên
- Feature engineering
- Model training và evaluation
- Export model và encoders

## 📈 Kết quả So sánh

| Phương pháp | Độ chính xác | Ưu điểm | Nhược điểm |
|-------------|--------------|---------|------------|
| **Major Prediction** | 65.0% | Đơn giản, nhanh | Không học từ dữ liệu |
| **LightGBM** | **86.4%** | **Học từ dữ liệu, accuracy cao** | Phức tạp hơn |

## 🎯 Highlights

- ✅ **Cải thiện +21.4%** accuracy với LightGBM
- ✅ **Đầy đủ 5 ngành** thay vì 4 ngành trước đó
- ✅ **Tỷ lệ trùng khớp 93.1%** giữa 2 phương pháp
- ✅ **Interactive dashboard** với Streamlit
- ✅ **Visualizations** chuyên nghiệp với Plotly

## 🛠 Technology Stack

- **Python 3.13**
- **Streamlit:** Web dashboard
- **LightGBM:** Machine learning
- **Pandas/NumPy:** Data processing
- **Plotly:** Interactive visualizations
- **Scikit-learn:** ML utilities

## 📝 Usage

1. **Khởi động dashboard:**
   ```bash
   streamlit run streamlit_dashboard_updated.py
   ```

2. **Train model mới:**
   ```bash
   python enhanced_major_prediction.py
   ```

3. **Xem notebook training:**
   ```bash
   jupyter notebook lgbm_major_prediction.ipynb
   ```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License.

## 👥 Authors

- **Research Team** - *Initial work*

## 🙏 Acknowledgments

- LightGBM team for the excellent ML framework
- Streamlit for the amazing dashboard framework
- Contributors and testers
