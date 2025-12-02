import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import threading
import time
import random
import re
import sqlite3
import numpy as np 
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import pickle

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Lottery AI V10.0 - Ensemble Models", layout="wide", page_icon="🧠")

# --- THƯ VIỆN AI NÂNG CAO ---
try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    st.error(f"⚠️ Thiếu thư viện AI: {e}")

# ==================================================================================
# CLASS: DATABASE
# ==================================================================================
class DBManager:
    def __init__(self, db_file="lottery.db"):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS lottery_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, province_code TEXT, numbers TEXT,
                UNIQUE(date, province_code))
        ''')
        self.conn.commit()

    def upsert(self, date, code, nums):
        try:
            self.conn.execute('INSERT OR REPLACE INTO lottery_results (date, province_code, numbers) VALUES (?, ?, ?)', 
                             (date, code, ",".join(nums)))
            self.conn.commit()
            return True
        except: return False

    def get_df(self, code):
        df = pd.read_sql_query("SELECT date, numbers FROM lottery_results WHERE province_code = ? ORDER BY date DESC", self.conn, params=(code,))
        if df.empty: return None
        data = []
        for _, r in df.iterrows():
            row = {'Ngay': r['date']}
            for i, n in enumerate(r['numbers'].split(',')): row[f'Giai_{i}'] = n
            data.append(row)
        return pd.DataFrame(data)

# ==================================================================================
# CLASS: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG CHO AI)
# ==================================================================================
class FeatureEngine:
    def __init__(self, df):
        self.df = df.copy()
        if not self.df.empty:
            try:
                self.df['DateObj'] = pd.to_datetime(self.df['Ngay'], dayfirst=True)
                self.df = self.df.sort_values(by='DateObj', ascending=True)
            except: pass
        
        self.history = []
        data_cols = [c for c in self.df.columns if c != 'Ngay']
        for _, row in self.df.iterrows():
            day_nums = []
            for col in data_cols:
                val = str(row[col])
                clean = ''.join(filter(str.isdigit, val))
                if len(clean) >= 2: day_nums.append(clean[-2:])
            self.history.append(day_nums)

    def extract_features(self, target_num, history_slice):
        """
        Trích xuất đặc trưng của 1 con số tại thời điểm cụ thể dựa trên lịch sử trước đó.
        Features: Freq, Gan, AutoCorr, Last_Appear_Distance, v.v.
        """
        # 1. Frequency (trong 30 kỳ gần nhất của slice)
        recent_30 = history_slice[-30:]
        flat_30 = [n for sub in recent_30 for n in sub]
        freq = flat_30.count(target_num)
        
        # 2. Gan (Khoảng cách chưa về)
        gan = 0
        for sub in reversed(history_slice):
            if target_num in sub: break
            gan += 1
            
        # 3. AutoCorrelation (Đơn giản hóa)
        series = [1 if target_num in sub else 0 for sub in history_slice[-50:]]
        if len(series) > 10 and np.var(series) > 0:
            # Lag 1 autocorrelation
            ac = pd.Series(series).autocorr(lag=1)
            ac = 0 if np.isnan(ac) else ac
        else:
            ac = 0
            
        return [freq, gan, ac]

    def create_training_dataset(self, lookback_days=100):
        """
        Tạo dữ liệu huấn luyện: 
        X = Các đặc trưng ngày hôm qua
        y = Kết quả ngày hôm nay (1: về, 0: không về)
        """
        X = []
        y = []
        
        # Chỉ lấy 100 kỳ gần nhất để train cho nhanh
        available_hist = self.history
        if len(available_hist) < 50: return None, None
        
        start_idx = len(available_hist) - lookback_days if len(available_hist) > lookback_days else 50
        
        for i in range(start_idx, len(available_hist)):
            past_data = available_hist[:i] # Dữ liệu quá khứ tính đến ngày i-1
            current_result = available_hist[i] # Kết quả thực tế ngày i
            
            # Lấy mẫu ngẫu nhiên 10 số để tạo data train (tránh quá tải)
            # Bao gồm cả số về và số không về
            sample_nums = set(current_result) # Positive samples
            while len(sample_nums) < 20: # Thêm Negative samples
                sample_nums.add(str(random.randint(0,99)).zfill(2))
            
            for num in sample_nums:
                features = self.extract_features(num, past_data)
                label = 1 if num in current_result else 0
                X.append(features)
                y.append(label)
                
        return np.array(X), np.array(y)

    def get_current_features(self):
        """Lấy đặc trưng ngày mới nhất để dự đoán"""
        X_pred = []
        nums_map = []
        for i in range(100):
            num = str(i).zfill(2)
            feat = self.extract_features(num, self.history)
            X_pred.append(feat)
            nums_map.append(num)
        return np.array(X_pred), nums_map

# ==================================================================================
# CLASS: MODEL MANAGER (QUẢN LÝ MÔ HÌNH)
# ==================================================================================
class ModelManager:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train_models(self, X, y, selected_models):
        if len(X) == 0: return
        
        # Scale dữ liệu
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # 1. Random Forest
        if 'Random Forest' in selected_models:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_scaled, y)
            self.models['RF'] = rf
            
        # 2. Gradient Boosting (XGBoost)
        if 'XGBoost' in selected_models:
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            xgb_model.fit(X_scaled, y)
            self.models['XGB'] = xgb_model
            
        # 3. Neural Network (MLP)
        if 'Neural Network (MLP)' in selected_models:
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
            mlp.fit(X_scaled, y)
            self.models['MLP'] = mlp
            
        # 4. Linear Regression (Base)
        if 'Linear Regression' in selected_models:
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            self.models['LR'] = lr

    def predict_ensemble(self, X_pred):
        if not self.models: return np.zeros(len(X_pred))
        
        scaler = self.scalers.get('main')
        if scaler:
            X_scaled = scaler.transform(X_pred)
        else:
            X_scaled = X_pred
            
        final_pred = np.zeros(len(X_pred))
        count = 0
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            final_pred += pred
            count += 1
            
        return final_pred / count if count > 0 else final_pred

# ==================================================================================
# UI & LOGIC CHÍNH
# ==================================================================================
# ... (Giữ nguyên các hàm scrape, parse HTML cũ) ...
def get_nums_from_html(html, is_mb):
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('table', class_=re.compile(r'result|table|kqxs'))
    for tbl in containers:
        cells = tbl.find_all(['td', 'span'])
        nums = []
        for c in cells:
            txt = c.get_text().strip()
            if any(x in txt.lower() for x in ['giải', 'đb', 'ngày']): continue
            found = re.findall(r'\b\d{2,6}\b', txt)
            nums.extend(found)
        expected = 27 if is_mb else 18
        if len(nums) >= expected: return nums
    return []

def scrape_data(code, days):
    db = DBManager()
    count = 0
    now = datetime.now()
    progress = st.progress(0)
    for i in range(days):
        d = now - timedelta(days=i)
        ds = d.strftime("%d-%m-%Y"); ds_disp = d.strftime("%d/%m/%Y")
        url = f"https://xoso.com.vn/{code}-{ds}.html"
        try:
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                nums = get_nums_from_html(res.text, code=='xsmb')
                if nums:
                    check = [n for n in nums if n not in [d.strftime("%d"), d.strftime("%m")]]
                    expected = 27 if code=='xsmb' else 18
                    if len(check) >= expected:
                        db.upsert(ds_disp, code, check[:expected])
                        count += 1
        except: pass
        progress.progress((i+1)/days)
    st.success(f"Đã cập nhật {count} kỳ.")

# --- CACHING MODEL TRAINING ---
@st.cache_resource
def train_ai_manager(df_json, selected_models):
    # df_json là trick để cache dataframe (hashable)
    df = pd.read_json(df_json)
    fe = FeatureEngine(df)
    X, y = fe.create_training_dataset(lookback_days=200)
    
    manager = ModelManager()
    if X is not None:
        manager.train_models(X, y, selected_models)
    
    return manager, fe

# --- GIAO DIỆN ---
st.title("🧠 Lottery AI V10.0 - Neural & Ensemble")

with st.sidebar:
    st.header("1. Dữ liệu")
    PROVINCES = {"Miền Bắc": "xsmb", "TP.HCM": "xshcm", "Đồng Nai": "xsdn", "Đà Nẵng": "xsdng"}
    prov_name = st.selectbox("Chọn Đài", list(PROVINCES.keys()))
    prov_code = PROVINCES[prov_name]
    if st.button("Cập nhật Data"): scrape_data(prov_code, 30)
    
    st.divider()
    st.header("2. Mô hình AI")
    models_opt = st.multiselect(
        "Chọn thuật toán tham gia dự đoán:",
        ["Random Forest", "XGBoost", "Neural Network (MLP)", "Linear Regression"],
        default=["Random Forest", "XGBoost"]
    )
    
    st.info("💡 Mẹo: Random Forest & XGBoost thường cho kết quả tốt nhất với dữ liệu dạng bảng.")
    
    btn_run = st.button("🚀 KÍCH HOẠT AI", type="primary")

if btn_run:
    db = DBManager()
    df = db.get_df(prov_code)
    
    if df is not None:
        st.write(f"Đang huấn luyện mô hình trên {len(df)} kỳ quay... (Tiến trình này được Cache)")
        
        # Train & Cache Models
        manager, fe = train_ai_manager(df.to_json(), models_opt)
        
        # Predict Today
        X_pred, nums_map = fe.get_current_features()
        scores = manager.predict_ensemble(X_pred)
        
        # Create Result DataFrame
        res_df = pd.DataFrame({'so': nums_map, 'ai_score': scores})
        
        # Combine with basic stats
        all_nums = [n for day in fe.history for n in day]
        freq = pd.Series(all_nums).value_counts().reset_index()
        freq.columns = ['so', 'freq']
        res_df = res_df.merge(freq, on='so', how='left').fillna(0)
        
        # Calculate Final Score (Hybrid: AI + Traditional Stats)
        # AI score thường từ 0-1 (hoặc thấp hơn), cần scale lên
        res_df['final_score'] = (res_df['ai_score'] * 70) + (res_df['freq']/res_df['freq'].max() * 30)
        res_df = res_df.sort_values(by='final_score', ascending=False)
        
        # --- DISPLAY ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Dự đoán của AI Ensemble")
            st.dataframe(res_df.head(20).style.background_gradient(subset=['final_score'], cmap='Greens'), use_container_width=True)
            
        with col2:
            st.subheader("📊 Độ tin cậy mô hình")
            st.write("Các mô hình đang chạy:")
            for m in manager.models:
                st.write(f"- ✅ {m}")
            
            # Simple Chart
            fig, ax = plt.subplots()
            top_10 = res_df.head(10)
            ax.bar(top_10['so'], top_10['final_score'], color='purple')
            st.pyplot(fig)
            
        # --- APRIORI (CẶP SỐ) ---
        st.divider()
        st.subheader("🔗 Phân tích Cặp Số (Association Rules)")
        # Logic đơn giản hóa cho Streamlit
        pair_counts = Counter()
        for day in fe.history[-100:]:
            u = sorted(list(set(day)))
            for p in itertools.combinations(u, 2):
                pair_counts[f"{p[0]}-{p[1]}"] += 1
        
        top_pairs = pair_counts.most_common(10)
        cols = st.columns(5)
        for i, (p, c) in enumerate(top_pairs):
            cols[i%5].metric(label=f"Cặp {p}", value=f"{c} lần")

    else:
        st.error("Chưa có dữ liệu.")
