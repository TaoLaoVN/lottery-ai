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

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Lottery AI V9.2 Mobile", layout="wide", page_icon="🎲")

# --- THƯ VIỆN AI ---
try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ Chưa cài scikit-learn. Một số tính năng AI sẽ bị hạn chế.")

# ==================================================================================
# CLASS: DATABASE & ANALYZER (LOGIC GIỮ NGUYÊN TỪ V9.2)
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

class AdvancedAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.history = []
        self.full_history = []
        if not self.df.empty:
            data_cols = [c for c in self.df.columns if c != 'Ngay']
            try:
                self.df['DateObj'] = pd.to_datetime(self.df['Ngay'], dayfirst=True)
                self.df = self.df.sort_values(by='DateObj', ascending=True)
            except: pass
            for _, row in self.df.iterrows():
                day_nums = []
                full_day_nums = []
                for col in data_cols:
                    val = str(row[col])
                    if val and val != 'nan':
                        clean = ''.join(filter(str.isdigit, val))
                        if len(clean) >= 2: day_nums.append(clean[-2:])
                        if len(clean) >= 3: full_day_nums.append(clean)
                self.history.append({'date': row['Ngay'], 'nums': day_nums})
                self.full_history.append({'date': row['Ngay'], 'nums': full_day_nums})

    def autocorr_strength(self, number, max_lag=30):
        series = [1 if number in h['nums'] else 0 for h in self.history]
        x = np.array(series, dtype=float)
        n = len(x)
        if n < 10: return 0.0
        x = x - x.mean()
        if x.var() == 0: return 0.0
        full_corr = np.correlate(x, x, mode='full')
        corr = full_corr[full_corr.size//2:] / (np.arange(n, 0, -1) * x.var())
        ac = corr[1:max_lag]
        return float(np.max(ac)) if len(ac) > 0 else 0.0

    def fft_cycle_strength(self, number):
        series = [1 if number in h['nums'] else 0 for h in self.history]
        x = np.array(series, dtype=float)
        n = len(x)
        if n < 10: return 0.0
        fft = np.fft.rfft(x - x.mean())
        ps = np.abs(fft)**2
        ps[0] = 0
        idx = np.argmax(ps)
        return float(ps[idx]) / (n/2) if n > 0 else 0

    def markov_chain_next(self, last_draw_nums):
        next_counts = {str(i).zfill(2): 0 for i in range(100)}
        if not last_draw_nums or len(self.history) < 2: return next_counts
        recent_hist = self.history[-200:]
        for i in range(len(recent_hist) - 1):
            matches = set(recent_hist[i]['nums']).intersection(set(last_draw_nums))
            if len(matches) > 0:
                weight = len(matches) 
                for num in recent_hist[i+1]['nums']:
                    if num in next_counts: next_counts[num] += weight
        return next_counts

    def kmeans_clustering(self, stats_df):
        if not SKLEARN_AVAILABLE: return stats_df
        X = stats_df[['freq', 'gan']].values
        n_clusters = 3 if len(X) >= 3 else 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        stats_df['cluster'] = kmeans.fit_predict(X)
        cluster_scores = {}
        for c in range(n_clusters):
            sub = stats_df[stats_df['cluster'] == c]
            if sub.empty: continue
            cluster_scores[c] = sub['freq'].mean()
        max_score = max(cluster_scores.values()) if cluster_scores else 1
        stats_df['kmeans_score'] = stats_df['cluster'].map(lambda x: cluster_scores.get(x, 0) / max_score)
        return stats_df

    def linear_trend(self):
        trends = {}
        if len(self.history) < 30: return {str(i).zfill(2): 0 for i in range(100)}
        recent_30 = self.history[-30:]
        for num in [str(i).zfill(2) for i in range(100)]:
            y = [1 if num in h['nums'] else 0 for h in recent_30]
            X = np.array(range(len(y))).reshape(-1, 1)
            if SKLEARN_AVAILABLE:
                reg = LinearRegression().fit(X, y)
                trends[num] = reg.coef_[0]
            else: trends[num] = 0
        min_v, max_v = min(trends.values()), max(trends.values())
        norm_trends = {}
        for k, v in trends.items():
            if max_v - min_v == 0: norm_trends[k] = 0
            else: norm_trends[k] = (v - min_v) / (max_v - min_v)
        return norm_trends

    def pair_influence_score(self, last_draw_nums):
        pair_counts = {}
        recent_history = self.history[-100:] 
        for h in recent_history:
            nums = sorted(list(set(h['nums'])))
            for a, b in itertools.combinations(nums, 2):
                pair = f"{a}-{b}"
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        scores = {str(i).zfill(2): 0 for i in range(100)}
        if not last_draw_nums: return scores
        for pair, cnt in pair_counts.items():
            a, b = pair.split('-')
            if a in last_draw_nums: scores[b] += cnt
            if b in last_draw_nums: scores[a] += cnt
        return scores

    def analyze_pairs_list(self, limit_days=100):
        pair_counter = Counter()
        recent = self.history[-limit_days:]
        for h in recent:
            unique_nums = sorted(list(set(h['nums'])))
            for pair in itertools.combinations(unique_nums, 2):
                pair_counter[f"{pair[0]}-{pair[1]}"] += 1
        return pair_counter.most_common(50)

    def predict_position_digit(self, position_from_end, top_k=3):
        counts = {str(i): 0 for i in range(10)}
        recent_history = self.full_history[-200:] 
        for h in recent_history:
            for num_str in h['nums']:
                if len(num_str) >= position_from_end:
                    digit = num_str[-position_from_end]
                    counts[digit] += 1
        sorted_digits = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [d[0] for d in sorted_digits[:top_k]]

    def generate_3d_4d_smart(self, top_2d_list):
        top_tram = self.predict_position_digit(3, top_k=3)
        top_nghin = self.predict_position_digit(4, top_k=2)
        res_3d = []
        res_4d = []
        for num_2d in top_2d_list:
            for cang in top_tram: res_3d.append(f"{cang}{num_2d}")
        for num_3d in res_3d:
            for cang_4 in top_nghin: res_4d.append(f"{cang_4}{num_3d}")
        return res_3d, res_4d

# ==================================================================================
# UI CHÍNH (STREAMLIT)
# ==================================================================================
PROVINCES = {
    "--- MIỀN BẮC ---": "xsmb", "Miền Bắc": "xsmb",
    "--- MIỀN NAM ---": "xshcm", "TP. HCM": "xshcm", "Đồng Tháp": "xsdt", "Cà Mau": "xscm", 
    "Bến Tre": "xsbt", "Vũng Tàu": "xsvt", "Bạc Liêu": "xsbl", "Đồng Nai": "xsdn", 
    "Cần Thơ": "xsct", "Sóc Trăng": "xsst", "Tây Ninh": "xstn", "An Giang": "xsag",
    "--- MIỀN TRUNG ---": "xsdng", "Đà Nẵng": "xsdng", "Khánh Hòa": "xskh"
}

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
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        d = now - timedelta(days=i)
        ds = d.strftime("%d-%m-%Y")
        ds_display = d.strftime("%d/%m/%Y")
        
        # Thử nhiều nguồn
        urls = [
            f"https://xoso.com.vn/{code}-{ds}.html",
            f"https://xskt.com.vn/{code}/ngay-{ds}"
        ]
        
        found = False
        for url in urls:
            try:
                res = requests.get(url, timeout=3)
                if res.status_code == 200:
                    nums = get_nums_from_html(res.text, code=='xsmb')
                    if nums:
                        # Lọc rác
                        check_n = [n for n in nums if n not in [d.strftime("%d"), d.strftime("%m"), d.strftime("%Y")]]
                        expected = 27 if code=='xsmb' else 18
                        if len(check_n) >= expected:
                            db.upsert(ds_display, code, check_n[:expected])
                            count += 1
                            found = True
                            break
            except: pass
        
        progress_bar.progress((i + 1) / days)
        status_text.text(f"Đang quét ngày {ds_display}...")
    
    status_text.text(f"Hoàn tất! Đã cập nhật {count} kỳ.")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# --- GIAO DIỆN ---
st.title("🎲 Lottery AI Ultimate V9.2 (Mobile)")
st.markdown("Hệ thống phân tích xổ số chuyên sâu sử dụng AI, chạy trực tiếp trên nền tảng Cloud.")

# Sidebar
with st.sidebar:
    st.header("1. Cấu hình Dữ liệu")
    prov_name = st.selectbox("Chọn Đài", list(PROVINCES.keys()))
    prov_code = PROVINCES[prov_name]
    
    days_scan = st.slider("Số ngày quét dữ liệu", 10, 100, 30)
    
    if st.button("♻️ Cập nhật Dữ liệu Mới", use_container_width=True):
        with st.spinner("Đang kết nối Server để tải dữ liệu..."):
            scrape_data(prov_code, days_scan)
        st.success("Cập nhật thành công!")

    st.divider()
    st.header("2. Trọng số AI")
    w_freq = st.slider("Tần suất", 0.0, 1.0, 0.2)
    w_gan = st.slider("Lô Gan", 0.0, 1.0, 0.1)
    w_markov = st.slider("Markov (Bạc nhớ)", 0.0, 1.0, 0.25)
    
    btn_run = st.button("🚀 CHẠY PHÂN TÍCH NGAY", type="primary", use_container_width=True)

# Main Content
if btn_run:
    db = DBManager()
    df = db.get_df(prov_code)
    
    if df is None or df.empty:
        st.error("Chưa có dữ liệu! Vui lòng bấm 'Cập nhật Dữ liệu Mới' ở menu bên trái trước.")
    else:
        st.info(f"Đang phân tích {len(df)} kỳ quay của {prov_name}...")
        analyzer = AdvancedAnalyzer(df)
        
        # --- LOGIC PHÂN TÍCH (V9.2) ---
        full_range = [str(i).zfill(2) for i in range(100)]
        stats_df = pd.DataFrame({'so': full_range})
        
        # 1. Cơ bản
        all_nums = [n for h in analyzer.history for n in h['nums']]
        freq = pd.Series(all_nums).value_counts().reset_index()
        freq.columns = ['so', 'freq']
        stats_df = stats_df.merge(freq, on='so', how='left').fillna(0)
        
        # 2. Gan
        draws = [set(h['nums']) for h in analyzer.history]
        gap_dict = {}
        for n in full_range:
            g = 0
            for d_set in draws[::-1]: # Duyệt ngược từ mới nhất
                if n in d_set: break
                g += 1
            gap_dict[n] = g
        stats_df['gan'] = stats_df['so'].map(gap_dict)
        
        # 3. AI
        last_draw_nums = list(draws[-1]) if draws else []
        stats_df['autocorr'] = stats_df['so'].apply(lambda x: analyzer.autocorr_strength(x))
        stats_df['fft'] = stats_df['so'].apply(lambda x: analyzer.fft_cycle_strength(x))
        stats_df['pair_score'] = stats_df['so'].map(analyzer.pair_influence_score(last_draw_nums))
        stats_df['markov_score'] = stats_df['so'].map(analyzer.markov_chain_next(last_draw_nums))
        stats_df = analyzer.kmeans_clustering(stats_df)
        stats_df['trend_score'] = stats_df['so'].map(analyzer.linear_trend())

        # 4. Score
        def norm(col): return col / (col.max() or 1)
        stats_df['score'] = (
            norm(stats_df['freq'])*w_freq + 
            (stats_df['gan']/(stats_df['gan'].max() or 1))*w_gan + 
            norm(stats_df['autocorr'])*0.15 + 
            norm(stats_df['fft'])*0.15 + 
            norm(stats_df['pair_score'])*0.1 + 
            norm(stats_df['markov_score'])*w_markov +
            norm(stats_df.get('kmeans_score', 0))*0.15
        )
        stats_df['final_score'] = (stats_df['score'] * 100).round(2)
        stats_df = stats_df.sort_values(by='final_score', ascending=False)
        
        # --- HIỂN THỊ KẾT QUẢ ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Top 2D", "🔗 Xiên 2", "🔮 3 Càng", "💎 4 Càng", "📊 Biểu đồ"])
        
        with tab1:
            st.dataframe(stats_df[['so', 'final_score', 'freq', 'gan', 'markov_score', 'pair_score']].head(20), use_container_width=True)
            
        with tab2:
            top_pairs = analyzer.analyze_pairs_list(100)
            pair_df = pd.DataFrame(top_pairs, columns=['Cặp Số', 'Số lần về cùng'])
            st.dataframe(pair_df, use_container_width=True)
            
        top_10_2d = stats_df['so'].head(10).tolist()
        smart_3d, smart_4d = analyzer.generate_3d_4d_smart(top_10_2d)
        
        with tab3:
            st.write("Dự đoán 3 càng dựa trên ghép Càng (Hàng trăm) + Top 10 Lô:")
            st.table(pd.DataFrame(smart_3d, columns=["Bộ số 3D"]))
            
        with tab4:
            st.write("Dự đoán 4 càng siêu phẩm:")
            st.table(pd.DataFrame(smart_4d, columns=["Bộ số 4D"]))
            
        with tab5:
            top_20 = stats_df.head(20)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(top_20['so'], top_20['final_score'], color='teal')
            st.pyplot(fig)