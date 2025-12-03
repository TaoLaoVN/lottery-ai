import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random
import re
import concurrent.futures
import sqlite3
import numpy as np
import itertools
from collections import Counter

# --- THƯ VIỆN AI ---
try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==================================================================================
# LỚP QUẢN LÝ DATABASE (GIỮ NGUYÊN)
# ==================================================================================
class DBManager:
    def __init__(self, db_file="lottery_ai_ultimate.db"):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS lottery_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                province_code TEXT,
                numbers TEXT,
                UNIQUE(date, province_code)
            )
        ''')
        self.conn.commit()

    def upsert_result(self, date, province_code, numbers_list):
        c = self.conn.cursor()
        nums_str = ",".join(numbers_list)
        try:
            c.execute('''
                INSERT INTO lottery_results (date, province_code, numbers) 
                VALUES (?, ?, ?)
                ON CONFLICT(date, province_code) 
                DO UPDATE SET numbers=excluded.numbers
            ''', (date, province_code, nums_str))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"DB Error: {e}")
            return False

    def get_data_frame(self, province_code):
        query = "SELECT date, numbers FROM lottery_results WHERE province_code = ? ORDER BY date DESC"
        df = pd.read_sql_query(query, self.conn, params=(province_code,))
        if df.empty: return None
        
        parsed_data = []
        for _, row in df.iterrows():
            nums = row['numbers'].split(',')
            row_dict = {'Ngay': row['date']}
            for idx, n in enumerate(nums):
                row_dict[f'Giai_{idx}'] = n
            parsed_data.append(row_dict)
        return pd.DataFrame(parsed_data)

# ==================================================================================
# LỚP PHÂN TÍCH CAO CẤP (GIỮ NGUYÊN)
# ==================================================================================
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
            for cang in top_tram:
                res_3d.append(f"{cang}{num_2d}")
        for num_3d in res_3d:
            for cang_4 in top_nghin:
                res_4d.append(f"{cang_4}{num_3d}")
        return res_3d, res_4d

# ==================================================================================
# CẤU HÌNH & HELPER FUNCTIONS
# ==================================================================================

PROVINCES = {
    "--- MIỀN BẮC ---": "xsmb", "Miền Bắc": "xsmb",
    "--- MIỀN NAM ---": "xshcm", "TP. HCM": "xshcm", "Đồng Tháp": "xsdt", "Cà Mau": "xscm", 
    "Bến Tre": "xsbt", "Vũng Tàu": "xsvt", "Bạc Liêu": "xsbl", "Đồng Nai": "xsdn", 
    "Cần Thơ": "xsct", "Sóc Trăng": "xsst", "Tây Ninh": "xstn", "An Giang": "xsag", 
    "Bình Thuận": "xsbth", "Vĩnh Long": "xsvl", "Bình Dương": "xsbd", "Trà Vinh": "xstv", 
    "Long An": "xsla", "Bình Phước": "xsbp", "Hậu Giang": "xshg", "Tiền Giang": "xstg", 
    "Kiên Giang": "xskg", "Đà Lạt": "xsld",
    "--- MIỀN TRUNG ---": "xsdng", "Huế": "xstth", "Phú Yên": "xspy", "Đắk Lắk": "xsdlk", 
    "Quảng Nam": "xsqnm", "Đà Nẵng": "xsdng", "Khánh Hòa": "xskh", "Bình Định": "xsbdi", 
    "Quảng Trị": "xsqt", "Quảng Bình": "xsqb", "Gia Lai": "xsgl", "Ninh Thuận": "xsnt", 
    "Quảng Ngãi": "xsqng", "Đắk Nông": "xsdno", "Kon Tum": "xskt"
}

MINHNGOC_SLUGS = {
    "xsmb": "mien-bac", "xshcm": "tp-hcm", "xsdt": "dong-thap", "xscm": "ca-mau",
    "xsbt": "ben-tre", "xsvt": "vung-tau", "xsbl": "bac-lieu", "xsdn": "dong-nai", 
    "xsct": "can-tho", "xsst": "soc-trang", "xstn": "tay-ninh", "xsag": "an-giang", 
    "xsbth": "binh-thuan", "xsvl": "vinh-long", "xsbd": "binh-duong", "xstv": "tra-vinh",
    "xsla": "long-an", "xsbp": "binh-phuoc", "xshg": "hau-giang", "xstg": "tien-giang", 
    "xskg": "kien-giang", "xsld": "da-lat", "xstth": "thua-thien-hue", "xspy": "phu-yen",
    "xsdlk": "dak-lak", "xsqnm": "quang-nam", "xsdng": "da-nang", "xskh": "khanh-hoa",
    "xsbdi": "binh-dinh", "xsqt": "quang-tri", "xsqb": "quang-binh", "xsgl": "gia-lai", 
    "xsnt": "ninh-thuan", "xsqng": "quang-ngai", "xsdno": "dak-nong", "xskt": "kon-tum"
}

SCHEDULE = {
    "xsmb": [0,1,2,3,4,5,6], "xshcm": [0,5], "xsdt": [0], "xscm": [0], "xsbt": [1], "xsvt": [1],
    "xsbl": [1], "xsdn": [2], "xsct": [2], "xsst": [2], "xstn": [3], "xsag": [3], "xsbth": [3],
    "xsvl": [4], "xsbd": [4], "xstv": [4], "xsla": [5], "xsbp": [5], "xshg": [5], "xstg": [6],
    "xskg": [6], "xsld": [6], "xstth": [0, 6], "xspy": [0], "xsdlk": [1], "xsqnm": [1],
    "xsdng": [2,5], "xskh": [2,6], "xsbdi": [3], "xsqt": [3], "xsqb": [3], "xsgl": [4],
    "xsnt": [4], "xsqng": [5], "xsdno": [5], "xskt": [6]
}

USER_AGENTS = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36']

def clean_garbage_data(numbers, target_date):
    if not numbers or len(numbers) < 3: return numbers
    d, m, y = target_date.strftime("%d"), target_date.strftime("%m"), target_date.strftime("%Y")
    if (numbers[0] == d or numbers[0] == str(int(d))) and (numbers[2] == y): return numbers[3:]
    if numbers[0] == y: return numbers[1:]
    return numbers

def generic_parser(html, is_mb):
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('table', class_=re.compile(r'result|table|kqxs'))
    for tbl in containers:
        cells = tbl.find_all(['td', 'span'])
        nums = []
        for c in cells:
            txt = c.get_text().strip()
            if any(x in txt.lower() for x in ['giải', 'đb', 'ngày', 'tháng', 'vé']): continue
            found = re.findall(r'\b\d{2,6}\b', txt)
            nums.extend(found)
        expected = 27 if is_mb else 18
        if len(nums) >= expected: return nums
    return []

def fetch_single_day(target_date, province_code):
    d_str = target_date.strftime("%d"); m_str = target_date.strftime("%m"); y_str = target_date.strftime("%Y")
    date_display = target_date.strftime("%d/%m/%Y")
    is_mb = (province_code == 'xsmb')
    sources = []
    mn_slug = MINHNGOC_SLUGS.get(province_code)
    if mn_slug: sources.append({'url': f"https://www.minhngoc.net.vn/ket-qua-xo-so/{mn_slug}/{d_str}-{m_str}-{y_str}.html", 'parser': generic_parser, 'name': 'MN'})
    sources.append({'url': f"https://xoso.com.vn/{province_code}-{d_str}-{m_str}-{y_str}.html", 'parser': generic_parser, 'name': 'XS.VN'})
    sources.append({'url': f"https://xskt.com.vn/{province_code}/ngay-{d_str}-{m_str}-{y_str}", 'parser': generic_parser, 'name': 'XSKT'})
    
    for src in sources:
        try:
            res = requests.get(src['url'], headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=5)
            if res.status_code == 200:
                raw_nums = generic_parser(res.text, is_mb)
                clean_nums = clean_garbage_data(raw_nums, target_date)
                expected = 27 if is_mb else 18
                if clean_nums and len(clean_nums) >= expected:
                    row = {'Ngay': date_display}
                    for idx, n in enumerate(clean_nums[:expected]): row[f'Giai_{idx}'] = n
                    return row, date_display, src['name']
        except: continue
    return None, date_display, "Fail"

# ==================================================================================
# STREAMLIT APP
# ==================================================================================

def main():
    st.set_page_config(page_title="Phân tích số liệu AI", layout="wide")
    st.title("Phân tích số liệu AI - MaTools")

    # --- KHỞI TẠO ---
    if 'db' not in st.session_state:
        st.session_state.db = DBManager()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # --- SIDEBAR (CONTROL CENTER) ---
    with st.sidebar:
        st.header("⚙️ Trung tâm điều khiển")
        
        # 1. Nguồn Dữ liệu
        st.subheader("1. Nguồn Dữ liệu")
        province_options = [k for k in PROVINCES.keys() if not k.startswith("---")]
        selected_province_name = st.selectbox("Chọn Đài/Tỉnh", province_options, index=0)
        selected_province_code = PROVINCES.get(selected_province_name)

        # Cập nhật Data (Scraping)
        with st.expander("♻️ Cập nhật Data Online"):
            days_to_scrape = st.number_input("Số ngày quét", min_value=1, max_value=5000, value=1)
            if st.button("Bắt đầu quét"):
                with st.spinner(f"Đang quét {selected_province_name}..."):
                    valid_days = SCHEDULE.get(selected_province_code, [0,1,2,3,4,5,6])
                    dates = []
                    curr = datetime.now()
                    for i in range(days_to_scrape):
                        d = curr - timedelta(days=i)
                        if d.weekday() in valid_days: dates.append(d)
                    
                    if not dates:
                        st.warning("Không có lịch quay cho đài này trong khoảng thời gian chọn.")
                    else:
                        progress_bar = st.progress(0)
                        success_count = 0
                        total = len(dates)
                        
                        # Chạy đa luồng (tương tự bản cũ)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                            future_to_date = {executor.submit(fetch_single_day, d, selected_province_code): d for d in dates}
                            for i, future in enumerate(concurrent.futures.as_completed(future_to_date)):
                                row, d_str, src = future.result()
                                if row:
                                    nums = [v for k,v in row.items() if k.startswith('Giai_')]
                                    if st.session_state.db.upsert_result(d_str, selected_province_code, nums):
                                        success_count += 1
                                progress_bar.progress((i + 1) / total)
                        
                        st.success(f"Quét xong! Thành công: {success_count}/{total}")

        # Load Data Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load dữ liệu DB", use_container_width=True):
                df = st.session_state.db.get_data_frame(selected_province_code)
                if df is not None:
                    st.session_state.data = df
                    st.success(f"Loaded {len(df)} kỳ từ DB.")
                else:
                    st.warning("Database chưa có dữ liệu cho đài này.")
        
        with col2:
             uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], label_visibility="collapsed")
             if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if 'Ngay' in df.columns:
                        df['DateObj'] = pd.to_datetime(df['Ngay'], dayfirst=True, errors='coerce')
                        df = df.sort_values(by='DateObj', ascending=False)
                    st.session_state.data = df
                    st.success(f"Loaded {len(df)} dòng.")
                except Exception as e:
                    st.error(f"Lỗi file: {e}")

        # 2. Cấu hình AI
        st.subheader("2. Cấu hình AI")
        use_autocorr = st.checkbox("AutoCorrelation", value=True)
        use_fft = st.checkbox("FFT (Sóng)", value=True)
        use_markov = st.checkbox("Markov Chain", value=True)
        use_pair = st.checkbox("Tương sinh (Pair)", value=True)
        use_kmeans = st.checkbox("K-Means", value=True)
        use_linreg = st.checkbox("Linear Trend", value=True)
        
        # Các trọng số mặc định (ẩn đi để giao diện gọn, có thể hardcode như bản gốc)
        config = {
            'freq': 1, 'gan': 1,
            'autocorr': 1 if use_autocorr else 0,
            'fft': 1 if use_fft else 0,
            'markov': 1 if use_markov else 0,
            'pair': 1 if use_pair else 0,
            'kmeans': 1 if use_kmeans else 0,
            'linreg': 1 if use_linreg else 0
        }

        # Nút Chạy Phân Tích
        if st.button("🚀 CHẠY PHÂN TÍCH", type="primary", use_container_width=True):
            if st.session_state.data is None:
                st.error("Chưa có dữ liệu! Vui lòng Load từ DB hoặc File.")
            else:
                run_analysis(st.session_state.data, config)

    # --- MAIN CONTENT ---
    if st.session_state.results:
        res = st.session_state.results
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["2D Analysis", "Xiên/Cặp", "3 Càng", "4 Càng", "Biểu Đồ"])
        
        with tab1:
            st.markdown("### Bảng Phân Tích Bạch Thủ/Lô")
            st.dataframe(res['stats_df'], use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown("### Top Cặp Số Hay Về Cùng Nhau")
            pairs_df = pd.DataFrame(res['top_pairs'], columns=['Cặp Số', 'Số lần'])
            pairs_df.index += 1
            st.dataframe(pairs_df, use_container_width=True)
            
        with tab3:
            st.markdown("### Dự Đoán 3 Càng (Ghép Cầu)")
            df_3d = pd.DataFrame(res['smart_3d'], columns=['Số 3D'])
            df_3d['Nguồn'] = df_3d['Số 3D'].apply(lambda x: f"Gốc {x[-2:]}")
            st.dataframe(df_3d, use_container_width=True)

        with tab4:
            st.markdown("### Dự Đoán 4 Càng (Ghép Cầu)")
            df_4d = pd.DataFrame(res['smart_4d'], columns=['Số 4D'])
            df_4d['Nguồn'] = df_4d['Số 4D'].apply(lambda x: f"Gốc {x[-2:]}")
            st.dataframe(df_4d, use_container_width=True)
            
        with tab5:
            st.markdown("### Biểu Đồ Top 20 Số Tiềm Năng")
            top_20 = res['stats_df'].head(20)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['red']*3 + ['teal']*(len(top_20)-3)
            ax.bar(top_20['so'], top_20['final_score'], color=colors)
            ax.set_title("TOP 20 SỐ TIỀM NĂNG NHẤT")
            st.pyplot(fig)

    else:
        st.info("👈 Vui lòng chọn nguồn dữ liệu và nhấn 'Chạy Phân Tích' ở menu bên trái.")

def run_analysis(df_original, config):
    with st.spinner("Đang khởi chạy AI Engine V9.2..."):
        try:
            full_range = [str(i).zfill(2) for i in range(100)]
            stats_df = pd.DataFrame({'so': full_range})
            
            # Helper
            def get_all_numbers(df):
                cols = [c for c in df.columns if c != 'Ngay']
                flat = []
                for _, r in df.iterrows():
                    for c in cols:
                        v = str(r[c])
                        if v and v!='nan':
                            cl = ''.join(filter(str.isdigit, v))
                            if len(cl)>=2: flat.append(cl[-2:])
                return flat

            # 1. Cơ bản
            all_nums = get_all_numbers(df_original)
            freq = pd.Series(all_nums).value_counts().reset_index()
            freq.columns = ['so', 'freq']
            stats_df = stats_df.merge(freq, on='so', how='left').fillna(0)
            
            # 2. Gan
            gap_dict = {}
            draws = []
            cols = [c for c in df_original.columns if c.startswith('Giai')]
            for _, row in df_original.iterrows():
                d_n = set()
                for c in cols:
                    v = str(row[c])
                    if len(v)>=2 and v!='nan': d_n.add(v[-2:].zfill(2))
                draws.append(d_n)
            for n in full_range:
                g = 0
                for d_set in draws:
                    if n in d_set: break
                    g += 1
                gap_dict[n] = g
            stats_df['gan'] = stats_df['so'].map(gap_dict)

            # 3. AI Analysis
            analyzer = AdvancedAnalyzer(df_original)
            
            stats_df['autocorr'] = stats_df['so'].apply(lambda x: analyzer.autocorr_strength(x))
            stats_df['fft'] = stats_df['so'].apply(lambda x: analyzer.fft_cycle_strength(x))
            
            last_draw_nums = list(draws[0]) if draws else []
            pair_scores = analyzer.pair_influence_score(last_draw_nums)
            stats_df['pair_score'] = stats_df['so'].map(pair_scores)
            
            stats_df['markov_score'] = stats_df['so'].map(analyzer.markov_chain_next(last_draw_nums))
            
            stats_df = analyzer.kmeans_clustering(stats_df)
            trend_scores = analyzer.linear_trend()
            stats_df['trend_score'] = stats_df['so'].map(trend_scores)

            # 4. Tính điểm
            def norm(col): return col / (col.max() or 1)
            
            w_freq = 0.15 if config['freq'] else 0
            w_gan = 0.10 if config['gan'] else 0
            w_ac = 0.10 if config['autocorr'] else 0
            w_fft = 0.10 if config['fft'] else 0
            w_pair = 0.10 if config['pair'] else 0
            w_markov = 0.20 if config['markov'] else 0
            w_kmeans = 0.15 if config['kmeans'] else 0
            w_linreg = 0.10 if config['linreg'] else 0
            
            stats_df['score'] = (
                norm(stats_df['freq'])*w_freq + 
                (stats_df['gan']/(stats_df['gan'].max() or 1))*w_gan + 
                norm(stats_df['autocorr'])*w_ac + 
                norm(stats_df['fft'])*w_fft + 
                norm(stats_df['pair_score'])*w_pair + 
                norm(stats_df['markov_score'])*w_markov +
                norm(stats_df.get('kmeans_score', 0))*w_kmeans +
                norm(stats_df['trend_score'])*w_linreg
            )
            
            stats_df['final_score'] = (stats_df['score'] * 100).round(2)
            stats_df = stats_df.sort_values(by='final_score', ascending=False)
            
            # Format display dataframe
            display_df = stats_df[['so', 'final_score', 'freq', 'gan', 'autocorr', 'fft', 'markov_score', 'pair_score', 'kmeans_score']].copy()
            display_df.columns = ['Số', 'ĐIỂM', 'Freq', 'Gan', 'AC', 'FFT', 'Markov', 'Pair', 'KMS']
            
            # Analyze Pairs
            top_pairs = analyzer.analyze_pairs_list(limit_days=100)
            
            # 3D/4D
            top_10_2d = stats_df['so'].head(10).tolist()
            smart_3d, smart_4d = analyzer.generate_3d_4d_smart(top_10_2d)

            # Save to session state
            st.session_state.results = {
                'stats_df': display_df,
                'top_pairs': top_pairs,
                'smart_3d': smart_3d,
                'smart_4d': smart_4d
            }
            st.success("Phân tích hoàn tất!")

        except Exception as e:
            st.error(f"Lỗi trong quá trình phân tích: {e}")
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()

