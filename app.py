
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random
import re
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
# LỚP QUẢN LÝ DATABASE
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
# LỚP PHÂN TÍCH CAO CẤP (AI ANALYZER)
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
# CẤU HÌNH TỈNH, SLUG, LỊCH
# ==================================================================================
provinces = {
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
minhngoc_slugs = {
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
schedule = {
    "xsmb": [0,1,2,3,4,5,6], "xshcm": [0,5], "xsdt": [0], "xscm": [0], "xsbt": [1], "xsvt": [1],
    "xsbl": [1], "xsdn": [2], "xsct": [2], "xsst": [2], "xstn": [3], "xsag": [3], "xsbth": [3],
    "xsvl": [4], "xsbd": [4], "xstv": [4], "xsla": [5], "xsbp": [5], "xshg": [5], "xstg": [6],
    "xskg": [6], "xsld": [6], "xstth": [0, 6], "xspy": [0], "xsdlk": [1], "xsqnm": [1],
    "xsdng": [2,5], "xskh": [2,6], "xsbdi": [3], "xsqt": [3], "xsqb": [3], "xsgl": [4],
    "xsnt": [4], "xsqng": [5], "xsdno": [5], "xskt": [6]
}

user_agents = ['Mozilla/5.0 ...'] # Rút gọn

# ==================================================================================
# STREAMLIT APP
# ==================================================================================
st.set_page_config(page_title="Lottery AI V9.2", layout="wide")
st.title("Lottery AI V9.2 - Streamlit Edition")

db = DBManager()
log_text = ""

def log(msg, level="INFO"):
    global log_text
    ts = datetime.now().strftime("%H:%M:%S")
    log_text += f"[{ts}] {level}: {msg}\n"

def clean_garbage_data(numbers, target_date):
    if not numbers or len(numbers) < 3: return numbers
    d, m, y = target_date.strftime("%d"), target_date.strftime("%m"), target_date.strftime("%Y")
    if (numbers[0] == d or numbers[0] == str(int(d))) and (numbers[2] == y): return numbers[3:]
    if numbers[0] == y: return numbers[1:]
    return numbers

def generic_parser(html, is_mb):
    soup = BeautifulSoup(html, 'html.parser')
    containers = soup.find_all('table', class_=re.compile(r'result\ntable\kqxs'))
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
    mn_slug = minhngoc_slugs.get(province_code)
    if mn_slug: sources.append({'url': f"https://www.minhngoc.net.vn/ket-qua-xo-so/{mn_slug}/{d_str}-{m_str}-{y_str}.html", 'parser': generic_parser, 'name': 'MN'})
    sources.append({'url': f"https://xoso.com.vn/{province_code}-{d_str}-{m_str}-{y_str}.html", 'parser': generic_parser, 'name': 'XS.VN'})
    sources.append({'url': f"https://xskt.com.vn/{province_code}/ngay-{d_str}-{m_str}-{y_str}", 'parser': generic_parser, 'name': 'XSKT'})
    for src in sources:
        try:
            res = requests.get(src['url'], headers={'User-Agent': random.choice(user_agents)}, timeout=5)
            if res.status_code == 200:
                raw_nums = generic_parser(res.text, is_mb)
                clean_nums = clean_garbage_data(raw_nums, target_date)
                expected = 27 if is_mb else 18
                if clean_nums and len(clean_nums) >= expected:
                    row = {'Ngay': date_display}
                    for idx, n in enumerate(clean_nums[:expected]): row[f'Giai_{idx}'] = n
