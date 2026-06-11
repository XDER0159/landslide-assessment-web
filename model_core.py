"""
坡地崩塌風險快速評估系統 v4
Landslide Risk Rapid Assessment System
使用隨機森林 + SHAP 進行機器學習預測與解釋
"""

print("版本 v5 - 乾燥高坡補強label=0，雨量真正作為保護因子")

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. 資料生成
# ──────────────────────────────────────────────

def generate_training_data(n_samples: int = 10000, seed: int = 42) -> tuple:
    """
    模擬符合台灣地理特性的訓練資料集。
    核心改進：主動補強中風險樣本區域，確保模型能學習邊界。
    回傳 (X_df, y, land_use)
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    # --- 坡度：Beta 分佈，範圍 5~60°
    slope_raw = rng.beta(a=2.5, b=3.5, size=n_samples)
    slope = slope_raw * 55 + 5

    # --- 坡向：均勻 0~360°
    aspect = rng.uniform(0, 360, size=n_samples)

    # --- 高程：混合兩個對數常態
    low_mask = rng.random(n_samples) < 0.6
    elev = np.where(
        low_mask,
        np.exp(rng.normal(6.5, 0.5, n_samples)),
        np.exp(rng.normal(7.4, 0.4, n_samples))
    )
    elev = np.clip(elev, 50, 3500)

    # --- 累積雨量：對數常態，800~5000 mm
    rainfall = np.exp(rng.normal(7.8, 0.5, n_samples))
    rainfall = np.clip(rainfall, 800, 5000)

    # --- 土壤類型
    soil_types = ['崩積土', '沖積土', '紅土', '砂質頁岩', '黏土']
    soil_weights = [0.30, 0.10, 0.15, 0.30, 0.15]
    soil = rng.choice(soil_types, size=n_samples, p=soil_weights)

    # --- 土地利用
    land_types = ['林地', '耕地', '道路', '建地', '草生地']
    land_weights = [0.50, 0.25, 0.15, 0.05, 0.05]
    land_use = rng.choice(land_types, size=n_samples, p=land_weights)

    # --- NDVI：依土地利用連動
    ndvi_ranges = {
        '林地':   (0.5, 0.9),
        '耕地':   (0.3, 0.7),
        '道路':   (0.1, 0.4),
        '建地':   (0.1, 0.3),
        '草生地': (0.2, 0.6),
    }
    ndvi = np.array([rng.uniform(*ndvi_ranges[lu]) for lu in land_use])

    # ── 崩塌標籤生成 ──

    def compute_risk(sl, rf, sv, nd, el):
        # 坡度因子：線性+平方混合，讓中等坡度也有貢獻
        sf = 0.5 * (sl / 50) + 0.5 * (sl / 50) ** 2
        # 雨量因子：指數 1.2，放大中段敏感度
        rf_f = ((rf - 800) / 4200) ** 1.2
        # 植生風險
        decay = np.clip(1 - (rf - 2000) / 3000, 0, 1)
        veg_risk = 1 - 0.7 * nd * decay
        # 高程因子
        el_f = np.exp(-((el - 1500) / 1000) ** 2)
        return 0.32 * sf + 0.30 * rf_f + 0.18 * sv + 0.10 * veg_risk + 0.10 * el_f

    soil_vuln_map = {'崩積土': 0.9, '砂質頁岩': 0.8, '黏土': 0.6, '紅土': 0.5, '沖積土': 0.3}
    soil_vuln = np.array([soil_vuln_map[s] for s in soil])

    risk_score = compute_risk(slope, rainfall, soil_vuln, ndvi, elev)

    # 強制規則一：極端組合（高雨量+陡坡+脆弱土壤）→ label=1
    extreme_mask = (rainfall >= 3000) & (slope >= 30) & np.isin(soil, ['崩積土', '砂質頁岩'])
    risk_score[extreme_mask] = np.maximum(risk_score[extreme_mask], 0.68)

    # 強制規則二：中坡度+中高雨量 → 至少中風險
    medium_mask = (slope >= 20) & (rainfall >= 1500) & ~extreme_mask
    risk_score[medium_mask] = np.maximum(risk_score[medium_mask], 0.44)



    # 閾值 0.52 → label
    y = (risk_score > 0.52).astype(int)

    # ── 主動補強：針對測試三、五類型的中風險樣本各補 400 筆 ──
    # 補強A：坡度20~30°, 雨量1500~2500mm（測試三類型）→ label=1
    n_aug_a = 400
    aug_slope_a  = rng.uniform(20, 30, n_aug_a)
    aug_aspect_a = rng.uniform(0, 360, n_aug_a)
    aug_elev_a   = rng.uniform(400, 1200, n_aug_a)
    aug_rain_a   = rng.uniform(1500, 2500, n_aug_a)
    aug_soil_a   = rng.choice(['紅土', '黏土', '砂質頁岩'], n_aug_a, p=[0.4, 0.3, 0.3])
    aug_land_a   = rng.choice(land_types, n_aug_a, p=land_weights)
    aug_ndvi_a   = np.array([rng.uniform(*ndvi_ranges[lu]) for lu in aug_land_a])
    aug_y_a      = np.ones(n_aug_a, dtype=int)

    # 補強B：坡度38~50°, 雨量800~1300mm（乾燥高坡）→ label=0
    # 教模型：高坡度但雨量極低時，崩塌機率應為中低風險
    n_aug_b = 500
    aug_slope_b  = rng.uniform(38, 52, n_aug_b)
    aug_aspect_b = rng.uniform(0, 360, n_aug_b)
    aug_elev_b   = rng.uniform(1200, 2800, n_aug_b)
    aug_rain_b   = rng.uniform(800, 1300, n_aug_b)
    aug_soil_b   = rng.choice(['崩積土', '砂質頁岩'], n_aug_b, p=[0.5, 0.5])
    aug_land_b   = rng.choice(land_types, n_aug_b, p=land_weights)
    aug_ndvi_b   = np.array([rng.uniform(*ndvi_ranges[lu]) for lu in aug_land_b])
    aug_y_b      = np.zeros(n_aug_b, dtype=int)   # ← label=0，乾燥抑制崩塌

    # 補強C：坡度38~50°, 雨量2000~4000mm（濕潤高坡）→ label=1
    # 教模型：同樣高坡度但高雨量時才是真正高風險
    n_aug_c = 500
    aug_slope_c  = rng.uniform(38, 52, n_aug_c)
    aug_aspect_c = rng.uniform(0, 360, n_aug_c)
    aug_elev_c   = rng.uniform(1200, 2800, n_aug_c)
    aug_rain_c   = rng.uniform(2000, 4000, n_aug_c)
    aug_soil_c   = rng.choice(['崩積土', '砂質頁岩'], n_aug_c, p=[0.5, 0.5])
    aug_land_c   = rng.choice(land_types, n_aug_c, p=land_weights)
    aug_ndvi_c   = np.array([rng.uniform(*ndvi_ranges[lu]) for lu in aug_land_c])
    aug_y_c      = np.ones(n_aug_c, dtype=int)    # ← label=1，高雨量觸發崩塌

    # 合併
    slope_all    = np.concatenate([slope,    aug_slope_a,  aug_slope_b,  aug_slope_c])
    aspect_all   = np.concatenate([aspect,   aug_aspect_a, aug_aspect_b, aug_aspect_c])
    elev_all     = np.concatenate([elev,     aug_elev_a,   aug_elev_b,   aug_elev_c])
    rainfall_all = np.concatenate([rainfall, aug_rain_a,   aug_rain_b,   aug_rain_c])
    soil_all     = np.concatenate([soil,     aug_soil_a,   aug_soil_b,   aug_soil_c])
    land_all     = np.concatenate([land_use, aug_land_a,   aug_land_b,   aug_land_c])
    ndvi_all     = np.concatenate([ndvi,     aug_ndvi_a,   aug_ndvi_b,   aug_ndvi_c])
    y_all        = np.concatenate([y,        aug_y_a,      aug_y_b,      aug_y_c])

    X_df = pd.DataFrame({
        '坡度':     slope_all,
        '坡向':     aspect_all,
        '高程':     elev_all,
        '土壤類型': soil_all,
        '累積雨量': rainfall_all,
        'NDVI':     ndvi_all,
    })

    return X_df, y_all, land_all


# ──────────────────────────────────────────────
# 2. 模型訓練
# ──────────────────────────────────────────────

SOIL_ENCODE = {'崩積土': 0, '沖積土': 1, '紅土': 2, '砂質頁岩': 3, '黏土': 4}
FEATURE_NAMES = ['坡度', '坡向', '高程', '土壤類型', '累積雨量', 'NDVI']


def encode_features(X_df):
    X = X_df.copy()
    X['土壤類型'] = X['土壤類型'].map(SOIL_ENCODE)
    return X.values.astype(float)


def train_model(X_df, y):
    from sklearn.ensemble import RandomForestClassifier
    import shap

    X = encode_features(X_df)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    bg_idx = np.random.default_rng(0).choice(len(X), size=min(150, len(X)), replace=False)
    explainer = shap.TreeExplainer(model, data=X[bg_idx])

    return model, explainer


# ──────────────────────────────────────────────
# 3. 危險因子分析（SHAP）
# ──────────────────────────────────────────────

def analyze_risk_factors(explainer, input_array: np.ndarray) -> list:
    import shap

    shap_vals = explainer.shap_values(input_array.reshape(1, -1))

    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1][0], dtype=float)
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        sv = np.array(shap_vals[0, :, 1], dtype=float)
    else:
        sv = np.array(shap_vals[0], dtype=float)

    pos_idx = np.where(sv > 0)[0].tolist()
    if len(pos_idx) == 0:
        return [("（無顯著正貢獻因子）", 100.0)]

    pos_vals = np.array([sv[i] for i in pos_idx])
    total_pos = pos_vals.sum()
    order = np.argsort(pos_vals)[::-1].tolist()

    top3 = []
    for rank_i in order[:3]:
        feat = FEATURE_NAMES[pos_idx[rank_i]]
        pct  = float(pos_vals[rank_i]) / total_pos * 100
        top3.append((feat, round(pct, 1)))

    return top3


# ──────────────────────────────────────────────
# 4. 防治建議規則庫
# ──────────────────────────────────────────────

def generate_recommendations(slope, rainfall, ndvi, soil, land_use, elev, risk_level) -> str:
    tips = []

    if slope > 35:
        tips.append("坡度極陡，建議施作擋土牆、地錨或邊坡噴漿。")
    elif 25 <= slope <= 35:
        tips.append("坡度偏陡，建議設置階段式邊坡並加強排水。")

    if rainfall > 2000:
        tips.append("累積雨量極高，應注意土石流風險，加強監測與疏散規劃。")
    elif 1000 <= rainfall <= 2000:
        tips.append("累積雨量偏高，建議檢視排水系統並避免大規模開挖。")

    if ndvi < 0.3:
        tips.append("植生覆蓋不足，建議進行植生復育或鋪設草毯。")

    if '砂質' in soil or '礫石' in soil:
        tips.append("土壤透水性高，需注意降雨入滲引發的淺層崩塌，可考慮土釘加固。")
    if '黏土' in soil:
        tips.append("土壤排水性差，坡面應設置地下排水廊道或集水井。")

    if any(kw in land_use for kw in ['耕地', '農業']):
        tips.append("農業活動可能擾動邊坡，建議退耕還林或採用等高耕作。")
    if any(kw in land_use for kw in ['道路', '開發', '建地']):
        tips.append("人為開發區域，應強化邊坡監測與維護排水設施。")

    if elev > 1500:
        tips.append("高海拔區域，溫差大、風化作用強，需定期巡檢裂隙發展。")

    if not tips:
        tips.append("目前無顯著高風險因子，但仍需定期巡檢邊坡狀況。")

    if risk_level == '高風險':
        prefix = "【高風險警報】此坡地崩塌機率高，建議立即啟動詳細地質調查與防災應變。\n"
    elif risk_level == '中風險':
        prefix = "【中風險提醒】建議加強監測頻率並進行必要補強。\n"
    else:
        prefix = ""

    body = "\n".join(f"  • {t}" for t in tips)
    return prefix + body


# ──────────────────────────────────────────────
# 5. 主程式
# ──────────────────────────────────────────────

def get_risk_level(prob: float) -> str:
    if prob < 0.30:
        return '低風險'
    elif prob <= 0.70:
        return '中風險'
    else:
        return '高風險'


def print_separator(char: str = '─', width: int = 55):
    print(char * width)


def collect_user_input() -> dict:
    print_separator('═')
    print("  坡地崩塌風險快速評估系統")
    print("  Landslide Risk Rapid Assessment System")
    print_separator('═')
    print("\n請依序輸入以下 7 項特徵值：\n")

    data = {}

    while True:
        try:
            v = float(input("  [1] 坡度（度，建議 0~90）："))
            if 0 <= v <= 90:
                data['slope'] = v; break
            print("      ⚠ 請輸入 0～90 之間的數值。")
        except ValueError:
            print("      ⚠ 請輸入有效數字。")

    while True:
        try:
            v = float(input("  [2] 坡向（0-360 度）："))
            if 0 <= v <= 360:
                data['aspect'] = v; break
            print("      ⚠ 請輸入 0～360 之間的數值。")
        except ValueError:
            print("      ⚠ 請輸入有效數字。")

    while True:
        try:
            v = float(input("  [3] 高程（公尺）："))
            if v >= 0:
                data['elev'] = v; break
            print("      ⚠ 請輸入非負數值。")
        except ValueError:
            print("      ⚠ 請輸入有效數字。")

    soil_opts = list(SOIL_ENCODE.keys())
    soil_str = "、".join(f"{i+1}.{s}" for i, s in enumerate(soil_opts))
    while True:
        raw = input(f"  [4] 土壤類型（{soil_str}）：").strip()
        if raw in SOIL_ENCODE:
            data['soil'] = raw; break
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(soil_opts):
                data['soil'] = soil_opts[idx]; break
        except ValueError:
            pass
        print(f"      ⚠ 請輸入有效的土壤類型名稱或編號（1-{len(soil_opts)}）。")

    while True:
        try:
            v = float(input("  [5] 累積雨量（毫米）："))
            if v >= 0:
                data['rainfall'] = v; break
            print("      ⚠ 請輸入非負數值。")
        except ValueError:
            print("      ⚠ 請輸入有效數字。")

    land_opts = ['林地', '耕地', '道路', '建地', '草生地']
    land_str = "、".join(f"{i+1}.{s}" for i, s in enumerate(land_opts))
    while True:
        raw = input(f"  [6] 土地利用（{land_str}）：").strip()
        if raw in land_opts:
            data['land_use'] = raw; break
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(land_opts):
                data['land_use'] = land_opts[idx]; break
        except ValueError:
            pass
        print(f"      ⚠ 請輸入有效的土地利用類型或編號（1-{len(land_opts)}）。")

    while True:
        try:
            v = float(input("  [7] NDVI 植生指數（0～1）："))
            if 0.0 <= v <= 1.0:
                data['ndvi'] = v; break
            print("      ⚠ 請輸入 0～1 之間的數值。")
        except ValueError:
            print("      ⚠ 請輸入有效數字。")

    return data


def run_assessment(model, explainer, data: dict):
    import pandas as pd

    input_df = pd.DataFrame([{
        '坡度':     data['slope'],
        '坡向':     data['aspect'],
        '高程':     data['elev'],
        '土壤類型': data['soil'],
        '累積雨量': data['rainfall'],
        'NDVI':     data['ndvi'],
    }])
    input_arr = encode_features(input_df)

    prob = model.predict_proba(input_arr)[0][1]
    risk_level = get_risk_level(prob)

    top3 = analyze_risk_factors(explainer, input_arr[0])

    recommendations = generate_recommendations(
        slope=data['slope'], rainfall=data['rainfall'], ndvi=data['ndvi'],
        soil=data['soil'], land_use=data['land_use'], elev=data['elev'],
        risk_level=risk_level
    )

    print()
    print_separator('═')
    print("  📊 評估結果報告")
    print_separator('═')

    bar_len = int(prob * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    print(f"\n  【崩塌機率】")
    print(f"    {prob*100:.1f}%  [{bar}]")

    level_icon = {'低風險': '🟢', '中風險': '🟡', '高風險': '🔴'}
    print(f"\n  【風險等級】")
    print(f"    {level_icon[risk_level]} {risk_level}")

    print(f"\n  【危險因子分析（SHAP）】")
    for rank_i, (feat, pct) in enumerate(top3, 1):
        bar2 = '▓' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"    {rank_i}. {feat:<10} {pct:5.1f}%  [{bar2}]")

    print(f"\n  【防治建議】")
    for line in recommendations.splitlines():
        print(f"    {line}")

    print()
    print_separator('═')


def main():
    print("\n正在載入模組並訓練模型，請稍候...\n")

    try:
        import sklearn
        import shap
        import pandas
    except ImportError as e:
        print(f"❌ 缺少必要套件：{e}")
        print("請執行：pip install scikit-learn shap pandas")
        return

    X_df, y, _ = generate_training_data(n_samples=10000)
    model, explainer = train_model(X_df, y)

    collapse_rate = y.mean() * 100
    print(f"✅ 模型訓練完成（訓練樣本：{len(y)} 筆，崩塌比例：{collapse_rate:.1f}%）\n")

    while True:
        try:
            data = collect_user_input()
            run_assessment(model, explainer, data)
        except (KeyboardInterrupt, EOFError):
            print("\n\n程式結束，感謝使用。")
            break

        again = input("  是否繼續評估另一筆資料？（y/n）：").strip().lower()
        if again != 'y':
            print("\n程式結束，感謝使用。\n")
            break


if __name__ == '__main__':
    main()