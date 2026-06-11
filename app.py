"""
坡地崩塌風險快速評估系統 - Streamlit 網頁前端
(完全不動原程式，直接載入原程式的核心演算法與規則庫)
"""
import streamlit as st
import numpy as np
import pandas as pd
import warnings

# 💡 直接從你的原程式碼檔案 (model_core) 匯入寫好的函式與設定
from model_core import (
    generate_training_data, 
    train_model, 
    encode_features, 
    analyze_risk_factors, 
    generate_recommendations,
    get_risk_level,
    SOIL_ENCODE
)

warnings.filterwarnings('ignore')

# 1. 網頁基本配置
st.set_page_config(
    page_title="坡地崩塌風險快速評估系統",
    page_icon="⛰️",
    layout="wide"
)

# 2. 利用 Streamlit 快取機制載入大腦，避免每次網頁重整都要重新訓練模型
@st.cache_resource
def init_ai_brain():
    # 完全調用你原程式的邏輯
    X_df, y, land_use_all = generate_training_data(n_samples=10000)
    model, explainer = train_model(X_df, y)
    return model, explainer, y.mean() * 100, len(y)

# 顯示載入動畫
with st.spinner("🌲 AI 坡地大數據載入與隨機森林模型訓練中，請稍候..."):
    model, explainer, collapse_rate, total_samples = init_ai_brain()

# 3. 網頁頁面標題
st.title("⛰️ 坡地崩塌風險快速評估系統 (網頁互動版)")
st.caption(f"後端引擎：隨機森林分類器 ｜ 訓練樣本總數：{total_samples} 筆 ｜ 全域基礎崩塌率：{collapse_rate:.1f}%")
st.markdown("---")

# 4. 網頁畫面佈局 (左右分欄：左邊輸入、右邊報告)
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📋 現地環境特徵參數輸入")
    
    # 將你原本的文字 input，全部升級為網頁互動元件
    slope = st.slider("1. 坡度 (度 °)", min_value=0.0, max_value=90.0, value=25.0, step=0.5, help="台灣高風險崩塌多發生在 25°~60° 之間")
    aspect = st.slider("2. 坡向 (0-360°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0, help="0:北, 90:東, 180:南, 270:西")
    elev = st.number_input("3. 高程 (公尺 m)", min_value=0.0, max_value=4000.0, value=500.0, step=50.0)
    
    soil_opts = list(SOIL_ENCODE.keys())
    soil = st.selectbox("4. 土壤類型", soil_opts, index=0)
    
    rainfall = st.number_input("5. 累積雨量 (毫米 mm)", min_value=0.0, max_value=6000.0, value=1200.0, step=100.0)
    
    land_opts = ['林地', '耕地', '道路', '建地', '草生地']
    land_use = st.selectbox("6. 土地利用", land_opts, index=0)
    
    ndvi = st.slider("7. NDVI 植生指數", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="數值越接近 1 代表地表植生越茂密、根系抓地力越好")

with col2:
    st.subheader("📊 AI 智能即時評估報告")
    
    # 整理網頁上的輸入資料，轉換成 Dataframe
    web_data = {
        'slope': slope, 'aspect': aspect, 'elev': elev, 
        'soil': soil, 'rainfall': rainfall, 'land_use': land_use, 'ndvi': ndvi
    }
    
    # 調用原程式的特徵編碼
    input_df = pd.DataFrame([{
        '坡度': web_data['slope'], '坡向': web_data['aspect'], '高程': web_data['elev'],
        '土壤類型': web_data['soil'], '累積雨量': web_data['rainfall'], 'NDVI': web_data['ndvi']
    }])
    input_arr = encode_features(input_df)
    
    # 預測崩塌機率
    prob = model.predict_proba(input_arr)[0][1]
    risk_level = get_risk_level(prob)
    
    # 顯示風險警告盒
    if risk_level == '低風險':
        st.success(f"### 🟢 風險等級：低風險 (崩塌機率: {prob*100:.1f}%)")
    elif risk_level == '中風險':
        st.warning(f"### 🟡 風險等級：中風險 (崩塌機率: {prob*100:.1f}%)")
    else:
        st.error(f"### 🔴 風險等級：高風險 (崩塌機率: {prob*100:.1f}%)")
        
    # 網頁進度條顯示機率
    st.progress(prob)
    
    # 5. SHAP 因子拆解 (完全調用原程式函式)
    st.markdown("#### 🔍 核心致災因子貢獻比 (SHAP)")
    top3 = analyze_risk_factors(explainer, input_arr[0])
    
    if "（無顯著正貢獻因子）" in top3[0][0]:
        st.info("💡 當前環境各項參數極為穩定，無顯著正向致災貢獻因子。")
    else:
        for rank_i, (feat, pct) in enumerate(top3, 1):
            st.write(f"**第 {rank_i} 主因：{feat}** (致災貢獻比例: {pct:.1f}%)")
            st.progress(pct / 100)

    # 6. 防治建議 (完全調用原程式規則庫)
    st.markdown("#### 🛠️ 客製化水土保持防治建議")
    recommendations = generate_recommendations(
        slope=web_data['slope'], rainfall=web_data['rainfall'], ndvi=web_data['ndvi'],
        soil=web_data['soil'], land_use=web_data['land_use'], elev=web_data['elev'],
        risk_level=risk_level
    )
    
    # 將回傳的文字漂亮地排版在網頁上
    for line in recommendations.splitlines():
        if line.strip():
            st.markdown(line.replace("•", "*"))

st.markdown("---")
st.caption("⚠️ 免責聲明：本網頁為機器學習演算法與水保統計模擬學術工具，現地之實際工程規劃與評估，仍須以專業技師之現地勘查與實地鑽探報告書為準。")