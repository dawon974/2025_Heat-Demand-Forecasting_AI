# ============================================
# [전지사 공용] "온도만" 유사일(6시간 세그먼트 추세) + 지사별 XGB/RF + 안전 폴백 리포트 (NaN 완전 정제 버전)
# ============================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ---------- 설정 ----------
FILE_PATH    = "C:/Users/USER/.vscode/heat_ag/전지사(2022-2025).xlsx"
TARGET_DATES = [pd.Timestamp("2025-11-06"), pd.Timestamp("2025-11-07")]
# 공휴일(=1)인 과거 날짜는 유사일 후보에서 제외할지 여부
EXCLUDE_HOLIDAY_CANDIDATES = True

W_ANALOG = 0.7
W_ML     = 0.3

ALPHA_ZSSE    = 1.0
BETA_SEG_MEAN = 0.6
GAMMA_SEG_SLP = 0.6

SEGMENTS = [(0,6), (6,12), (12,18), (18,24)]

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# ---------- 유틸 ----------

def _finite_or(x, fallback):
    v = np.asarray(x, float)
    if not np.isfinite(v).all():
        v = np.where(np.isfinite(v), v, fallback)
    return v

def _prep(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    if ("일자" not in " ".join(df.columns) and "Date" not in " ".join(df.columns)) and len(df)>0:
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.drop(index=0).reset_index(drop=True)

    r = {}
    for c in df.columns:
        n = str(c).strip(); nl = n.lower().replace(" ","")
        if n in ["일자","Date","date"]: r[c]="Date"
        elif n in ["시간","Hour","hour","LOAD_1"]: r[c]="Hour"
        elif nl in ["열수요실적","열수요","actual","부하","부하실적"]: r[c]="Actual"
        elif nl in ["기상청실적","기상청","temperature","temp","기상청실적(온도)","실측온도","온도실적"]: r[c]="temp_real"
        elif nl in ["기상청예측","예보","forecast","temperatureforecast","temp_pred","예보온도","온도예측","fcst"]: r[c]="temp_fcst"
        elif n in ["Is_Holiday","공휴일","휴일여부"]: r[c]="Is_Holiday"
    df = df.rename(columns=r)

    if ("Date" not in df.columns) or ("Hour" not in df.columns):
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    if "Actual" in df: df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    if "temp_real" in df: df["temp_real"] = pd.to_numeric(df["temp_real"], errors="coerce")
    if "temp_fcst" in df: df["temp_fcst"] = pd.to_numeric(df["temp_fcst"], errors="coerce")
    df["Is_Holiday"] = pd.to_numeric(df.get("Is_Holiday", 0), errors="coerce").fillna(0)

    df["Hour0"] = df["Hour"].apply(lambda h: h-1 if 1<=h<=24 else h).astype(int)
    df = df.dropna(subset=["Date","Hour0"]).sort_values(["Date","Hour0"]).reset_index(drop=True)
    return df

def _ensure_24h(day_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if day_df.empty:
        return pd.DataFrame()
    d = day_df["Date"].iloc[0]
    base = pd.DataFrame({"Date":[d]*24, "Hour0": list(range(24))})
    m = base.merge(day_df, on=["Date","Hour0"], how="left")
    for c in cols:
        if c in m:
            m[c] = m[c].interpolate().bfill().ffill()
        else:
            m[c] = np.nan
    # 세그 평균 → 일평균 → 0.0
    for c in cols:
        if c in m:
            seg_means = []
            for s,e in SEGMENTS:
                seg = m.loc[(m["Hour0"]>=s)&(m["Hour0"]<e), c]
                seg_means.append(np.nanmean(seg))
            day_mean = np.nanmean(m[c])
            fill_val = 0.0 if not np.isfinite(day_mean) else day_mean
            v = m[c].to_numpy(dtype=float)
            if np.isnan(v).any():
                # 세그먼트별 채움
                for s,e,sm in zip([0,6,12,18],[6,12,18,24], seg_means):
                    seg_fill = fill_val if not np.isfinite(sm) else sm
                    idx = (m["Hour0"]>=s)&(m["Hour0"]<e)&(m[c].isna())
                    m.loc[idx, c] = seg_fill
            # 최종 잔여 NaN 0.0
            m[c] = m[c].fillna(0.0)
    m["Is_Holiday"] = m.get("Is_Holiday", 0).fillna(0)
    return m

def _recent7h_mean(df_all: pd.DataFrame, T: pd.Timestamp, col="temp_real") -> np.ndarray:
    hist = df_all[(df_all["Date"]>=T-pd.Timedelta(days=7)) & (df_all["Date"]<T)]
    if col not in df_all.columns:
        fb = 0.0
    else:
        fb = df_all[col].dropna().mean()
        if not np.isfinite(fb):
            fb = 0.0
    out=[]
    for h in range(24):
        v = hist.loc[hist["Hour0"]==h, col].dropna()
        out.append(v.mean() if len(v)>0 and np.isfinite(v.mean()) else fb)
    return np.array(out, dtype=float)

def target_temp_24(df_all: pd.DataFrame, T: pd.Timestamp) -> np.ndarray:
    fut = df_all[df_all["Date"]==T]
    if not fut.empty and "temp_fcst" in df_all.columns and fut["temp_fcst"].notna().any():
        fut = _ensure_24h(fut, ["temp_fcst"])
        arr = fut["temp_fcst"].to_numpy(dtype=float)
    elif not fut.empty and "temp_real" in df_all.columns and fut["temp_real"].notna().any():
        fut = _ensure_24h(fut, ["temp_real"])
        arr = fut["temp_real"].to_numpy(dtype=float)
    else:
        arr = _recent7h_mean(df_all, T, "temp_real")
    # 최종 finite 보장
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr

def _seg_features(temp24: np.ndarray):
    means = []
    slopes= []
    for s,e in SEGMENTS:
        y = temp24[s:e]
        x = np.arange(s,e).reshape(-1,1)
        m = float(np.nanmean(y))
        if len(y)>=2 and np.isfinite(y).all():
            reg = LinearRegression().fit(x, y)
            slp = float(reg.coef_[0])
        else:
            slp = 0.0
        means.append(m); slopes.append(slp)
    return np.array(means, float), np.array(slopes, float)

def _z(x):
    x = np.asarray(x, float)
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(m): m = 0.0
    if not np.isfinite(s) or s==0: s = 1.0
    return (x - m) / s

def _distance_by_segments(target24: np.ndarray, cand24: np.ndarray) -> float:
    z_sse = float(np.nansum((_z(target24) - _z(cand24))**2))
    tm, ts = _seg_features(target24)
    cm, cs = _seg_features(cand24)
    seg_mean_sse = float(np.nansum((tm - cm)**2))
    seg_slp_sse  = float(np.nansum((ts - cs)**2))
    return ALPHA_ZSSE*z_sse + BETA_SEG_MEAN*seg_mean_sse + GAMMA_SEG_SLP*seg_slp_sse

def rank_similar_days_by_temp(df_all: pd.DataFrame, T: pd.Timestamp):
    t24 = target_temp_24(df_all, T)
    hist_days = sorted(df_all[df_all["Date"] < T]["Date"].dt.normalize().unique())
    out = []

    for d in hist_days:
        # 온도(실측)과 공휴일 플래그 함께 보정해서 24h 구성
        day = _ensure_24h(df_all[df_all["Date"] == d], ["temp_real"])
        if day.empty:
            continue

        # ✅ 과거 날짜가 공휴일이면 후보에서 제외
        if EXCLUDE_HOLIDAY_CANDIDATES:
            # 일단위 공휴일 여부: 시각별 Is_Holiday의 평균이 0.5 이상이면 1로 간주
            day_hol = 1 if np.nanmean(day.get("Is_Holiday", 0)) >= 0.5 else 0
            if day_hol == 1:
                continue

        # 유효 온도 체크 및 보정
        if day["temp_real"].isna().any():
            continue
        c24 = day["temp_real"].to_numpy(dtype=float)
        if not np.isfinite(c24).any():
            c24 = np.where(np.isfinite(c24), c24, 0.0)

        score = _distance_by_segments(t24, c24)
        out.append((score, d, c24))

    out.sort(key=lambda x: x[0])
    return t24, out

def choose_analog_with_actual(df_all: pd.DataFrame, ranked: list):
    for score, d, _ in ranked:
        day = _ensure_24h(df_all[df_all["Date"]==d], ["Actual"])
        if not day.empty and day["Actual"].notna().all():
            a = day["Actual"].to_numpy(dtype=float)
            if not np.isfinite(a).all():
                a = np.where(np.isfinite(a), a, np.nan)
            if np.isfinite(a).all():
                return d, a, score
    return None, None, None

def _cal_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofyear"] = out["Date"].dt.dayofyear/365.0
    out["weekday"]   = out["Date"].dt.weekday/6.0
    out["month"]     = out["Date"].dt.month/12.0
    out["hour_norm"] = out["Hour0"]/24.0
    yrs = out["Date"].dt.year
    out["year_norm"] = (yrs-yrs.min())/(yrs.max()-yrs.min() if yrs.max()!=yrs.min() else 1.0)
    return out

def _clean_train_matrix(hf: pd.DataFrame, X_cols, y_col="Actual"):
    X = hf[X_cols].copy()
    y = hf[y_col].copy()
    # NaN 채움: temperature 우선 보강
    if "temperature" in X.columns:
        # temperature가 비었으면 day별 temp_real 평균으로 대치 시도
        miss = ~np.isfinite(X["temperature"])
        if miss.any():
            # (가능하면) 원본 temp_real을 사용해 보강
            if "temp_real" in hf.columns:
                tr = hf["temp_real"]
                tr = np.where(np.isfinite(tr), tr, np.nan)
                X.loc[miss, "temperature"] = np.nanmean(tr)
        # 그래도 남으면 0.0
        X["temperature"] = X["temperature"].fillna(0.0)
    # 나머지 피처도 비정상값 0.0 처리
    for c in X.columns:
        X[c] = X[c].astype(float)
        X[c] = np.where(np.isfinite(X[c]), X[c], 0.0)
    # y도 finite만 사용
    mask = np.isfinite(y.to_numpy(dtype=float))
    X = X.loc[mask]
    y = y.loc[mask]
    # 최종 NaN 제거
    valid = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y

# ---------- 실행 ----------
xls = pd.ExcelFile(FILE_PATH)
all_rows = []
ml_only_records = []
rf_disabled_log = []   # 어떤 지사에서 RF를 비활성(대체)했는지 기록

X_COLS = ["dayofyear","weekday","month","hour_norm","temperature","Is_Holiday","year_norm"]

for sheet in xls.sheet_names:
    df_sheet = _prep(xls.parse(sheet))
    if df_sheet.empty:
        print(f"[스킵] {sheet}: 필수 컬럼 부족/형식 불일치")
        continue

    first_T = min(TARGET_DATES)

    # 지사별 모델 초기화
    xgb_model = None
    rf_model  = None

    # 학습 데이터 준비
    if "Actual" in df_sheet.columns:
        hist_for_ml = df_sheet[(df_sheet["Date"] < first_T) & df_sheet["Actual"].notna()].copy()
    else:
        hist_for_ml = pd.DataFrame()

    if hist_for_ml.empty:
        print(f"[{sheet}] 실적 부족 → ML 비활성 (유사일 Analog만 or ML 100% 폴백)")
    else:
        hist_for_ml["temperature"] = hist_for_ml.get("temp_real", np.nan)
        hf = _cal_feats(hist_for_ml)
        X_train, y_train = _clean_train_matrix(hf.assign(temp_real=hist_for_ml.get("temp_real", np.nan)), X_COLS, "Actual")

        if len(X_train) < 24:  # 최소 샘플 수 엄격화
            print(f"[{sheet}] ML 학습 샘플 부족({len(X_train)}) → ML 비활성")
        else:
            # XGB
            xgb_model = xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.07, max_depth=5, min_child_weight=5,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.2,
                tree_method="hist", random_state=42, n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)

            # RF (NaN 엄격 방지 위해 clean한 X_train 사용)
            try:
                rf_model = RandomForestRegressor(
                    n_estimators=500, max_depth=None, min_samples_leaf=3, max_features="sqrt",
                    bootstrap=True, random_state=42, n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
            except Exception as e:
                print(f"[{sheet}] RF 학습 실패 → XGB만 사용. 사유: {e}")
                rf_model = None
                rf_disabled_log.append({"지사":sheet, "사유":"RF 학습 실패(입력 품질/샘플수)"})

    print(f"\n=== [{sheet}] 온도 기반 유사일 선택 & 예측 ===")
    for T in TARGET_DATES:
        t24, ranked = rank_similar_days_by_temp(df_sheet, T)
        if not ranked:
            print(f"  {T.date()} : 유사일 후보 없음(온도시계열 부족)")
            fut = pd.DataFrame({"Date":[T]*24, "Hour0": list(range(24))})
            fut["temperature"] = target_temp_24(df_sheet, T)
            fut["Is_Holiday"]  = 1 if T.weekday()>=5 else 0
            Xf = _cal_feats(fut)[X_COLS].copy()
            for c in X_COLS:
                Xf[c] = np.where(np.isfinite(Xf[c]), Xf[c], 0.0)

            if xgb_model is not None:
                pred_xgb = xgb_model.predict(Xf)
            else:
                pred_xgb = np.full(24, np.nan)

            if rf_model is not None:
                try:
                    pred_rf  = rf_model.predict(Xf)
                except Exception as e:
                    print(f"  [주의][{sheet} {T.date()}] RF 예측 실패 → XGB로 대체. 사유: {e}")
                    pred_rf  = pred_xgb.copy()
                    rf_disabled_log.append({"지사":sheet, "사유":"RF 예측 실패"})
            else:
                pred_rf = pred_xgb.copy()

            final_xgb, final_rf = pred_xgb, pred_rf
            ml_only_records.append({"지사":sheet,"날짜":T.date(),"사유":"유사일 없음"})
            for h, fx, fr in zip(range(1,25), final_xgb, final_rf):
                all_rows.append({
                    "Branch": sheet, "Date": T.date(), "Hour": h,
                    "TempOnly_BestDate": None, "TempOnly_Score": np.nan,
                    "Analog": np.nan,
                    "Pred_XGB": float(fx) if np.isfinite(fx) else np.nan,
                    "Pred_RF":  float(fr) if np.isfinite(fr) else np.nan,
                    "Final_XGB": float(fx) if np.isfinite(fx) else 0.0,
                    "Final_RF":  float(fr) if np.isfinite(fr) else 0.0,
                })
            continue

        best_day, analog, best_score = choose_analog_with_actual(df_sheet, ranked)

        fut = pd.DataFrame({"Date":[T]*24, "Hour0": list(range(24))})
        fut["temperature"] = t24
        fut["Is_Holiday"]  = 1 if T.weekday()>=5 else 0
        Xf = _cal_feats(fut)[X_COLS].copy()
        for c in X_COLS:
            Xf[c] = np.where(np.isfinite(Xf[c]), Xf[c], 0.0)

        if xgb_model is not None:
            pred_xgb = xgb_model.predict(Xf)
        else:
            pred_xgb = np.full(24, np.nan)

        if rf_model is not None:
            try:
                pred_rf  = rf_model.predict(Xf)
            except Exception as e:
                print(f"  [주의][{sheet} {T.date()}] RF 예측 실패 → XGB로 대체. 사유: {e}")
                pred_rf  = pred_xgb.copy()
                rf_disabled_log.append({"지사":sheet, "사유":"RF 예측 실패"})
        else:
            pred_rf = pred_xgb.copy()

        hour_out = np.arange(1,25)

        if best_day is not None and analog is not None:
            if np.isfinite(pred_xgb).all():
                final_xgb = W_ANALOG*analog + W_ML*pred_xgb
            else:
                final_xgb = analog.copy()

            if np.isfinite(pred_rf).all():
                final_rf  = W_ANALOG*analog + W_ML*pred_rf
            else:
                # RF가 안되면 XGB로 밀어줌(표 형태 유지)
                final_rf  = W_ANALOG*analog + W_ML*(pred_xgb if np.isfinite(pred_xgb).all() else np.zeros(24))

            print(f"  - {T.date()}  Selected(실적有)={best_day.date()}  score={best_score:.4f}")
            for h, a, px, pr, fx, fr in zip(hour_out, analog, pred_xgb, pred_rf, final_xgb, final_rf):
                all_rows.append({
                    "Branch": sheet, "Date": T.date(), "Hour": int(h),
                    "TempOnly_BestDate": best_day.date(), "TempOnly_Score": float(best_score),
                    "Analog": float(a),
                    "Pred_XGB": float(px) if np.isfinite(px) else np.nan,
                    "Pred_RF":  float(pr) if np.isfinite(pr) else np.nan,
                    "Final_XGB": float(fx), "Final_RF": float(fr),
                })
        else:
            chosen_date, chosen_score = ranked[0][1], ranked[0][0]
            # 유사일 실적이 없으므로 ML 100%
            fx = pred_xgb if np.isfinite(pred_xgb).all() else np.zeros(24)
            fr = pred_rf  if np.isfinite(pred_rf).all()  else fx
            print(f"  - {T.date()}  Selected(실적無)={chosen_date.date()}  score={chosen_score:.4f}  → ML 100%")
            ml_only_records.append({"지사":sheet,"날짜":T.date(),"사유":"유사일 실적 없음"})
            for h, px, pr, _fx, _fr in zip(hour_out, pred_xgb, pred_rf, fx, fr):
                all_rows.append({
                    "Branch": sheet, "Date": T.date(), "Hour": int(h),
                    "TempOnly_BestDate": chosen_date.date(), "TempOnly_Score": float(chosen_score),
                    "Analog": np.nan,
                    "Pred_XGB": float(px) if np.isfinite(px) else np.nan,
                    "Pred_RF":  float(pr) if np.isfinite(pr) else np.nan,
                    "Final_XGB": float(_fx), "Final_RF": float(_fr),
                })

# ---------- 출력 ----------
if all_rows:
    df_all = pd.DataFrame(all_rows).sort_values(["Branch","Date","Hour"]).reset_index(drop=True)
    print("\n✅ [전지사] 온도 기반(실적 불필요) 유사일 + 지사별 ML 보조 예측 완료")

    sel_tbl = (
        df_all.groupby(["Branch","Date"])["TempOnly_BestDate"]
              .first()
              .unstack("Date")
    )
    print("\n🗓️ [전지사] 타깃일별 유사일(TempOnly_BestDate)")
    try:
        from IPython.display import display
        display(sel_tbl)
    except Exception:
        print(sel_tbl)

    xgb_wide = (
        df_all.pivot_table(index=["Branch","Date"], columns="Hour", values="Final_XGB", aggfunc="first")
              .round(2).reset_index().sort_values(["Branch","Date"])
    )
    xgb_wide["날짜"] = pd.to_datetime(xgb_wide["Date"]).dt.strftime("%Y%m%d")
    xgb_wide = xgb_wide.rename(columns={"Branch":"지사"}).drop(columns=["Date"])
    hour_cols = [c for c in range(1,25) if c in xgb_wide.columns]
    xgb_wide = xgb_wide[["지사","날짜", *hour_cols]]

    print("\n📊 시간별 예측 (Final_XGB) — 행=지사·날짜, 열=1~24시")
    try:
        display(xgb_wide)
    except Exception:
        print(xgb_wide.head())

    rf_wide = (
        df_all.pivot_table(index=["Branch","Date"], columns="Hour", values="Final_RF", aggfunc="first")
              .round(2).reset_index().sort_values(["Branch","Date"])
    )
    rf_wide["날짜"] = pd.to_datetime(rf_wide["Date"]).dt.strftime("%Y%m%d")
    rf_wide = rf_wide.rename(columns={"Branch":"지사"}).drop(columns=["Date"])
    hour_cols_rf = [c for c in range(1,25) if c in rf_wide.columns]
    rf_wide = rf_wide[["지사","날짜", *hour_cols_rf]]

    print("\n📊 시간별 예측 (Final_RF) — 행=지사·날짜, 열=1~24시")
    try:
        display(rf_wide)
    except Exception:
        print(rf_wide.head())

    if ml_only_records:
        ml_df = pd.DataFrame(ml_only_records).sort_values(["지사","날짜"]).reset_index(drop=True)
        print("\n⚠️ ML 100% 폴백 발생 목록 (지사·날짜·사유)")
        try:
            display(ml_df)
        except Exception:
            print(ml_df)

    if rf_disabled_log:
        rf_df = pd.DataFrame(rf_disabled_log).drop_duplicates().reset_index(drop=True)
        print("\nℹ️ RF 비활성/대체 로그")
        try:
            display(rf_df)
        except Exception:
            print(rf_df)
else:
    print("\n[알림] 결과 없음 (온도/시계열/실적 부족 가능)")