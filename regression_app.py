import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }
h1,h2,h3 { font-family: 'Space Mono', monospace; color: #e6edf3; }

[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
.stDataFrame td, .stDataFrame th { color: #e6edf3 !important; background: #161b22 !important; }
.stSelectbox div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
.stMultiSelect div[data-baseweb="select"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="option"] { background: #161b22 !important; color: #e6edf3 !important; }
div[data-baseweb="popover"] { background: #161b22 !important; }
.stTextInput input, .stTextArea textarea { color: #e6edf3 !important; background: #161b22 !important; }
div[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

.hero-title {
    font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#58a6ff,#bc8cff,#f778ba);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2;
}
.hero-sub { color:#8b949e; font-size:1rem; margin-bottom:1.5rem; }

.section-hdr {
    font-family:'Space Mono',monospace; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:2px; color:#58a6ff; margin-bottom:0.8rem;
    border-bottom:1px solid #21262d; padding-bottom:0.5rem;
}
.card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.card-accent { border-left:4px solid #58a6ff; }

.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem;
         font-weight:600; font-family:'Space Mono',monospace; margin-right:4px; }
.badge-blue   { background:#1f3a5f; color:#58a6ff; }
.badge-green  { background:#1a3a2a; color:#3fb950; }
.badge-yellow { background:#3a2d10; color:#d29922; }
.badge-red    { background:#3d1f1f; color:#f85149; }
.badge-purple { background:#2d1f5f; color:#bc8cff; }

.warn-box { background:#2a1f0a; border:1px solid #d29922; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#d29922; font-size:0.88rem; }
.ok-box   { background:#0a2a14; border:1px solid #3fb950; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#3fb950; font-size:0.88rem; }
.err-box  { background:#2a0a0a; border:1px solid #f85149; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#f85149; font-size:0.88rem; }
.info-box { background:#0a1a2a; border:1px solid #58a6ff; border-radius:8px; padding:0.9rem 1.1rem; margin:0.5rem 0; color:#a8d8ff; font-size:0.88rem; }

.interpret-box {
    background:#1c2333; border:1px solid #58a6ff; border-radius:10px;
    padding:1.1rem 1.3rem; margin:0.8rem 0; color:#e6edf3;
    font-size:0.88rem; line-height:1.8;
}

.stButton>button {
    background:linear-gradient(90deg,#1f3a5f,#2d4a7a); color:#58a6ff; border:1px solid #58a6ff;
    border-radius:8px; font-family:'Space Mono',monospace; font-weight:700;
    padding:0.5rem 1.2rem; transition:all 0.2s;
}
.stButton>button:hover { background:linear-gradient(90deg,#2d4a7a,#3a5a9a); opacity:0.9; }

.run-btn>button {
    background:linear-gradient(90deg,#238636,#2ea043) !important; color:white !important;
    border:none !important;
}

div[data-testid="stExpander"] { background:#161b22; border:1px solid #30363d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────
DEFAULTS = {
    "df": None,
    "filename": "",
    "header_row": 0,
    "run_history": [],      # list of result dicts per run
    "run_counter": 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper utilities ───────────────────────────────────────────────────────

def detect_header_row(raw_df, max_scan=10):
    """
    Scan the first `max_scan` rows to find the true header row.
    The header row is the first row where:
      - Most cells are non-empty strings (not numbers, not NaN)
      - The row below contains predominantly numeric values
    Returns the header row index (0-based).
    """
    n_cols = raw_df.shape[1]
    scan = min(max_scan, raw_df.shape[0] - 1)

    for i in range(scan):
        row = raw_df.iloc[i]
        # Count cells that look like text labels (not numeric, not null)
        text_count = sum(
            1 for v in row
            if v is not None
            and not (isinstance(v, float) and np.isnan(v))
            and not isinstance(v, (int, float, np.integer, np.floating))
        )
        text_ratio = text_count / n_cols if n_cols > 0 else 0

        # Check that the NEXT row is mostly numeric
        if i + 1 < raw_df.shape[0]:
            next_row = raw_df.iloc[i + 1]
            num_count = sum(1 for v in next_row if isinstance(v, (int, float, np.integer, np.floating))
                            and not (isinstance(v, float) and np.isnan(v)))
            num_ratio = num_count / n_cols if n_cols > 0 else 0
        else:
            num_ratio = 0

        if text_ratio >= 0.5 and num_ratio >= 0.4:
            return i

    return 0   # fallback: row 0 is the header


def load_data(uploaded_file):
    """Load CSV or Excel, auto-detecting the true header row."""
    name = uploaded_file.name.lower()
    header_row = 0
    detected_msg = None

    if name.endswith(".csv"):
        # Read raw first to detect header
        uploaded_file.seek(0)
        try:
            raw = pd.read_csv(uploaded_file, header=None, nrows=15)
        except Exception:
            uploaded_file.seek(0)
            raw = pd.read_csv(uploaded_file, header=None, nrows=15, encoding='latin1')
        header_row = detect_header_row(raw)
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, header=header_row)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=header_row, encoding='latin1')

    elif name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        xf = pd.ExcelFile(uploaded_file)
        sheet_names = xf.sheet_names
        if len(sheet_names) > 1:
            chosen = st.sidebar.selectbox("Sheet name", sheet_names)
        else:
            chosen = sheet_names[0]
        # Read raw (no header) to detect
        uploaded_file.seek(0)
        raw = pd.read_excel(uploaded_file, sheet_name=chosen, header=None, nrows=15)
        header_row = detect_header_row(raw)
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=chosen, header=header_row)
    else:
        st.error("Unsupported file type.")
        return None, 0

    # Drop rows that are entirely NaN (often trailing empty rows in Excel)
    df = df.dropna(how='all').reset_index(drop=True)

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Coerce columns that look numeric but were read as object
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted

    return df, header_row


def suggest_variables(df):
    """
    Heuristically suggest which column is the dependent variable (Y)
    and which are independent variables (X).

    Rules:
    - Y candidates: numeric columns whose name contains output-like keywords
      (price, cost, revenue, sales, profit, expense, score, output, result,
       doanh, chi, gia, ket, thu, buchanan, votes_for)
      OR the numeric column with the highest variance relative to mean (CV).
      Tie-break: last numeric column (common in textbook datasets).
    - X candidates: all other numeric columns.
    - Text/ID columns (high cardinality strings, index-like) are flagged separately.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None, [], [], "Không tìm thấy cột số nào."

    # Keywords that suggest an outcome/dependent variable
    Y_KEYWORDS = [
        "price", "cost", "revenue", "sales", "profit", "expense", "score",
        "output", "result", "income", "wage", "salary", "return", "loss",
        "target", "label", "y", "dependent", "outcome", "response",
        # Vietnamese
        "gia", "chi", "phi", "thu", "doanh", "ket", "qua", "luong",
        # Case-specific
        "buchanan", "vote", "expense", "carats",
    ]

    best_y = None
    best_score = -1
    reasons = {}

    for col in numeric_cols:
        col_lower = col.lower()
        score = 0
        reason_parts = []

        # Keyword match
        kw_matches = [kw for kw in Y_KEYWORDS if kw in col_lower]
        if kw_matches:
            score += 3
            reason_parts.append(f"tên gợi ý biến kết quả ({', '.join(kw_matches)})")

        # High CV (coefficient of variation) → likely outcome with wide range
        col_data = df[col].dropna()
        if col_data.mean() != 0:
            cv = col_data.std() / abs(col_data.mean())
            if cv > 1.5:
                score += 1
                reason_parts.append(f"hệ số biến thiên cao (CV={cv:.2f})")

        # Last column heuristic (common in structured datasets)
        if col == numeric_cols[-1]:
            score += 0.5
            reason_parts.append("cột số cuối cùng")

        reasons[col] = reason_parts if reason_parts else ["không có dấu hiệu rõ ràng"]
        if score > best_score:
            best_score = score
            best_y = col

    # If no keyword matched at all, default to last numeric col
    if best_y is None:
        best_y = numeric_cols[-1]

    best_x = [c for c in numeric_cols if c != best_y]

    # Build explanation
    y_reason = "; ".join(reasons.get(best_y, []))
    explanation = (
        f"🎯 **Gợi ý Y = `{best_y}`** — {y_reason}.\n\n"
        f"📌 **Gợi ý X = {', '.join([f'`{c}`' for c in best_x])}** — các biến số còn lại.\n\n"
        "Bạn có thể thay đổi lựa chọn bên dưới."
    )

    return best_y, best_x, numeric_cols, explanation


def data_quality_report(df):
    issues = []
    numeric_df = df.select_dtypes(include=[np.number])

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        miss_cols = missing[missing > 0]
        pct = (miss_cols / len(df) * 100).round(1)
        issues.append({
            "level": "warn",
            "msg": f"Thiếu dữ liệu: {total_missing} giá trị null trong {len(miss_cols)} cột — " +
                   ", ".join([f"{c} ({p}%)" for c, p in pct.items()])
        })

    # Duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append({"level": "warn", "msg": f"Hàng trùng lặp: {dupes} hàng bị lặp."})

    # Low-variance columns
    for col in numeric_df.columns:
        if numeric_df[col].std() == 0:
            issues.append({"level": "err", "msg": f"Cột '{col}' có phương sai = 0 (hằng số), không dùng được trong hồi quy."})

    # Outliers via IQR
    outlier_cols = []
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
        if n_out > 0:
            outlier_cols.append(f"{col} ({n_out} điểm)")
    if outlier_cols:
        issues.append({"level": "warn", "msg": "Outlier (IQR): " + ", ".join(outlier_cols)})

    # Non-numeric columns
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        issues.append({"level": "info", "msg": f"Cột phi số (cần encode hoặc loại bỏ trước khi hồi quy): {', '.join(non_num)}"})

    if not issues:
        issues.append({"level": "ok", "msg": "Dữ liệu sạch — không phát hiện vấn đề nghiêm trọng."})

    return issues


def get_ols_stats(model):
    reg_stats = pd.DataFrame({
        'Chỉ số': ['Multiple R', 'R²', 'Adjusted R²', 'Std Error', 'Observations'],
        'Giá trị': [
            float(np.sqrt(model.rsquared)),
            float(model.rsquared),
            float(model.rsquared_adj),
            float(np.sqrt(model.mse_resid)),
            int(model.nobs)
        ]
    })
    anova = pd.DataFrame({
        'ANOVA': ['Regression', 'Residual', 'Total'],
        'df': [int(model.df_model), int(model.df_resid), int(model.df_model + model.df_resid)],
        'SS': [model.ess, model.ssr, model.centered_tss],
        'MS': [model.ess / model.df_model, model.ssr / model.df_resid, np.nan],
        'F': [model.fvalue, np.nan, np.nan],
        'Significance F': [model.f_pvalue, np.nan, np.nan]
    })
    coef_df = pd.DataFrame({
        'Biến': model.params.index,
        'Hệ số': model.params.values,
        'Std Error': model.bse.values,
        't Stat': model.tvalues.values,
        'P-value': model.pvalues.values,
        'Lower 95%': model.conf_int()[0].values,
        'Upper 95%': model.conf_int()[1].values
    })
    return reg_stats, anova, coef_df


def build_X(df, ind_vars, model_type, x_col_for_quad=None):
    """Build design matrix based on model type."""
    if model_type == "Tuyến tính (Linear)":
        X = sm.add_constant(df[ind_vars])
    elif model_type == "Bậc hai (Quadratic)":
        X_df = df[ind_vars].copy()
        for col in ind_vars:
            X_df[col + "²"] = df[col] ** 2
        X = sm.add_constant(X_df)
    elif model_type == "Bậc hai Centered":
        x_col = ind_vars[0] if len(ind_vars) == 1 else (x_col_for_quad or ind_vars[0])
        rest = [c for c in ind_vars if c != x_col]
        x_mean = df[x_col].mean()
        X_df = pd.DataFrame({'Xc': df[x_col] - x_mean, 'Xc2': (df[x_col] - x_mean) ** 2})
        for c in rest:
            X_df[c] = df[c]
        X = sm.add_constant(X_df)
    elif model_type == "Logarithmic (Log Y)":
        y_check = None
        X = sm.add_constant(df[ind_vars])
        # y will be log-transformed outside
    elif model_type == "Tương tác (Interaction)":
        X_df = df[ind_vars].copy()
        cols = list(ind_vars)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                X_df[f"{cols[i]}×{cols[j]}"] = df[cols[i]] * df[cols[j]]
        X = sm.add_constant(X_df)
    else:
        X = sm.add_constant(df[ind_vars])
    return X


def run_regression(df, dep_var, ind_vars, model_type):
    df_clean = df[[dep_var] + ind_vars].dropna()
    y_raw = df_clean[dep_var]

    if model_type == "Logarithmic (Log Y)":
        if (y_raw <= 0).any():
            raise ValueError("Log Y yêu cầu tất cả giá trị Y > 0.")
        y = np.log(y_raw)
    else:
        y = y_raw

    X = build_X(df_clean, ind_vars, model_type)
    model = sm.OLS(y, X).fit()
    return model, df_clean, y


def make_plots(model, df_clean, dep_var, ind_vars, model_type):
    """Generate trend + residual + Q-Q plots, return as bytes."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    DARK = '#0d1117'
    PANEL = '#161b22'
    GRID = '#21262d'
    RED = '#f85149'
    BLUE = '#58a6ff'
    GRAY = '#8b949e'
    GREEN = '#3fb950'

    # ── Plot 1: Trend / Scatter with fitted line (first independent var) ──────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    x_col = ind_vars[0]
    x_vals = df_clean[x_col]

    if model_type == "Logarithmic (Log Y)":
        y_fitted = np.exp(model.fittedvalues)
        y_actual = df_clean[dep_var]
    else:
        y_fitted = model.fittedvalues
        y_actual = df_clean[dep_var]

    ax1.scatter(x_vals, y_actual, color=GRAY, alpha=0.6, s=30, label='Thực tế')

    # Smooth prediction line over x range
    x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
    df_range = df_clean.copy()
    pred_rows = []
    for xv in x_range:
        row = {c: df_clean[c].mean() for c in ind_vars}
        row[x_col] = xv
        pred_rows.append(row)
    df_pred_range = pd.DataFrame(pred_rows)
    X_range = build_X(df_pred_range, ind_vars, model_type)

    try:
        preds = model.get_prediction(X_range)
        pf = preds.summary_frame(alpha=0.05)
        y_line = pf['mean'] if model_type != "Logarithmic (Log Y)" else np.exp(pf['mean'])
        y_lo   = pf['mean_ci_lower'] if model_type != "Logarithmic (Log Y)" else np.exp(pf['mean_ci_lower'])
        y_hi   = pf['mean_ci_upper'] if model_type != "Logarithmic (Log Y)" else np.exp(pf['mean_ci_upper'])
        ax1.plot(x_range, y_line, color=RED, lw=2, label='Đường hồi quy')
        ax1.fill_between(x_range, y_lo, y_hi, color=RED, alpha=0.15, label='95% CI')
    except Exception:
        ax1.plot(x_vals, y_fitted, color=RED, lw=2, label='Fitted')

    ax1.set_title(f'Trend: {dep_var} ~ {x_col}', color='#e6edf3', fontsize=10)
    ax1.set_xlabel(x_col, color=GRAY, fontsize=9)
    ax1.set_ylabel(dep_var, color=GRAY, fontsize=9)
    ax1.tick_params(colors=GRAY)
    ax1.grid(True, alpha=0.2, color=GRID)
    ax1.spines[:].set_color(GRID)
    ax1.legend(fontsize=8, labelcolor='#e6edf3', facecolor=PANEL)

    # ── Plot 2: Residual vs Fitted ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    ax2.scatter(model.fittedvalues, model.resid, color=BLUE, alpha=0.5, s=30)
    ax2.axhline(0, color=RED, linestyle='--', lw=1.5)
    ax2.set_title('Residual Plot (Fitted vs Residuals)', color='#e6edf3', fontsize=10)
    ax2.set_xlabel('Giá trị dự báo (Fitted)', color=GRAY, fontsize=9)
    ax2.set_ylabel('Phần dư (Residuals)', color=GRAY, fontsize=9)
    ax2.tick_params(colors=GRAY)
    ax2.grid(True, alpha=0.2, color=GRID)
    ax2.spines[:].set_color(GRID)

    # ── Plot 3: Q-Q Plot ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    resid_std = (model.resid - model.resid.mean()) / model.resid.std()
    (osm, osr), (slope, intercept, r) = stats.probplot(model.resid, dist="norm")
    ax3.scatter(osm, osr, color=BLUE, alpha=0.6, s=25, label='Phần dư')
    x_line = np.array([min(osm), max(osm)])
    ax3.plot(x_line, slope * x_line + intercept, color=RED, lw=2, label='Đường chuẩn')
    ax3.set_title('Q-Q Plot (Kiểm tra chuẩn hóa phần dư)', color='#e6edf3', fontsize=10)
    ax3.set_xlabel('Quantile lý thuyết', color=GRAY, fontsize=9)
    ax3.set_ylabel('Quantile thực tế', color=GRAY, fontsize=9)
    ax3.tick_params(colors=GRAY)
    ax3.grid(True, alpha=0.2, color=GRID)
    ax3.spines[:].set_color(GRID)
    ax3.legend(fontsize=8, labelcolor='#e6edf3', facecolor=PANEL)

    # ── Plot 4: Actual vs Predicted ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    y_pred_plot = np.exp(model.fittedvalues) if model_type == "Logarithmic (Log Y)" else model.fittedvalues
    y_act_plot  = df_clean[dep_var]
    ax4.scatter(y_act_plot, y_pred_plot, color=GREEN, alpha=0.5, s=30)
    mn = min(y_act_plot.min(), y_pred_plot.min())
    mx = max(y_act_plot.max(), y_pred_plot.max())
    ax4.plot([mn, mx], [mn, mx], color=RED, lw=1.5, linestyle='--', label='Perfect fit')
    ax4.set_title('Actual vs Predicted', color='#e6edf3', fontsize=10)
    ax4.set_xlabel('Giá trị thực tế', color=GRAY, fontsize=9)
    ax4.set_ylabel('Giá trị dự báo', color=GRAY, fontsize=9)
    ax4.tick_params(colors=GRAY)
    ax4.grid(True, alpha=0.2, color=GRID)
    ax4.spines[:].set_color(GRID)
    ax4.legend(fontsize=8, labelcolor='#e6edf3', facecolor=PANEL)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close()
    buf.seek(0)
    return buf


def interpret_model(model, model_type, dep_var, ind_vars):
    """Generate Vietnamese interpretation text."""
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    f_p = model.f_pvalue
    n = int(model.nobs)
    k = int(model.df_model)
    rmse = float(np.sqrt(model.mse_resid))

    lines = []

    # Overall model fit
    lines.append("📊 ĐÁNH GIÁ TỔNG THỂ MÔ HÌNH")
    if r2 >= 0.9:
        fit_quality = "rất tốt"
    elif r2 >= 0.7:
        fit_quality = "tốt"
    elif r2 >= 0.5:
        fit_quality = "trung bình"
    else:
        fit_quality = "yếu"
    lines.append(f"• R² = {r2:.4f} → Mô hình giải thích được {r2*100:.1f}% biến động của {dep_var}. Độ phù hợp {fit_quality}.")
    lines.append(f"• Adjusted R² = {adj_r2:.4f} (sau khi điều chỉnh số biến k={k}).")

    if f_p < 0.001:
        lines.append(f"• F-test p-value < 0.001 → Mô hình có ý nghĩa thống kê rất cao (***)")
    elif f_p < 0.05:
        lines.append(f"• F-test p-value = {f_p:.4f} → Mô hình có ý nghĩa thống kê (**).")
    else:
        lines.append(f"• F-test p-value = {f_p:.4f} → Mô hình KHÔNG có ý nghĩa thống kê (p ≥ 0.05).")

    lines.append(f"• RMSE = {rmse:.4f} (sai số dự báo trung bình).")
    lines.append("")

    # Coefficients
    lines.append("🔢 PHÂN TÍCH HỆ SỐ HỒI QUY")
    for var, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
        if var == "const":
            lines.append(f"• Hằng số (const) = {coef:.4f}.")
            continue
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "⚠ không có ý nghĩa"))
        direction = "tăng" if coef > 0 else "giảm"
        lines.append(f"• {var}: hệ số = {coef:.4f}, p = {pval:.4f} ({sig}) → khi {var} tăng 1 đơn vị, {dep_var} {direction} {abs(coef):.4f} đơn vị (các biến khác không đổi).")

    lines.append("")

    # Residual diagnostics hint
    lines.append("🔍 GỢI Ý CHẨN ĐOÁN")
    if adj_r2 < r2 - 0.05 and k > 2:
        lines.append("• Adjusted R² thấp hơn R² nhiều → có thể có biến thừa, cân nhắc loại bỏ biến không có ý nghĩa.")
    if rmse > (model.model.endog.std() * 0.5):
        lines.append("• RMSE tương đối cao so với độ lệch chuẩn Y → sai số dự báo còn lớn.")
    if model_type in ["Bậc hai (Quadratic)", "Bậc hai Centered"]:
        lines.append("• Mô hình bậc 2: kiểm tra hệ số bậc 2 có ý nghĩa không. Nếu không → quay về mô hình tuyến tính.")
    lines.append("• Kiểm tra biểu đồ Residual: phần dư nên phân bố ngẫu nhiên quanh 0.")
    lines.append("• Q-Q Plot: các điểm gần đường thẳng → phần dư xấp xỉ chuẩn, giả thuyết OLS được thỏa.")

    return "\n".join(lines)


def export_to_excel(run_history):
    """Export all run results to Excel with detailed sheets."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        wb = writer.book
        hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#1f3a5f', 'font_color': '#58a6ff', 'border': 1})
        num_fmt = wb.add_format({'num_format': '#,##0.0000', 'border': 1})
        cell_fmt = wb.add_format({'border': 1, 'font_color': '#e6edf3'})

        # Summary sheet
        summary_rows = []
        for r in run_history:
            summary_rows.append({
                'Run #': r['run_id'],
                'Biến phụ thuộc': r['dep_var'],
                'Biến độc lập': ', '.join(r['ind_vars']),
                'Loại mô hình': r['model_type'],
                'N': r['n'],
                'R²': r['r2'],
                'Adj R²': r['adj_r2'],
                'RMSE': r['rmse'],
                'F-stat': r['fstat'],
                'F p-value': r['f_pvalue'],
            })
        sum_df = pd.DataFrame(summary_rows)
        sum_df.to_excel(writer, sheet_name='Summary', index=False)
        ws_sum = writer.sheets['Summary']
        for col_num, val in enumerate(sum_df.columns):
            ws_sum.write(0, col_num, val, hdr_fmt)
        ws_sum.set_column('A:J', 18, num_fmt)

        # Individual run sheets
        for r in run_history:
            sheet_name = f"Run{r['run_id']}_{r['model_type'][:12].replace(' ','_')}"[:31]
            row = 0

            rs_df, an_df, cf_df = r['tables']
            rs_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
            row += len(rs_df) + 2

            an_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
            row += len(an_df) + 2

            cf_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)

            ws = writer.sheets[sheet_name]
            ws.set_column('B:H', 18, num_fmt)

            # Interpretation
            row2 = row + len(cf_df) + 3
            ws.write(row2, 0, 'DIỄN GIẢI', hdr_fmt)
            for i, line in enumerate(r['interpretation'].split('\n')):
                ws.write(row2 + 1 + i, 0, line, cell_fmt)

    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">📈 Regression<br>Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Phân tích hồi quy tuyến tính & phi tuyến</div>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader("Upload dữ liệu (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])

    if uploaded:
        try:
            df_loaded, hdr_row = load_data(uploaded)
            if df_loaded is not None:
                st.session_state["df"] = df_loaded
                st.session_state["filename"] = uploaded.name
                st.session_state["header_row"] = hdr_row
                st.success(f"✅ Đã tải: {uploaded.name}")
                st.caption(f"{df_loaded.shape[0]} hàng × {df_loaded.shape[1]} cột")
                if hdr_row > 0:
                    st.info(f"📌 Tự phát hiện header tại dòng {hdr_row + 1}")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    if st.session_state["df"] is not None:
        st.divider()
        if st.button("🗑️ Xóa lịch sử chạy"):
            st.session_state["run_history"] = []
            st.session_state["run_counter"] = 0
            st.rerun()
        if st.session_state["run_history"]:
            st.caption(f"🔢 {len(st.session_state['run_history'])} lần chạy đã lưu")

    st.divider()
    st.markdown('<div style="color:#8b949e;font-size:0.75rem;">Powered by statsmodels · OLS</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════
df = st.session_state["df"]

if df is None:
    st.markdown('<div class="hero-title">📈 Regression Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Tải file dữ liệu (.xlsx, .csv) từ sidebar để bắt đầu phân tích hồi quy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card card-accent">
    <b>Tính năng chính:</b><br><br>
    🔍 <b>Kiểm tra chất lượng dữ liệu</b> — phát hiện missing, duplicate, outlier tự động<br>
    📌 <b>Chọn biến linh hoạt</b> — biến phụ thuộc, biến độc lập tùy chọn<br>
    📐 <b>Nhiều loại mô hình</b> — tuyến tính, bậc 2, centered, log, tương tác<br>
    📊 <b>4 biểu đồ mỗi lần chạy</b> — Trend, Residual, Q-Q, Actual vs Predicted<br>
    📋 <b>Bảng so sánh tích lũy</b> — lưu tất cả kết quả để so sánh<br>
    💬 <b>Diễn giải tự động</b> — giải thích hệ số, R², ý nghĩa thống kê<br>
    ⬇️ <b>Xuất Excel chi tiết</b> — đầy đủ bảng ANOVA, hệ số, diễn giải
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Section 1: Data Preview & Quality ─────────────────────────────────────
st.markdown('<div class="section-hdr">① DỮ LIỆU & KIỂM TRA CHẤT LƯỢNG</div>', unsafe_allow_html=True)

# Header detection notice
hdr_row = st.session_state.get("header_row", 0)
if hdr_row > 0:
    st.markdown(
        f'<div class="ok-box">✅ <b>Tự động phát hiện tiêu đề:</b> App đã bỏ qua {hdr_row} dòng trên cùng '
        f'và dùng dòng {hdr_row + 1} làm tên cột. Dữ liệu hiển thị bên dưới đã đúng định dạng.</div>',
        unsafe_allow_html=True
    )

col_prev, col_qual = st.columns([1.6, 1])

with col_prev:
    with st.expander("📋 Xem trước dữ liệu (20 dòng đầu)", expanded=True):
        st.dataframe(df.head(20), use_container_width=True, height=270)

with col_qual:
    st.markdown("**Báo cáo chất lượng dữ liệu**")
    issues = data_quality_report(df)
    level_map = {"ok": "ok-box", "warn": "warn-box", "err": "err-box", "info": "info-box"}
    icon_map  = {"ok": "✅", "warn": "⚠️", "err": "🚨", "info": "ℹ️"}
    for iss in issues:
        css  = level_map.get(iss["level"], "info-box")
        icon = icon_map.get(iss["level"], "•")
        st.markdown(f'<div class="{css}">{icon} {iss["msg"]}</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.caption(f"Kích thước: {df.shape[0]} hàng × {df.shape[1]} cột")
    st.caption(f"Cột số: {len(numeric_cols)} | Cột phi số: {df.shape[1] - len(numeric_cols)}")

st.divider()

# ── Section 2: Model Configuration ────────────────────────────────────────
st.markdown('<div class="section-hdr">② CẤU HÌNH MÔ HÌNH HỒI QUY</div>', unsafe_allow_html=True)

if len(numeric_cols) < 2:
    st.error("Cần ít nhất 2 cột số để thực hiện hồi quy.")
    st.stop()

# ── Smart variable suggestion ─────────────────────────────────────────────
sug_y, sug_x, _, sug_text = suggest_variables(df)

with st.expander("🤖 Gợi ý biến tự động (click để xem / ẩn)", expanded=True):
    st.markdown(sug_text)
    # Correlation heatmap mini-table: show top correlations with suggested Y
    if sug_y and len(numeric_cols) >= 2:
        corr_series = df[numeric_cols].corr()[sug_y].drop(sug_y).sort_values(key=abs, ascending=False)
        corr_df = corr_series.reset_index()
        corr_df.columns = ["Biến X", f"Tương quan với {sug_y}"]
        corr_df[f"Tương quan với {sug_y}"] = corr_df[f"Tương quan với {sug_y}"].round(4)

        def color_corr(val):
            try:
                v = float(val)
                if abs(v) >= 0.7:
                    return "background-color:#1a3a2a; color:#3fb950"
                elif abs(v) >= 0.4:
                    return "background-color:#3a2d10; color:#d29922"
                else:
                    return "background-color:#2a1f1f; color:#8b949e"
            except Exception:
                return ""

        try:
            styled = corr_df.style.map(color_corr, subset=[f"Tương quan với {sug_y}"])
        except AttributeError:
            styled = corr_df.style.applymap(color_corr, subset=[f"Tương quan với {sug_y}"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(200, 35 * len(corr_df) + 38))
        st.caption("🟢 |r| ≥ 0.7 mạnh · 🟡 0.4–0.7 trung bình · ⬜ < 0.4 yếu")

st.markdown("")  # spacing

# ── Variable selectors (pre-filled with suggestion) ───────────────────────
col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1.2, 1])

# Find suggested Y index for selectbox default
sug_y_idx = numeric_cols.index(sug_y) if sug_y in numeric_cols else 0

with col_cfg1:
    dep_var = st.selectbox(
        "🎯 Biến phụ thuộc (Y)",
        options=numeric_cols,
        index=sug_y_idx,
        help="Biến bạn muốn dự báo / giải thích. Đã được gợi ý tự động — có thể thay đổi."
    )

with col_cfg2:
    ind_options = [c for c in numeric_cols if c != dep_var]
    # Use suggested X, filtered to only valid options
    default_x = [c for c in sug_x if c in ind_options] or (ind_options[:1] if ind_options else [])
    ind_vars = st.multiselect(
        "📌 Biến độc lập (X)",
        options=ind_options,
        default=default_x,
        help="Chọn một hoặc nhiều biến giải thích. Đã gợi ý tự động — có thể thêm/bỏ."
    )

with col_cfg3:
    model_type = st.selectbox(
        "📐 Loại mô hình",
        options=[
            "Tuyến tính (Linear)",
            "Bậc hai (Quadratic)",
            "Bậc hai Centered",
            "Logarithmic (Log Y)",
            "Tương tác (Interaction)"
        ],
        help=(
            "Linear: Y = a + b₁X₁ + … | "
            "Quadratic: + bX² | "
            "Centered: giảm đa cộng tuyến | "
            "Log Y: ln(Y) = … | "
            "Interaction: thêm tích X₁×X₂"
        )
    )

# Model type info box
MODEL_INFO = {
    "Tuyến tính (Linear)":     "Hồi quy tuyến tính chuẩn. Phù hợp khi quan hệ Y~X là đường thẳng.",
    "Bậc hai (Quadratic)":     "Thêm X² vào mô hình — phù hợp khi đồ thị phần dư có dạng cong (U-shape).",
    "Bậc hai Centered":        "Bậc 2 với X được trừ giá trị trung bình — giảm đa cộng tuyến giữa X và X².",
    "Logarithmic (Log Y)":     "Biến đổi ln(Y) trước khi hồi quy — phù hợp khi Y tăng theo cấp số nhân.",
    "Tương tác (Interaction)": "Thêm tích X₁×X₂ — phù hợp khi ảnh hưởng của X₁ phụ thuộc vào X₂.",
}
st.markdown(f'<div class="info-box">ℹ️ <b>{model_type}</b>: {MODEL_INFO[model_type]}</div>', unsafe_allow_html=True)

st.divider()

# ── Section 3: Run Model ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">③ CHẠY MÔ HÌNH</div>', unsafe_allow_html=True)

run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Chạy Hồi Quy", type="primary", use_container_width=True)

if run_clicked:
    if not ind_vars:
        st.error("Vui lòng chọn ít nhất 1 biến độc lập.")
    else:
        with st.spinner("Đang chạy mô hình..."):
            try:
                model, df_clean, y = run_regression(df, dep_var, ind_vars, model_type)
                rs_df, an_df, cf_df = get_ols_stats(model)
                interp = interpret_model(model, model_type, dep_var, ind_vars)
                plot_buf = make_plots(model, df_clean, dep_var, ind_vars, model_type)

                st.session_state["run_counter"] += 1
                run_id = st.session_state["run_counter"]

                run_result = {
                    "run_id": run_id,
                    "dep_var": dep_var,
                    "ind_vars": ind_vars.copy(),
                    "model_type": model_type,
                    "n": int(model.nobs),
                    "r2": round(model.rsquared, 4),
                    "adj_r2": round(model.rsquared_adj, 4),
                    "rmse": round(float(np.sqrt(model.mse_resid)), 4),
                    "fstat": round(model.fvalue, 4),
                    "f_pvalue": round(model.f_pvalue, 6),
                    "tables": (rs_df, an_df, cf_df),
                    "interpretation": interp,
                    "plot_buf": plot_buf,
                }
                st.session_state["run_history"].append(run_result)
                st.success(f"✅ Run #{run_id} hoàn thành — R² = {model.rsquared:.4f}")

            except Exception as e:
                st.error(f"Lỗi khi chạy mô hình: {e}")


# ── Section 4: Latest Run Results ──────────────────────────────────────────
if st.session_state["run_history"]:
    latest = st.session_state["run_history"][-1]

    st.markdown(f'<div class="section-hdr">④ KẾT QUẢ MÔ HÌNH — Run #{latest["run_id"]}: {latest["dep_var"]} ~ {" + ".join(latest["ind_vars"])} [{latest["model_type"]}]</div>', unsafe_allow_html=True)

    # Metrics row
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("R²", f"{latest['r2']:.4f}")
    mc2.metric("Adj R²", f"{latest['adj_r2']:.4f}")
    mc3.metric("RMSE", f"{latest['rmse']:.4f}")
    mc4.metric("F-stat", f"{latest['fstat']:.2f}")
    mc5.metric("F p-value", f"{latest['f_pvalue']:.4f}")

    # Tables
    tab1, tab2, tab3 = st.tabs(["📊 Regression Statistics", "📋 ANOVA", "🔢 Coefficients"])
    rs_df, an_df, cf_df = latest["tables"]
    with tab1:
        st.dataframe(rs_df, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(an_df, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(cf_df, use_container_width=True, hide_index=True)

    # Charts
    st.markdown("**📈 Biểu đồ phân tích**")
    latest["plot_buf"].seek(0)
    st.image(latest["plot_buf"], use_container_width=True)

    # Interpretation
    st.markdown("**💬 Diễn giải kết quả**")
    st.markdown(f'<div class="interpret-box">{latest["interpretation"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    st.divider()


# ── Section 5: Run History Comparison ──────────────────────────────────────
if len(st.session_state["run_history"]) > 0:
    st.markdown('<div class="section-hdr">⑤ SO SÁNH CÁC MÔ HÌNH ĐÃ CHẠY</div>', unsafe_allow_html=True)

    history = st.session_state["run_history"]
    compare_rows = []
    for r in history:
        compare_rows.append({
            "Run #": r["run_id"],
            "Y": r["dep_var"],
            "X (biến độc lập)": ", ".join(r["ind_vars"]),
            "Mô hình": r["model_type"],
            "N": r["n"],
            "R²": r["r2"],
            "Adj R²": r["adj_r2"],
            "RMSE": r["rmse"],
            "F-stat": r["fstat"],
            "F p-value": r["f_pvalue"],
        })

    cmp_df = pd.DataFrame(compare_rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Best model recommendation
    if len(history) > 1:
        best_idx = cmp_df["Adj R²"].idxmax()
        best = cmp_df.iloc[best_idx]
        st.markdown(f"""
        <div class="ok-box">
        🏆 <b>Mô hình tốt nhất theo Adjusted R²:</b> Run #{int(best['Run #'])} — {best['Mô hình']}
        với Y = {best['Y']}, X = {best['X (biến độc lập)']}<br>
        Adj R² = <b>{best['Adj R²']:.4f}</b>, RMSE = <b>{best['RMSE']:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

        # Additional guidance
        min_rmse_idx = cmp_df["RMSE"].idxmin()
        if min_rmse_idx != best_idx:
            best_rmse = cmp_df.iloc[min_rmse_idx]
            st.markdown(f"""
            <div class="info-box">
            ℹ️ <b>RMSE thấp nhất:</b> Run #{int(best_rmse['Run #'])} — {best_rmse['Mô hình']}
            (RMSE = {best_rmse['RMSE']:.4f}).
            Nếu mục tiêu là tối thiểu sai số dự báo, hãy xem xét mô hình này.
            </div>
            """, unsafe_allow_html=True)

    # Detailed history expander
    if len(history) > 1:
        with st.expander("🔍 Xem chi tiết từng lần chạy"):
            for r in history:
                st.markdown(f"**Run #{r['run_id']}: {r['dep_var']} ~ {', '.join(r['ind_vars'])} [{r['model_type']}]**")
                _, _, cf = r["tables"]
                st.dataframe(cf, use_container_width=True, hide_index=True)
                st.caption(f"R² = {r['r2']} | Adj R² = {r['adj_r2']} | RMSE = {r['rmse']}")
                st.divider()

    st.divider()

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⑥ XUẤT KẾT QUẢ EXCEL</div>', unsafe_allow_html=True)

    excel_col, info_col = st.columns([1, 2])
    with excel_col:
        excel_bytes = export_to_excel(st.session_state["run_history"])
        fname = f"RegressionResults_{st.session_state['filename'].split('.')[0]}.xlsx"
        st.download_button(
            label="⬇️ Tải xuống Excel (tất cả lần chạy)",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with info_col:
        st.markdown("""
        <div class="info-box">
        File Excel bao gồm:<br>
        • <b>Sheet Summary</b>: bảng so sánh tất cả lần chạy<br>
        • <b>Sheet Run#N</b>: Regression Statistics + ANOVA + Coefficients + Diễn giải cho từng lần chạy
        </div>
        """, unsafe_allow_html=True)
