"""Enterprise CSS Design System for Enterprise Knowledge Navigator (EKN).

Provides minimal, modern, and restrained styling adhering to enterprise typography,
clean borders, status badges, metric cards, and citation components.
"""

ENTERPRISE_CSS = """
<style>
/* ========================================================================= */
/* Global Enterprise Light SaaS Theme Overrides                              */
/* ========================================================================= */
:root {
    --bg-page: #F8FAFC;
    --bg-surface: #FFFFFF;
    --bg-sidebar: #F5F5FF;
    --bg-subtle: #F1F5F9;
    --border-color: #E2E8F0;
    --text-primary: #172033;
    --text-secondary: #475569;
    --text-muted: #64748B;
    --brand-primary: #4F46E5;
    --brand-hover: #4338CA;
    --active-bg: #EEF2FF;
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
}

/* Hide Streamlit default chrome & deploy button for native SaaS look */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { background-color: transparent !important; }
.stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* Force light background on entire app viewport */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #F8FAFC !important;
    color: #172033 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: #F8FAFC !important;
}

/* Full-width content container utilizing available space beside sidebar */
.main .block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* Sidebar: pale lavender background (#F5F5FF) with clean right border */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #F5F5FF !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* ========================================================================= */
/* Clean Enterprise Radio & Segmented Navigation Tabs                        */
/* ========================================================================= */

/* 1. HIDE ALL DEFAULT CIRCULAR RADIO BULLETS EVERYWHERE IN APP */
[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]),
[data-testid="stRadioOption"] .ex50key4,
[data-testid="stRadioOption"] > div > div[class*="ex50key4"],
[data-testid="stRadioOption"] input[type="radio"],
[data-testid="stRadio"] [data-testid="stRadioOption"] > span:first-child {
    display: none !important;
}

/* 2. SIDEBAR NAVIGATION RADIO (VERTICAL LIST) */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.25rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    padding: 0.5rem 0.85rem !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #334155 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:hover,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background-color: #EDE9FE !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]:hover p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover p {
    color: #4338CA !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #EEF2FF !important;
    border-color: #C7D2FE !important;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.08) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p {
    color: #4F46E5 !important;
    font-weight: 700 !important;
}

/* 3. HORIZONTAL SEGMENTED CONTROLS (TABS) */
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {
    background: #F1F5F9 !important;
    padding: 3px !important;
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    gap: 4px !important;
    display: inline-flex !important;
    align-items: center !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"],
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 0.35rem 0.85rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    margin: 0 !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] p,
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label p {
    color: #475569 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"]:hover,
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:hover {
    color: #1E293B !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"]:hover p,
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:hover p {
    color: #1E293B !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:has(input:checked),
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label[data-checked="true"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}

[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:has(input:checked) p,
[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label[data-checked="true"] p {
    color: #4F46E5 !important;
    font-weight: 700 !important;
}

/* 4. VERTICAL RADIO (E.G. ACTIVE SCOPE IN RIGHT PANEL) */
[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) [data-testid="stRadioOption"],
[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) > label {
    padding: 0.4rem 0.65rem !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
}

[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) [data-testid="stRadioOption"]:hover,
[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) > label:hover {
    background-color: #F8FAFC !important;
}

[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) > label:has(input:checked) {
    background-color: #EEF2FF !important;
    border-color: #C7D2FE !important;
}

[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stRadio"] div[role="radiogroup"]:not([aria-orientation="horizontal"]) > label:has(input:checked) p {
    color: #4F46E5 !important;
    font-weight: 600 !important;
}

/* ========================================================================= */
/* Streamlit Native Tabs (st.tabs) Segmented Control Styling                 */
/* ========================================================================= */
[data-testid="stTabs"] {
    background: transparent !important;
}

div[data-baseweb="tab-list"] {
    background-color: #F1F5F9 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 4px !important;
    display: inline-flex !important;
}

button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.95rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

button[data-baseweb="tab"]:hover {
    color: #1E293B !important;
    background-color: rgba(255, 255, 255, 0.5) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #4F46E5 !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}

div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Headers and typography */
h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em;
}

.stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"] p {
    color: #172033;
    line-height: 1.55;
}

[data-testid="stWidgetLabel"] label, label[data-baseweb="label"] {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Containers and Form Cards */
[data-testid="stVerticalBlock"] > div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stContainer"][border="true"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

/* Form Inputs & Select Controls */
input[type="text"], input[type="password"], textarea, [data-baseweb="input"], [data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    color: #172033 !important;
}

input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15) !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border: 1px solid #4F46E5 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}

button[kind="primary"]:hover {
    background-color: #4338CA !important;
    border-color: #4338CA !important;
}

button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}

button[kind="secondary"]:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Header Banner */
.ekn-header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.85rem;
    margin-bottom: 1.25rem;
}

.ekn-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.ekn-subtitle {
    font-size: 0.92rem;
    color: #64748b;
    margin-top: 0;
}

/* Metric / Stat Cards */
.ekn-stat-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.ekn-stat-icon-wrapper {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}

.ekn-stat-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.15;
}

.ekn-stat-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-top: 0.35rem;
}

/* Status Badges */
.ekn-badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 4px;
    line-height: 1.2;
    text-transform: capitalize;
}

.ekn-badge-online, .ekn-badge-completed, .ekn-badge-ready {
    background-color: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.ekn-badge-processing, .ekn-badge-indexing {
    background-color: #fefce8;
    color: #a16207;
    border: 1px solid #fef08a;
}

.ekn-badge-queued, .ekn-badge-uploaded {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.ekn-badge-failed, .ekn-badge-offline {
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.ekn-badge-neutral, .ekn-badge-not-ready {
    background-color: #f8fafc;
    color: #64748b;
    border: 1px solid #cbd5e1;
}

.ekn-badge-duplicate {
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}

/* Empty State Box */
.ekn-empty-state {
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    padding: 2.25rem 1.5rem;
    text-align: center;
    background-color: #f8fafc;
    margin: 0.75rem 0;
}

.ekn-empty-title {
    font-weight: 600;
    color: #1e293b;
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}

.ekn-empty-desc {
    color: #64748b;
    font-size: 0.88rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.45;
}

/* Citation Card */
.ekn-citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background-color: #EEF2FF;
    color: #4F46E5;
    border: 1px solid #C7D2FE;
    border-radius: 6px;
    padding: 0.22rem 0.65rem;
    font-size: 0.80rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-top: 0.35rem;
    margin-bottom: 0.35rem;
    text-decoration: none;
    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.05);
    transition: all 0.15s ease;
}

.ekn-citation-pill:hover {
    background-color: #E0E7FF;
    border-color: #A5B4FC;
    color: #3730A3;
}

.ekn-citation-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
}

.ekn-citation-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}

.ekn-citation-tag {
    background-color: #dbeafe;
    color: #1d4ed8;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}

.ekn-citation-source {
    font-weight: 600;
    font-size: 0.88rem;
    color: #1e293b;
}

.ekn-citation-page {
    color: #64748b;
    font-size: 0.82rem;
}

.ekn-citation-section {
    font-size: 0.8rem;
    color: #475569;
    font-style: italic;
    margin-bottom: 0.35rem;
}

.ekn-citation-excerpt {
    font-size: 0.84rem;
    color: #334155;
    background-color: #ffffff;
    border: 1px solid #f1f5f9;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.4rem;
    line-height: 1.4;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Authentication Role Cards */
.ekn-role-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 2.0rem 1.75rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    min-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease-in-out;
}

.ekn-role-card:hover {
    border-color: #4F46E5;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.1);
}

.ekn-role-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}

.ekn-role-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
}

.ekn-role-desc {
    font-size: 0.92rem;
    color: #475569;
    line-height: 1.5;
    margin-bottom: 1.25rem;
}

.ekn-portal-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.ekn-portal-badge-admin {
    background: #eef2ff;
    color: #4f46e5;
    border: 1px solid #c7d2fe;
}

.ekn-portal-badge-emp {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

/* Employee Workspace Custom Three-Panel Styling */
.ekn-emp-header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0.65rem 1.0rem;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.ekn-conv-item {
    padding: 0.5rem 0.65rem;
    border-radius: 6px;
    margin-bottom: 0.25rem;
    border: 1px solid transparent;
}

.ekn-conv-item:hover {
    background: #f1f5f9;
}

.ekn-conv-item-active {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
}

.ekn-empty-chat {
    text-align: center;
    max-width: 580px;
    margin: 2.5rem auto 1.5rem auto;
}

.ekn-empty-chat-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.ekn-empty-chat-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* Assistant Markdown Headings Normalization */
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-top: 1.0rem !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.35rem !important;
}

[data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {
    margin-top: 0.25rem !important;
    margin-bottom: 0.75rem !important;
    padding-left: 1.25rem !important;
}

[data-testid="stChatMessage"] li {
    margin-bottom: 0.25rem !important;
    line-height: 1.5 !important;
    color: #172033 !important;
}
</style>
"""


def format_bytes(bytes_count: int) -> str:
    """Format byte count into human-readable representation."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.1f} MB"
    return f"{bytes_count / (1024 * 1024 * 1024):.2f} GB"


def format_date(dt_str: str) -> str:
    """Format ISO datetime string for clean display."""
    if not dt_str:
        return "-"
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str[:16]


def get_status_badge_html(status: str) -> str:
    """Generate clean HTML badge for document processing or indexing status."""
    badge_map = {
        "completed": "ekn-badge-completed",
        "ready": "ekn-badge-ready",
        "uploaded": "ekn-badge-uploaded",
        "queued": "ekn-badge-queued",
        "processing": "ekn-badge-processing",
        "indexing": "ekn-badge-indexing",
        "failed": "ekn-badge-failed",
        "offline": "ekn-badge-offline",
        "online": "ekn-badge-online",
        "duplicate": "ekn-badge-duplicate",
        "not_ready": "ekn-badge-not-ready",
    }
    cls = badge_map.get(status.lower().replace(" ", "_"), "ekn-badge-neutral")
    return f"<span class='ekn-badge {cls}'>{status}</span>"
