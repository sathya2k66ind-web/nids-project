import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# MILITARY HUD CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #010a01 !important;
    color: #00ff41 !important;
}

.stApp { background-color: #010a01 !important; }

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.15) 2px,
        rgba(0,0,0,0.15) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

section[data-testid="stSidebar"] {
    background: #000a00 !important;
    border-right: 1px solid #003300 !important;
}
section[data-testid="stSidebar"] * { color: #00ff41 !important; }

div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #00ff41 !important;
    text-shadow: 0 0 10px #00ff41, 0 0 20px rgba(0,255,65,0.4) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #006600 !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricDelta"] svg { display: none; }

div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #00ff41 !important;
    color: #00ff41 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 8px rgba(0,255,65,0.2) !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(0,255,65,0.1) !important;
    box-shadow: 0 0 20px rgba(0,255,65,0.5) !important;
}

div[data-testid="stDataFrame"] { border: 1px solid #003300 !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid #003300 !important;
    border-radius: 0 !important;
    color: #006600 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,65,0.08) !important;
    border-color: #00ff41 !important;
    color: #00ff41 !important;
    box-shadow: 0 0 10px rgba(0,255,65,0.3) !important;
}

div[data-testid="stProgress"] > div > div {
    background: #00ff41 !important;
    box-shadow: 0 0 8px #00ff41 !important;
}

div[data-baseweb="select"] > div {
    background: #000a00 !important;
    border: 1px solid #003300 !important;
    color: #00ff41 !important;
    border-radius: 0 !important;
}

.hud-box {
    border: 1px solid #003300;
    background: rgba(0,255,65,0.02);
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
    font-family: 'Share Tech Mono', monospace;
}
.hud-box::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 10px; height: 10px;
    border-top: 2px solid #00ff41;
    border-left: 2px solid #00ff41;
}
.hud-box::after {
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 10px; height: 10px;
    border-bottom: 2px solid #00ff41;
    border-right: 2px solid #00ff41;
}

.hud-title {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #006600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #002200;
    padding-bottom: 6px;
}

.threat-critical { color: #ff0000 !important; text-shadow: 0 0 15px #ff0000, 0 0 30px rgba(255,0,0,0.5) !important; animation: threatBlink 0.5s infinite !important; }
.threat-high     { color: #ff6600 !important; text-shadow: 0 0 12px #ff6600 !important; }
.threat-elevated { color: #ffaa00 !important; text-shadow: 0 0 10px #ffaa00 !important; }
.threat-low      { color: #00ff41 !important; text-shadow: 0 0 8px #00ff41 !important; }

@keyframes threatBlink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

.packet-normal { color: #00ff41; }
.packet-dos    { color: #ff0000; text-shadow: 0 0 6px #ff0000; }
.packet-probe  { color: #ffaa00; text-shadow: 0 0 6px #ffaa00; }
.packet-r2l    { color: #ff6600; text-shadow: 0 0 6px #ff6600; }
.packet-u2r    { color: #ff00ff; text-shadow: 0 0 6px #ff00ff; }

.stat-row   { display: flex; gap: 8px; margin-bottom: 8px; font-size: 13px; }
.stat-label { color: #006600; min-width: 120px; }
.stat-val   { color: #00ff41; text-shadow: 0 0 6px #00ff41; }

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 13px;
    color: #00ff41;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #003300;
    padding-bottom: 8px;
    margin-bottom: 16px;
    text-shadow: 0 0 8px rgba(0,255,65,0.4);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD REAL KDD CUP 99 DATASET
# ─────────────────────────────────────────
COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

# KDD99 raw label → our 5 categories
LABEL_MAP = {
    'normal.': 'normal',
    # DoS
    'back.':'dos','land.':'dos','neptune.':'dos','pod.':'dos',
    'smurf.':'dos','teardrop.':'dos','apache2.':'dos','udpstorm.':'dos',
    'processtable.':'dos','worm.':'dos',
    # Probe
    'ipsweep.':'probe','nmap.':'probe','portsweep.':'probe','satan.':'probe',
    'mscan.':'probe','saint.':'probe',
    # R2L
    'ftp_write.':'r2l','guess_passwd.':'r2l','imap.':'r2l','multihop.':'r2l',
    'phf.':'r2l','spy.':'r2l','warezclient.':'r2l','warezmaster.':'r2l',
    'sendmail.':'r2l','named.':'r2l','snmpgetattack.':'r2l','snmpguess.':'r2l',
    'xlock.':'r2l','xsnoop.':'r2l','httptunnel.':'r2l',
    # U2R
    'buffer_overflow.':'u2r','loadmodule.':'u2r','perl.':'u2r','rootkit.':'u2r',
    'mailbomb.':'u2r','ps.':'u2r','sqlattack.':'u2r','xterm.':'u2r',
}

FEATURES = [
    'duration','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'same_srv_rate','diff_srv_rate'
]

@st.cache_data
def load_real_dataset():
    df = pd.read_csv('kddcup.csv', header=None, names=COLUMNS)
    df['label'] = df['label'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])          # drop any unknown labels
    df = df[FEATURES + ['label']]
    # Sample 50k rows so training stays fast on Streamlit Cloud
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
    return df


# ─────────────────────────────────────────
# TRAIN MODEL ON REAL DATA
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    df = load_real_dataset()
    X  = df[FEATURES]
    le = LabelEncoder()
    y  = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    fi = pd.DataFrame({
        'feature':    FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    n_train = len(X_train)
    n_test  = len(X_test)

    return model, le, acc, fi, n_train, n_test


# ─────────────────────────────────────────
# SIMULATE ONE PACKET (live sim unchanged)
# ─────────────────────────────────────────
def simulate_packet(model, le):
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    weights      = [0.55, 0.22, 0.10, 0.08, 0.05]
    true_label   = np.random.choice(attack_types, p=weights)

    templates = {
        'normal': dict(duration=random.randint(0,300),  src_bytes=random.randint(100,5000), dst_bytes=random.randint(100,5000), land=0, wrong_fragment=random.randint(0,1), urgent=0, hot=random.randint(0,5),  num_failed_logins=0, logged_in=1, num_compromised=0, count=random.randint(1,100),  srv_count=random.randint(1,100),  serror_rate=round(random.uniform(0,.1),2),  srv_serror_rate=round(random.uniform(0,.1),2), rerror_rate=round(random.uniform(0,.1),2),  same_srv_rate=round(random.uniform(.8,1),2), diff_srv_rate=round(random.uniform(0,.1),2)),
        'dos':    dict(duration=random.randint(0,5),    src_bytes=random.randint(0,500),    dst_bytes=0,                        land=random.randint(0,1), wrong_fragment=random.randint(0,5), urgent=0, hot=0, num_failed_logins=0, logged_in=0, num_compromised=0, count=random.randint(200,512), srv_count=random.randint(200,512), serror_rate=round(random.uniform(.8,1),2),  srv_serror_rate=round(random.uniform(.8,1),2), rerror_rate=round(random.uniform(0,.1),2),  same_srv_rate=round(random.uniform(.9,1),2), diff_srv_rate=round(random.uniform(0,.05),2)),
        'probe':  dict(duration=random.randint(0,10),   src_bytes=random.randint(0,1000),   dst_bytes=random.randint(0,1000),   land=0, wrong_fragment=0, urgent=0, hot=random.randint(0,3), num_failed_logins=0, logged_in=0, num_compromised=0, count=random.randint(50,512),  srv_count=random.randint(1,50),   serror_rate=round(random.uniform(0,.3),2),  srv_serror_rate=round(random.uniform(0,.3),2), rerror_rate=round(random.uniform(.3,.8),2), same_srv_rate=round(random.uniform(0,.3),2), diff_srv_rate=round(random.uniform(.4,1),2)),
        'r2l':    dict(duration=random.randint(0,100),  src_bytes=random.randint(100,3000), dst_bytes=random.randint(100,3000), land=0, wrong_fragment=0, urgent=0, hot=random.randint(0,10), num_failed_logins=random.randint(1,5), logged_in=0, num_compromised=0, count=random.randint(1,30),   srv_count=random.randint(1,30),   serror_rate=round(random.uniform(0,.2),2),  srv_serror_rate=round(random.uniform(0,.2),2), rerror_rate=round(random.uniform(0,.2),2),  same_srv_rate=round(random.uniform(.5,1),2), diff_srv_rate=round(random.uniform(0,.2),2)),
        'u2r':    dict(duration=random.randint(0,50),   src_bytes=random.randint(100,2000), dst_bytes=random.randint(100,2000), land=0, wrong_fragment=0, urgent=random.randint(0,3), hot=random.randint(5,30), num_failed_logins=random.randint(0,3), logged_in=1, num_compromised=random.randint(1,10), count=random.randint(1,20), srv_count=random.randint(1,20), serror_rate=round(random.uniform(0,.2),2), srv_serror_rate=round(random.uniform(0,.2),2), rerror_rate=round(random.uniform(0,.2),2), same_srv_rate=round(random.uniform(.5,1),2), diff_srv_rate=round(random.uniform(0,.3),2)),
    }

    pkt = templates[true_label]
    X   = pd.DataFrame([pkt])[FEATURES]
    pred_encoded = model.predict(X)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]
    conf         = model.predict_proba(X)[0].max()
    src_ip = f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"
    dst_ip = f"192.168.{random.randint(0,5)}.{random.randint(1,50)}"

    return {
        'src_ip':     src_ip,
        'dst_ip':     dst_ip,
        'true':       true_label,
        'pred':       pred_label,
        'confidence': round(conf * 100, 1),
        'port':       random.choice([80, 443, 22, 21, 23, 3306, 8080, 53]),
        'protocol':   random.choice(['TCP', 'UDP', 'ICMP']),
        'bytes':      pkt['src_bytes'],
    }


# ─────────────────────────────────────────
# THREAT LEVEL
# ─────────────────────────────────────────
def get_threat(attack_counts):
    total   = sum(attack_counts.values()) or 1
    attacks = total - attack_counts.get('normal', 0)
    ratio   = attacks / total
    if ratio >= 0.5:   return 'CRITICAL', '#ff0000'
    elif ratio >= 0.3: return 'HIGH',     '#ff6600'
    elif ratio >= 0.1: return 'ELEVATED', '#ffaa00'
    else:              return 'LOW',      '#00ff41'


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'launched'      not in st.session_state: st.session_state.launched      = False
if 'running'       not in st.session_state: st.session_state.running       = False
if 'packets'       not in st.session_state: st.session_state.packets       = []
if 'attack_counts' not in st.session_state: st.session_state.attack_counts = {'normal':0,'dos':0,'probe':0,'r2l':0,'u2r':0}
if 'history'       not in st.session_state: st.session_state.history       = []


# ─────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────
def show_landing():
    st.markdown("""
    <div style="min-height:90vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 20px;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#003300;letter-spacing:4px;margin-bottom:20px;text-transform:uppercase;">
        ██ CLASSIFIED // NETWORK DEFENSE SYSTEM // CLEARANCE REQUIRED ██
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:clamp(28px,5vw,52px);font-weight:900;color:#00ff41;
                text-shadow:0 0 20px #00ff41,0 0 40px rgba(0,255,65,0.4);
                letter-spacing:4px;margin-bottom:8px;">
        INTRUSION DETECTION
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:16px;color:#006600;
                letter-spacing:3px;margin-bottom:36px;">
        >> NETWORK SECURITY SYSTEM v2.0 // ML-POWERED // KDD CUP 99
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:700px;width:100%;margin-bottom:40px;">
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">DETECTION ENGINE</div>
            <div style="font-size:14px;color:#00ff41;">RANDOM FOREST<br><span style="color:#005500;font-size:12px;">100 estimators</span></div>
        </div>
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">THREAT CLASSES</div>
            <div style="font-size:14px;color:#00ff41;">5 CATEGORIES<br><span style="color:#005500;font-size:12px;">DoS·Probe·R2L·U2R</span></div>
        </div>
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">DATASET</div>
            <div style="font-size:14px;color:#00ff41;">KDD CUP 99<br><span style="color:#005500;font-size:12px;">494K real packets</span></div>
        </div>
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">LIVE SIM</div>
            <div style="font-size:14px;color:#00ff41;">REAL-TIME<br><span style="color:#005500;font-size:12px;">packet stream</span></div>
        </div>
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">THREAT LEVEL</div>
            <div style="font-size:14px;color:#00ff41;">4-TIER HUD<br><span style="color:#005500;font-size:12px;">LOW→CRITICAL</span></div>
        </div>
        <div style="border:1px solid #003300;padding:16px;background:rgba(0,255,65,0.02);">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#006600;letter-spacing:2px;margin-bottom:8px;">FEATURES</div>
            <div style="font-size:14px;color:#00ff41;">17 SIGNALS<br><span style="color:#005500;font-size:12px;">per packet analyzed</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#004400;letter-spacing:2px;margin-bottom:8px;">
        >> SYSTEM STATUS: STANDBY // AWAITING OPERATOR
    </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("◈  INITIALIZE SYSTEM", use_container_width=True):
            st.session_state.launched = True
            st.rerun()


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def show_main():
    with st.spinner("▶ LOADING REAL KDD CUP 99 DATASET & TRAINING MODEL..."):
        model, le, acc, fi, n_train, n_test = train_model()

    threat_level, threat_color = get_threat(st.session_state.attack_counts)
    total_packets = len(st.session_state.packets)
    total_attacks = total_packets - st.session_state.attack_counts.get('normal', 0)

    # TOP BAR
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid #003300;padding-bottom:12px;margin-bottom:20px;">
        <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:900;
                    color:#00ff41;text-shadow:0 0 15px #00ff41;letter-spacing:3px;">
            ◈ NIDS // NETWORK INTRUSION DETECTION
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;
                    color:{threat_color};text-shadow:0 0 10px {threat_color};letter-spacing:2px;">
            THREAT: {threat_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # METRIC ROW
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("PACKETS ANALYZED", f"{total_packets:,}")
    with c2: st.metric("ATTACKS DETECTED", f"{total_attacks:,}")
    with c3: st.metric("MODEL ACCURACY",   f"{acc*100:.1f}%")
    with c4: st.metric("DoS DETECTED",     f"{st.session_state.attack_counts.get('dos',0):,}")
    with c5: st.metric("PROBE DETECTED",   f"{st.session_state.attack_counts.get('probe',0):,}")

    st.markdown("<div style='border-bottom:1px solid #002200;margin:16px 0;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["◈ LIVE MONITOR", "◈ MODEL INTEL", "◈ THREAT ANALYSIS"])

    # ── TAB 1: LIVE MONITOR ──
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown('<div class="section-header">// PACKET STREAM //</div>', unsafe_allow_html=True)

            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                if st.button("▶  START MONITORING", use_container_width=True):
                    st.session_state.running = True
            with ctrl2:
                if st.button("◼  STOP", use_container_width=True):
                    st.session_state.running = False
            with ctrl3:
                if st.button("↺  RESET", use_container_width=True):
                    st.session_state.packets       = []
                    st.session_state.attack_counts = {'normal':0,'dos':0,'probe':0,'r2l':0,'u2r':0}
                    st.session_state.history       = []
                    st.session_state.running       = False
                    st.rerun()

            log_placeholder   = st.empty()
            chart_placeholder = st.empty()

            if st.session_state.running:
                for _ in range(8):
                    pkt = simulate_packet(model, le)
                    st.session_state.packets.append(pkt)
                    st.session_state.attack_counts[pkt['pred']] += 1
                    st.session_state.history.append({
                        'tick':   len(st.session_state.history),
                        'normal': st.session_state.attack_counts['normal'],
                        'dos':    st.session_state.attack_counts['dos'],
                        'probe':  st.session_state.attack_counts['probe'],
                        'r2l':    st.session_state.attack_counts['r2l'],
                        'u2r':    st.session_state.attack_counts['u2r'],
                    })

            recent    = list(reversed(st.session_state.packets[-20:]))
            COLOR_MAP = {'normal':'#00ff41','dos':'#ff0000','probe':'#ffaa00','r2l':'#ff6600','u2r':'#ff00ff'}

            log_html = '<div style="font-family:\'Share Tech Mono\',monospace;font-size:13px;border:1px solid #002200;background:#000500;padding:12px;height:320px;overflow-y:auto;">'
            for p in recent:
                c    = COLOR_MAP.get(p['pred'], '#00ff41')
                flag = '' if p['pred'] == 'normal' else ' ⚠'
                log_html += f"""
                <div style="margin-bottom:4px;border-bottom:1px solid #001100;padding-bottom:4px;">
                  <span style="color:#004400;">[PKT]</span>
                  <span style="color:#006600;"> {p['src_ip']} → {p['dst_ip']}</span>
                  <span style="color:#004400;"> {p['protocol']}:{p['port']}</span>
                  <span style="color:{c};font-weight:bold;"> [{p['pred'].upper()}{flag}]</span>
                  <span style="color:#003300;"> {p['confidence']}%</span>
                </div>"""
            log_html += '</div>'
            log_placeholder.markdown(log_html, unsafe_allow_html=True)

            if st.session_state.history:
                hist_df = pd.DataFrame(st.session_state.history).tail(30)
                chart_placeholder.line_chart(
                    hist_df.set_index('tick')[['normal','dos','probe','r2l','u2r']],
                    use_container_width=True,
                    height=200
                )

            if st.session_state.running:
                time.sleep(0.6)
                st.rerun()

        with col_right:
            st.markdown('<div class="section-header">// THREAT HUD //</div>', unsafe_allow_html=True)
            threat_level, threat_color = get_threat(st.session_state.attack_counts)

            st.markdown(f"""
            <div class="hud-box" style="text-align:center;padding:24px;">
                <div class="hud-title">THREAT LEVEL</div>
                <div style="font-family:'Orbitron',monospace;font-size:28px;font-weight:900;
                            color:{threat_color};text-shadow:0 0 20px {threat_color};letter-spacing:4px;">
                    {threat_level}
                </div>
                <div style="margin-top:12px;font-size:12px;color:#004400;letter-spacing:1px;">
                    {total_attacks} THREATS / {total_packets} TOTAL
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:16px;">// ATTACK BREAKDOWN //</div>', unsafe_allow_html=True)
            attack_labels = {
                'normal': ('NORMAL', '#00ff41'),
                'dos':    ('DoS',    '#ff0000'),
                'probe':  ('PROBE',  '#ffaa00'),
                'r2l':    ('R2L',    '#ff6600'),
                'u2r':    ('U2R',    '#ff00ff'),
            }
            for key, (label, color) in attack_labels.items():
                count = st.session_state.attack_counts.get(key, 0)
                pct   = int(count / total_packets * 100) if total_packets else 0
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:12px;margin-bottom:3px;">
                        <span style="color:{color};">{label}</span>
                        <span style="color:#004400;">{count} ({pct}%)</span>
                    </div>
                    <div style="height:6px;background:#001100;border:1px solid #002200;">
                        <div style="height:100%;width:{pct}%;background:{color};box-shadow:0 0 6px {color};transition:width 0.3s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:16px;">// ATTACK DESCRIPTIONS //</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#005500;line-height:1.8;">
                <div><span style="color:#ff0000;">DoS</span> — Denial of Service. Flood attack.</div>
                <div><span style="color:#ffaa00;">PROBE</span> — Port scan. Reconnaissance.</div>
                <div><span style="color:#ff6600;">R2L</span> — Remote to Local. Unauthorized access.</div>
                <div><span style="color:#ff00ff;">U2R</span> — User to Root. Privilege escalation.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: MODEL INTEL ──
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">// MODEL PERFORMANCE //</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="hud-box">
                <div class="hud-title">CLASSIFIER STATS</div>
                <div class="stat-row"><span class="stat-label">ALGORITHM</span><span class="stat-val">Random Forest</span></div>
                <div class="stat-row"><span class="stat-label">ESTIMATORS</span><span class="stat-val">100 trees</span></div>
                <div class="stat-row"><span class="stat-label">MAX DEPTH</span><span class="stat-val">15 nodes</span></div>
                <div class="stat-row"><span class="stat-label">ACCURACY</span><span class="stat-val">{acc*100:.2f}%</span></div>
                <div class="stat-row"><span class="stat-label">FEATURES</span><span class="stat-val">17 signals</span></div>
                <div class="stat-row"><span class="stat-label">DATASET</span><span class="stat-val">KDD Cup 99</span></div>
                <div class="stat-row"><span class="stat-label">TRAINING SET</span><span class="stat-val">{n_train:,} packets</span></div>
                <div class="stat-row"><span class="stat-label">TEST SET</span><span class="stat-val">{n_test:,} packets</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-header">// TOP FEATURES BY IMPORTANCE //</div>', unsafe_allow_html=True)
            top_fi = fi.head(10).copy()
            top_fi['importance_pct'] = (top_fi['importance'] * 100).round(2)

            for _, row in top_fi.iterrows():
                pct = int(row['importance_pct'])
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:12px;margin-bottom:2px;">
                        <span style="color:#00ff41;">{row['feature']}</span>
                        <span style="color:#005500;">{row['importance_pct']}%</span>
                    </div>
                    <div style="height:4px;background:#001100;border:1px solid #002200;">
                        <div style="height:100%;width:{min(pct*3,100)}%;background:#00ff41;box-shadow:0 0 4px #00ff41;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: THREAT ANALYSIS ──
    with tab3:
        st.markdown('<div class="section-header">// ATTACK TYPE REFERENCE //</div>', unsafe_allow_html=True)
        attacks_info = {
            "DoS (Denial of Service)": {
                "color": "#ff0000",
                "desc": "Overwhelms the target with traffic making it unavailable. High packet count, high error rate, near-zero destination bytes.",
                "signals": ["count > 200", "serror_rate > 0.8", "dst_bytes ≈ 0"],
                "real": "SYN Flood, Ping of Death, Smurf Attack"
            },
            "Probe (Reconnaissance)": {
                "color": "#ffaa00",
                "desc": "Scans the network to map open ports and vulnerabilities. High diff_srv_rate, high rerror_rate.",
                "signals": ["diff_srv_rate > 0.4", "rerror_rate > 0.3", "srv_count low"],
                "real": "Port Scan, Nmap, Portsweep"
            },
            "R2L (Remote to Local)": {
                "color": "#ff6600",
                "desc": "Attacker gains unauthorized access from a remote machine. Multiple failed logins, not logged in.",
                "signals": ["num_failed_logins > 0", "logged_in = 0", "hot > 0"],
                "real": "FTP Write, Guess Password, Phf"
            },
            "U2R (User to Root)": {
                "color": "#ff00ff",
                "desc": "Local user gains root access. High hot count, urgent flags, compromised connections.",
                "signals": ["hot > 5", "urgent > 0", "num_compromised > 0"],
                "real": "Buffer Overflow, Rootkit, Loadmodule"
            },
        }
        for name, info in attacks_info.items():
            with st.expander(f"► {name}"):
                st.markdown(f"""
                <div style="font-family:'Share Tech Mono',monospace;padding:8px 0;">
                    <div style="color:{info['color']};font-size:14px;margin-bottom:10px;text-shadow:0 0 8px {info['color']};">
                        {info['desc']}
                    </div>
                    <div style="color:#005500;font-size:12px;margin-bottom:6px;">KEY SIGNALS:</div>
                    {''.join(f'<div style="color:#00aa44;font-size:12px;">  ► {s}</div>' for s in info['signals'])}
                    <div style="color:#005500;font-size:12px;margin-top:8px;">REAL WORLD EXAMPLES:</div>
                    <div style="color:#008800;font-size:12px;">  {info['real']}</div>
                </div>
                """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.markdown('<div class="section-header">// SYSTEM STATUS //</div>', unsafe_allow_html=True)
        status = "◉ ACTIVE" if st.session_state.running else "◎ STANDBY"
        color  = "#00ff41" if st.session_state.running else "#006600"
        st.markdown(f'<div style="color:{color};font-family:\'Orbitron\',monospace;font-size:14px;letter-spacing:2px;margin-bottom:16px;">{status}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#005500;line-height:2;">
            MODEL ......... RF<br>
            DATASET ....... KDD99<br>
            FEATURES ...... 17<br>
            CLASSES ....... 5<br>
            DEPLOY ........ CLOUD<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px;">// HOW IT WORKS //</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#005500;line-height:1.8;">
            1. Real KDD Cup 99 data loaded<br>
               (494K labeled packets)<br><br>
            2. Random Forest trained<br>
               on 40K packet sample<br><br>
            3. Live sim feeds packets<br>
               through trained model<br><br>
            4. Threat level computed<br>
               from attack ratio
        </div>
        """, unsafe_allow_html=True)

        if st.button("← BACK TO LANDING", use_container_width=True):
            st.session_state.launched      = False
            st.session_state.running       = False
            st.session_state.packets       = []
            st.session_state.attack_counts = {'normal':0,'dos':0,'probe':0,'r2l':0,'u2r':0}
            st.session_state.history       = []
            st.rerun()


# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
if not st.session_state.launched:
    show_landing()
else:
    show_main()