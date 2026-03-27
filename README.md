# 🛡️ NIDS — Network Intrusion Detection System

> A military HUD-aesthetic network intrusion detection system powered by a Random Forest classifier trained on the real KDD Cup 99 dataset.

![Military HUD](https://img.shields.io/badge/aesthetic-Military_HUD-00ff41?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-00ff41?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-00ff41?style=for-the-badge&logo=streamlit)
![Random Forest](https://img.shields.io/badge/Model-Random_Forest-00ff41?style=for-the-badge)
![KDD Cup 99](https://img.shields.io/badge/Dataset-KDD_Cup_99-00ff41?style=for-the-badge)

---

## 🖥️ Live Demo

🔗 [Click to Launch NIDS](https://your-streamlit-url.streamlit.app)

---

## 📌 What It Does

Simulates a real-time network security operations center. A Random Forest model trained on the KDD Cup 99 dataset classifies incoming network packets into 5 categories — Normal, DoS, Probe, R2L, U2R — and displays results on a live military HUD dashboard.

---

## ⚙️ Features

| Feature | Description |
|---|---|
| 🤖 ML Detection | Random Forest classifier trained on real KDD Cup 99 data |
| 📡 Live Simulation | Real-time packet stream updating every 0.6 seconds |
| ☢️ Threat Level HUD | 4-tier threat indicator — LOW / ELEVATED / HIGH / CRITICAL |
| 📊 Live Chart | Line chart showing attack frequency over time |
| 🧠 Model Intel | Feature importances + classifier stats |
| 🔍 Threat Analysis | Expandable attack type reference cards |

---

## 🧠 How It Works
KDD Cup 99 Dataset (494K packets) loaded
↓
50,000 packet sample selected for training
↓
Random Forest trained on 17 network features
↓
Live simulation generates new packets
↓
Model classifies each packet in real time
→ NORMAL — legitimate traffic
→ DoS — Denial of Service flood
→ PROBE — Port scan / Reconnaissance
→ R2L — Remote to Local unauthorized access
→ U2R — User to Root privilege escalation
↓
Threat level computed from attack ratio
↓
Military HUD dashboard updates live

text


---

## 🛠️ Tech Stack

- **ML Model** — Random Forest (scikit-learn, 100 estimators, max depth 15)
- **Dataset** — KDD Cup 99 (10% subset, 494K labeled network packets)
- **Backend** — Python, Streamlit
- **Visualization** — Streamlit native charts
- **Fonts** — Orbitron, Share Tech Mono
- **Deploy** — Streamlit Cloud

---

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees |
| Max Depth | 15 nodes |
| Training Set | ~40,000 packets |
| Test Set | ~10,000 packets |
| Features | 17 network signals |
| Classes | 5 (Normal, DoS, Probe, R2L, U2R) |
| Dataset | KDD Cup 99 |

---

## 🔍 17 Features Used

| Feature | What It Measures |
|---|---|
| `duration` | Length of connection |
| `src_bytes` | Bytes sent from source |
| `dst_bytes` | Bytes sent to destination |
| `logged_in` | Successful login status |
| `count` | Connections to same host |
| `srv_count` | Connections to same service |
| `serror_rate` | % SYN error connections |
| `srv_serror_rate` | % SYN errors same service |
| `rerror_rate` | % REJ error connections |
| `same_srv_rate` | % connections same service |
| `diff_srv_rate` | % connections different service |
| `hot` | Number of hot indicators |
| `num_failed_logins` | Failed login attempts |
| `num_compromised` | Compromised conditions |
| `land` | Source/dest same host/port |
| `wrong_fragment` | Wrong fragments count |
| `urgent` | Urgent packets count |

---

## 🚨 Attack Categories

| Category | Description | Real Examples |
|---|---|---|
| **DoS** | Overwhelms target with traffic | SYN Flood, Ping of Death, Smurf |
| **Probe** | Scans network for vulnerabilities | Nmap, Portsweep, Satan |
| **R2L** | Unauthorized remote access | FTP Write, Guess Password |
| **U2R** | Local user gains root access | Buffer Overflow, Rootkit |

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/nids-project
cd nids-project
pip install -r requirements.txt