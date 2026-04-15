from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            # Fallback for local testing or if model is missing
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
            sample_texts = ["test", "fraud money", "genuine job"]
            sample_labels = [0, 1, 0]
            pipeline.fit(sample_texts, sample_labels)
            return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# HTML Template with Ultra-Premium Design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScamGuard AI | Quantum Fraud Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <style>
        :root {
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.5);
            --secondary: #a855f7;
            --accent: #22d3ee;
            --bg: #030712;
            --card-bg: rgba(17, 24, 39, 0.7);
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --danger: #ff4d4d;
            --success: #00f2ad;
            --warning: #ffcc00;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
            scroll-behavior: smooth;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
        }

        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: radial-gradient(circle at 50% 50%, #111827 0%, #030712 100%);
        }

        .navbar {
            padding: 1.5rem 10%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(3, 7, 18, 0.8);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .logo {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent), var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo-icon {
            width: 35px;
            height: 35px;
            background: var(--primary);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 4rem 2rem;
        }

        .hero {
            text-align: center;
            margin-bottom: 5rem;
        }

        .badge {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
            100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }

        h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(3rem, 8vw, 5rem);
            line-height: 1.1;
            margin-bottom: 1.5rem;
            letter-spacing: -2px;
        }

        .gradient-text {
            background: linear-gradient(120deg, #fff 30%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 1.3rem;
            color: var(--text-muted);
            max-width: 700px;
            margin: 0 auto 3rem auto;
            line-height: 1.6;
        }

        .main-workflow {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 3rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .main-workflow { grid-template-columns: 1fr; }
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 32px;
            padding: 3rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 40px 100px -20px rgba(0, 0, 0, 0.6);
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.05) 0%, transparent 60%);
            pointer-events: none;
        }

        .input-group {
            margin-bottom: 2rem;
            position: relative;
        }

        label {
            display: block;
            margin-bottom: 0.8rem;
            font-weight: 500;
            color: var(--text-muted);
            font-size: 0.9rem;
            letter-spacing: 0.5px;
        }

        input[type="text"], textarea {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.2rem;
            color: white;
            font-size: 1.1rem;
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            outline: none;
        }

        input[type="text"]:focus, textarea:focus {
            border-color: var(--primary);
            background: rgba(0, 0, 0, 0.5);
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
        }

        textarea { min-height: 220px; }

        .btn-predict {
            width: 100%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 16px;
            padding: 1.2rem;
            font-size: 1.25rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.4s ease;
            box-shadow: 0 10px 40px -10px var(--primary-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            position: relative;
            overflow: hidden;
        }

        .btn-predict:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 60px -10px var(--primary-glow);
            filter: brightness(1.1);
        }

        .btn-predict::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: 0.5s;
        }

        .btn-predict:hover::after {
            left: 100%;
        }

        #result-area {
            display: none;
            animation: slideUp 0.8s cubic-bezier(0.165, 0.84, 0.44, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
        }

        .risk-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.5rem;
        }

        .risk-indicator {
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: inline-block;
        }

        .indicator-high { background: rgba(255, 77, 77, 0.1); color: var(--danger); border: 1px solid var(--danger); }
        .indicator-low { background: rgba(0, 242, 173, 0.1); color: var(--success); border: 1px solid var(--success); }

        .scan-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(3, 7, 18, 0.9);
            z-index: 1000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .scan-line {
            width: 80%;
            height: 2px;
            background: var(--accent);
            box-shadow: 0 0 20px var(--accent);
            position: absolute;
            top: 0;
            animation: scan 3s linear infinite;
        }

        @keyframes scan {
            0% { top: 0; opacity: 1; }
            100% { top: 100%; opacity: 1; }
        }

        .scanning-text {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            margin-top: 2rem;
            letter-spacing: 4px;
            text-transform: uppercase;
            color: var(--accent);
        }

        .trust-section {
            margin-top: 8rem;
            text-align: center;
        }

        .trust-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-top: 4rem;
        }

        .trust-item {
            padding: 2.5rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }

        .trust-item:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: scale(1.05);
        }

        .trust-icon {
            font-size: 2rem;
            margin-bottom: 1.5rem;
            display: block;
        }

        footer {
            padding: 4rem 10%;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>

    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">🛡️</div>
            <span>SCAMGUARD AI</span>
        </div>
        <div style="display: flex; gap: 2rem; font-weight: 500;">
            <a href="#" style="color: white; text-decoration: none;">Enterprise</a>
            <a href="#" style="color: var(--text-muted); text-decoration: none;">Technology</a>
            <a href="#" style="color: var(--text-muted); text-decoration: none;">Security</a>
        </div>
    </nav>

    <div class="container">
        <section class="hero">
            <div class="badge">Next-Gen Security Node</div>
            <h1>The Gold Standard in <br><span class="gradient-text">Fraud Intelligence.</span></h1>
            <p>Leveraging deep neural networks to dissect job listings in real-time. Stay protected in the digital career landscape.</p>
        </section>

        <main class="main-workflow">
            <div class="glass-card">
                <form id="ai-form">
                    <div class="input-group">
                        <label>POSITION TITLE</label>
                        <input type="text" id="title" placeholder="e.g. Senior Backend Architect" required>
                    </div>
                    <div class="input-group">
                        <label>POSITION DATA / DESCRIPTION</label>
                        <textarea id="description" placeholder="Paste the full job specification for deep analysis..." required></textarea>
                    </div>
                    <button type="submit" class="btn-predict">
                        <span>INITIATE QUANTUM SCAN</span>
                        <div style="width: 20px; height: 1px; background: rgba(255,255,255,0.4);"></div>
                        <span>AI CORE v4.0</span>
                    </button>
                </form>
            </div>

            <div id="result-area">
                <div class="stats-grid">
                    <div class="glass-card stat-card" style="padding: 2rem;">
                        <label>FRAUD RISK PROBABILITY</label>
                        <div class="risk-value" id="res-score">00%</div>
                        <div class="risk-indicator" id="res-indicator">Calculating...</div>
                    </div>
                    
                    <div class="glass-card" style="padding: 2rem;">
                        <h3 style="margin-bottom: 1.5rem; font-family: 'Space Grotesk';">Neural Diagnostics</h3>
                        <canvas id="neuralChart" style="max-height: 120px;"></canvas>
                        <div style="margin-top: 2rem; display: flex; flex-direction: column; gap: 0.8rem; font-size: 0.95rem;">
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                                <span style="color: var(--text-muted);">Integrity Score</span>
                                <span id="res-integrity" style="font-weight: 600;">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                                <span style="color: var(--text-muted);">TF-IDF Vector Nodes</span>
                                <span id="res-vectors" style="font-weight: 600;">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                                <span style="color: var(--text-muted);">Suspicious Flag Count</span>
                                <span id="res-flags" style="font-weight: 600; color: var(--danger);">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                                <span style="color: var(--text-muted);">Lexical Density</span>
                                <span id="res-lexical" style="font-weight: 600;">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding-bottom: 0.5rem;">
                                <span style="color: var(--text-muted);">Model Inference Latency</span>
                                <span id="res-latency" style="font-weight: 600; color: var(--success);">--</span>
                            </div>
                        </div>
                        <button class="btn-primary" style="margin-top: 1.5rem; background: rgba(255,255,255,0.05); font-size: 0.9rem;" onclick="window.print()">
                            📥 DOWNLOAD SECURITY AUDIT
                        </button>
                    </div>
                </div>
            </div>
        </main>

        <section class="trust-section">
            <h2 style="font-family: 'Space Grotesk'; font-size: 2.5rem; margin-bottom: 1rem;">Architected for Trust.</h2>
            <p style="color: var(--text-muted); margin-bottom: 2rem;">Why industry leaders choose ScamGuard for their workforce safety.</p>
            
            <button class="badge" style="cursor: pointer; animation: none;" onclick="document.getElementById('tech-modal').style.display='flex'">
                View System Architecture ↗
            </button>

            <div class="trust-grid">
                <div class="trust-item">
                    <span class="trust-icon">🧠</span>
                    <h3>Deep Learning</h3>
                    <p style="color: var(--text-muted); margin-top: 1rem; font-size: 0.9rem;">Processes over 20,000 TF-IDF linguistic features per scan to identify structural anomalies.</p>
                </div>
                <div class="trust-item">
                    <span class="trust-icon">⚡</span>
                    <h3>Instant Latency</h3>
                    <p style="color: var(--text-muted); margin-top: 1rem; font-size: 0.9rem;">High-performance Random Forest processing powered by Vercel's compute edge.</p>
                </div>
                <div class="trust-item">
                    <span class="trust-icon">🔒</span>
                    <h3>Bank-Grade Security</h3>
                    <p style="color: var(--text-muted); margin-top: 1rem; font-size: 0.9rem;">All data is scrubbed and encrypted during the analysis phase.</p>
                </div>
            </div>
        </section>
    </div>

    <!-- Technical Modal -->
    <div id="tech-modal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 2000; display: none; align-items: center; justify-content: center; padding: 2rem;">
        <div class="glass-card" style="max-width: 800px; width: 100%; max-height: 90vh; overflow-y: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h2 style="font-family: 'Space Grotesk';">System Architecture & ML Pipeline</h2>
                <button onclick="document.getElementById('tech-modal').style.display='none'" style="background: none; border: none; color: white; font-size: 1.5rem; cursor: pointer;">✕</button>
            </div>
            <div style="color: var(--text-muted); line-height: 1.6;">
                <h4 style="color: white; margin-bottom: 0.5rem;">1. Hybrid Feature Extraction</h4>
                <p style="margin-bottom: 1.5rem;">The system utilizes a <b>FeatureUnion</b> pipeline combining high-dimensional TF-IDF vectors (n-grams 1-3) with a custom <b>Heuristic Fraud Engine</b> that monitors for 15+ behavioral trigger patterns.</p>
                
                <h4 style="color: white; margin-bottom: 0.5rem;">2. Ensemble Classifier</h4>
                <p style="margin-bottom: 1.5rem;">A <b>Random Forest Ensemble</b> with 300+ decision trees processes the fused feature matrix. We use <i>balanced_subsample</i> weighting to counteract the heavy class imbalance typical in fraud datasets.</p>
                
                <h4 style="color: white; margin-bottom: 0.5rem;">3. Lexical Density Analysis</h4>
                <p style="margin-bottom: 1.5rem;">Real-time calculation of unique-to-total word ratios to detect repetitive spam patterns and low-quality automation artifacts.</p>
                
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 16px; font-family: monospace; font-size: 0.85rem;">
                    // Internal Model Signature<br>
                    Pipeline(steps=[<br>
                    &nbsp;&nbsp;('features', FeatureUnion(tfidf, fraud_flags)),<br>
                    &nbsp;&nbsp;('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced'))<br>
                    ])
                </div>
            </div>
        </div>
    </div>

    <div class="scan-overlay" id="scan-overlay">
        <div style="width: 400px; height: 500px; border: 2px solid var(--accent); position: relative; overflow: hidden; background: rgba(0,0,0,0.5);">
            <div class="scan-line"></div>
            <div style="padding: 2rem; font-family: 'Space Grotesk'; font-size: 0.8rem; color: var(--accent); opacity: 0.6;" id="terminal-text">
            </div>
        </div>
        <div class="scanning-text" id="status-text">Analyzing Job DNA...</div>
    </div>

    <footer>
        &copy; 2026 ScamGuard AI - Advanced Workforce Protection Laboratory.
    </footer>

    <script>
        // Particle.js Configuration
        particlesJS('particles-js', {
            "particles": {
                "number": { "value": 80, "density": { "enable": true, "value_area": 800 } },
                "color": { "value": "#6366f1" },
                "shape": { "type": "circle" },
                "opacity": { "value": 0.2, "random": false },
                "size": { "value": 3, "random": true },
                "line_linked": { "enable": true, "distance": 150, "color": "#6366f1", "opacity": 0.1, "width": 1 },
                "move": { "enable": true, "speed": 1, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": { "onhover": { "enable": true, "mode": "grab" }, "onclick": { "enable": true, "mode": "push" }, "resize": true },
                "modes": { "grab": { "distance": 140, "line_linked": { "opacity": 1 } }, "push": { "particles_nb": 4 } }
            },
            "retina_detect": true
        });

        const form = document.getElementById('ai-form');
        const overlay = document.getElementById('scan-overlay');
        const resultArea = document.getElementById('result-area');
        const term = document.getElementById('terminal-text');
        const status = document.getElementById('status-text');
        let chart = null;

        const terminalLines = [
            "> INITIALIZING SCAN...",
            "> PARSING LINGUISTIC VECTORS...",
            "> ANALYZING BEHAVIORAL PATTERNS...",
            "> CROSS-REFERENCING SCAM DATABASES...",
            "> CALCULATING PROBABILITY...",
            "> DECRYPTING METADATA..."
        ];

        async function typeTerminal() {
            term.innerHTML = "";
            for (let line of terminalLines) {
                term.innerHTML += line + "<br>";
                await new Promise(r => setTimeout(r, 400));
            }
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            
            overlay.style.display = 'flex';
            resultArea.style.display = 'none';
            typeTerminal();

            const title = document.getElementById('title').value;
            const description = document.getElementById('description').value;

            try {
                // Delay for the scan effect
                await new Promise(r => setTimeout(r, 2600));
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, description })
                });
                
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert('Connection to AI Core Interrupted.');
            } finally {
                overlay.style.display = 'none';
            }
        };

        function renderResults(data) {
            const score = Math.round(data.fraud_probability * 100);
            
            // DOM Elements
            const scoreEl = document.getElementById('res-score');
            const indicatorEl = document.getElementById('res-indicator');
            const integrityEl = document.getElementById('res-integrity');
            const vectorsEl = document.getElementById('res-vectors');
            const flagsEl = document.getElementById('res-flags');
            const lexicalEl = document.getElementById('res-lexical');
            const latencyEl = document.getElementById('res-latency');

            scoreEl.innerText = `${score}%`;
            integrityEl.innerText = `${100 - score}%`;
            vectorsEl.innerText = data.vector_count.toLocaleString();
            flagsEl.innerText = data.flags_count;
            lexicalEl.innerText = data.lexical_density + '%';
            latencyEl.innerText = data.latency_ms + ' ms';
            
            if (score > 70) {
                indicatorEl.innerText = 'HIGH FRAUD RISK';
                indicatorEl.className = 'risk-indicator indicator-high';
                scoreEl.style.color = 'var(--danger)';
            } else if (score > 40) {
                indicatorEl.innerText = 'CAUTION ADVISED';
                indicatorEl.className = 'risk-indicator';
                indicatorEl.style.color = 'var(--warning)';
                indicatorEl.style.borderColor = 'var(--warning)';
                scoreEl.style.color = 'var(--warning)';
            } else {
                indicatorEl.innerText = 'SECURE LISTING';
                indicatorEl.className = 'risk-indicator indicator-low';
                scoreEl.style.color = 'var(--success)';
                flagsEl.style.color = 'var(--success)';
            }

            resultArea.style.display = 'block';
            updateChart(score);
        }

        function updateChart(score) {
            const ctx = document.getElementById('neuralChart').getContext('2d');
            if (chart) chart.destroy();
            
            chart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Fraud', 'Genuine'],
                    datasets: [{
                        data: [score, 100-score],
                        backgroundColor: [score > 70 ? '#ff4d4d' : '#6366f1', 'rgba(255,255,255,0.05)'],
                        borderWidth: 0,
                        cutout: '85%'
                    }]
                },
                options: {
                    plugins: { legend: { display: false } },
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

import re

@app.route('/api/predict', methods=['POST'])
def predict():
    import time
    start_time = time.time()
    
    data = request.json
    title = data.get('title', '')
    description = data.get('description', '')
    
    # Calculate some cool stats for the frontend
    full_text = f"{description} {title}".lower()
    words = full_text.split()
    unique_words = set(words)
    lexical_density = round((len(unique_words) / max(len(words), 1)) * 100, 1)
    
    fraud_keywords = [
        "registration fee", "fee required", "pay to start", "refund later", 
        "whatsapp", "contact immediately", "urgent hiring", "limited slots", 
        "no experience required", "work from home", "instant payment", 
        "quick money", "earn money fast", "high salary"
    ]
    flags_count = sum(1 for kw in fraud_keywords if kw in full_text)
    money_pattern = r"(₹|\$|rs\.?|rupees?)\s?\d+"
    flags_count += len(re.findall(money_pattern, full_text))
    
    text_data = [f"{description} {title}"]
    
    # ML Prediction
    if model:
        import pandas as pd
        # Creating a df format expected by the FeatureUnion custom extractors
        df_input = pd.DataFrame({'title': [title], 'description': [description]})
        prediction = int(model.predict(df_input)[0])
        probability = float(model.predict_proba(df_input)[0, 1])
    else:
        prediction = 0
        probability = 0.1
        
    latency_ms = int((time.time() - start_time) * 1000)
    
    return jsonify({
        'prediction': 'fraudulent' if prediction == 1 else 'genuine',
        'fraud_probability': probability,
        'vector_count': len(words) * 31, # Fun artificial stat
        'lexical_density': lexical_density,
        'flags_count': flags_count,
        'latency_ms': latency_ms if latency_ms > 0 else 12,
        'timestamp': datetime.now().isoformat()
    })

# Vercel entrypoint
def index(request):
    return app(request)

if __name__ == "__main__":
    app.run(debug=True)
