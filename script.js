// Connect to Backend WebSocket
const socket = io('http://localhost:5000');
let currentSessionId = null;

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const chatForm = document.getElementById('chat-form');
const typingIndicator = document.getElementById('typing-indicator');

// Metrics Elements
const metrics = {
    depression: { bar: document.getElementById('bar-depression'), val: document.getElementById('val-depression') },
    anxiety: { bar: document.getElementById('bar-anxiety'), val: document.getElementById('val-anxiety') },
    bpd: { bar: document.getElementById('bar-bpd'), val: document.getElementById('val-bpd') },
    ptsd: { bar: document.getElementById('bar-ptsd'), val: document.getElementById('val-ptsd') }
};

// ==========================================
// Initialization
// ==========================================

socket.on('connect', () => {
    console.log('Connected to server');
    if (!currentSessionId) createSession();
});

socket.on('connect_error', (error) => {
    console.error('Connection Error:', error);
    addSystemMessage("⚠️ Connection error. Please ensure the backend server is running on port 5000.");
});

async function createSession() {
    try {
        const response = await fetch('http://localhost:5000/api/session/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'user_' + Math.floor(Math.random() * 1000) })
        });
        const data = await response.json();
        currentSessionId = data.session_id;
        
        socket.emit('join_session', { session_id: currentSessionId });
        console.log('Joined session:', currentSessionId);
    } catch (error) {
        console.error('Failed to create session:', error);
        addSystemMessage("⚠️ Failed to connect to backend server. Is it running?");
    }
}

// ==========================================
// Chat Logic
// ==========================================

chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text) return;

    // Add user message
    addMessage(text, 'user');
    
    // Clear input
    messageInput.value = '';
    
    // Send to server
    if (currentSessionId) {
        socket.emit('send_message', {
            session_id: currentSessionId,
            message: text
        });
        showTyping();
    }
});

// Allow Enter key to submit
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

// ==========================================
// Socket Events
// ==========================================

socket.on('message_analyzed', (data) => {
    hideTyping();
    
    // Provide a subtle AI response (simulated for now, or based on analysis)
    const analysis = data.analysis;
    const responseText = generateEmpathyResponse(analysis);
    
    setTimeout(() => {
        addMessage(responseText, 'system');
    }, 600);
    
    // Update Dashboard
    updateMetrics(analysis.predictions);
});

socket.on('crisis_alert', (data) => {
    // Show Crisis Modal
    document.getElementById('crisis-modal').classList.remove('hidden');
    addSystemMessage(`🚨 ${data.message}`);
});

// ==========================================
// UI Functions
// ==========================================

function addMessage(text, type) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${text}</p>`;
    
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    msgDiv.appendChild(content);
    msgDiv.appendChild(meta);
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addSystemMessage(text) {
    addMessage(text, 'system');
}

function showTyping() {
    typingIndicator.classList.remove('hidden');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideTyping() {
    typingIndicator.classList.add('hidden');
}

function updateMetrics(predictions) {
    // Update bars and text
    updateMetric('depression', predictions.depression);
    updateMetric('anxiety', predictions.anxiety);
    updateMetric('bpd', predictions.bpd);
    updateMetric('ptsd', predictions.ptsd);
    
    // Update chart if we implemented one
    updateChart(predictions);
}

function updateMetric(key, value) {
    const percentage = Math.round(value * 100);
    metrics[key].val.textContent = `${percentage}%`;
    metrics[key].bar.style.width = `${percentage}%`;
    
    // Color coding based on severity
    if (value > 0.7) metrics[key].bar.style.background = '#ef4444'; // Red
    else if (value > 0.4) metrics[key].bar.style.background = '#eab308'; // Yellow
}

function generateEmpathyResponse(analysis) {
    // Simple logic to vary responses based on dominant emotion
    const preds = analysis.predictions;
    const dominant = Object.keys(preds).reduce((a, b) => preds[a] > preds[b] ? a : b);
    
    if (analysis.crisis_level === 'HIGH' || analysis.crisis_level === 'CRITICAL') {
        return "I'm detecting that you're in significant distress. Please know that help is available. Would you like to see some resources?";
    }
    
    const responses = {
        depression: [
            "I hear you. It sounds heavy.",
            "That sounds really difficult to carry alone.",
            "I'm listening. Tell me more about how you're feeling."
        ],
        anxiety: [
            "It sounds like there's a lot on your mind right now.",
            "Take a deep breath. I'm here with you.",
            "That sounds overwhelming."
        ],
        default: [
            "I understand.",
            "Thank you for sharing that with me.",
            "I'm here to listen."
        ]
    };
    
    const list = responses[dominant] || responses.default;
    return list[Math.floor(Math.random() * list.length)];
}

// ==========================================
// Chart.js Visualization
// ==========================================
const ctx = document.getElementById('emotionChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'radar',
    data: {
        labels: ['Depression', 'Anxiety', 'BPD', 'PTSD'],
        datasets: [{
            label: 'Current State',
            data: [0, 0, 0, 0],
            backgroundColor: 'rgba(99, 102, 241, 0.2)',
            borderColor: 'rgba(99, 102, 241, 1)',
            borderWidth: 2
        }]
    },
    options: {
        scales: {
            r: {
                beginAtZero: true,
                max: 1,
                grid: { color: 'rgba(255,255,255,0.1)' },
                pointLabels: { color: '#94a3b8' },
                ticks: { display: false }
            }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

function updateChart(predictions) {
    chart.data.datasets[0].data = [
        predictions.depression,
        predictions.anxiety,
        predictions.bpd,
        predictions.ptsd
    ];
    chart.update();
}

// New Session Button
document.getElementById('new-session-btn').addEventListener('click', () => {
    chatContainer.innerHTML = '';
    addSystemMessage("Started a new session. How can I help you?");
    createSession();
});


// ==========================================
// Tab Switching & Data Loading
// ==========================================

const navBtns = document.querySelectorAll('.nav-btn');
const views = document.querySelectorAll('.view');
let detailedChart = null;

navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Update nav active state
        navBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show target view
        const tabName = btn.dataset.tab;
        views.forEach(view => {
            view.classList.remove('active');
            if (view.id === `${tabName}-view`) {
                view.classList.add('active');
            }
        });

        // Load specific tab data
        if (tabName === 'analytics') loadAnalytics();
        if (tabName === 'resources') loadResources();
    });
});

async function loadAnalytics() {
    if (!currentSessionId) return;
    
    try {
        const response = await fetch(`http://localhost:5000/api/session/${currentSessionId}/summary`);
        const data = await response.json();
        
        // Update Stats
        document.getElementById('stat-duration').textContent = `${data.duration_minutes} min`;
        document.getElementById('stat-messages').textContent = data.message_count;
        document.getElementById('stat-dominant').textContent = data.dominant_condition.charAt(0).toUpperCase() + data.dominant_condition.slice(1);
        document.getElementById('stat-alerts').textContent = data.crisis_alerts;
        
        // Init/Update Detailed Chart
        renderDetailedChart(data.average_scores);
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

async function loadResources() {
    const container = document.getElementById('resources-container');
    container.innerHTML = '<div class="loading-resources">Loading resources...</div>';
    
    try {
        const response = await fetch('http://localhost:5000/api/resources');
        const data = await response.json();
        const resources = data.resources;
        
        container.innerHTML = '';
        
        // Loop through regions (US, UK, Int)
        for (const [region, info] of Object.entries(resources)) {
            const card = document.createElement('div');
            card.className = 'resource-card';
            
            card.innerHTML = `
                <h3><i class="fa-solid fa-earth-americas"></i> ${region} Resources</h3>
                <p>Support available in your region</p>
                <div class="resource-links">
                    ${info.hotline ? `
                        <a href="tel:${info.hotline}" class="resource-link">
                            <i class="fa-solid fa-phone"></i>
                            <div>
                                <strong>Call ${info.hotline}</strong>
                                <span>${info.name || 'Emergency Hotline'}</span>
                            </div>
                        </a>
                    ` : ''}
                    
                    ${info.text ? `
                        <a href="#" class="resource-link">
                            <i class="fa-solid fa-comment-sms"></i>
                            <div>
                                <strong>${info.text}</strong>
                                <span>Text Support</span>
                            </div>
                        </a>
                    ` : ''}
                    
                    ${info.website ? `
                        <a href="${info.website}" target="_blank" class="resource-link">
                            <i class="fa-solid fa-globe"></i>
                            <div>
                                <strong>Visit Website</strong>
                                <span>Official Support Site</span>
                            </div>
                        </a>
                    ` : ''}
                </div>
            `;
            
            container.appendChild(card);
        }
    } catch (error) {
        container.innerHTML = '<p>Failed to load resources.</p>';
        console.error(error);
    }
}

function renderDetailedChart(scores) {
    const ctx = document.getElementById('detailedChart').getContext('2d');
    
    if (detailedChart) {
        detailedChart.destroy();
    }
    
    detailedChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Depression', 'Anxiety', 'BPD', 'PTSD'],
            datasets: [{
                label: 'Average Session Score',
                data: [scores.depression, scores.anxiety, scores.bpd, scores.ptsd],
                backgroundColor: [
                    '#6366f1',
                    '#f472b6',
                    '#38bdf8',
                    '#fb923c'
                ],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
