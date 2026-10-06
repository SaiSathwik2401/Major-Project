from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
from datetime import datetime
import uuid
import numpy as np

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage (in production, use database)
active_sessions = {}
chat_histories = {}

class MentalHealthChatAPI:
    """
    Real-time chat integration with mental health monitoring
    Features:
    1. REST API endpoints
    2. WebSocket for real-time communication
    3. Continuous monitoring during chat
    4. Session management
    5. Chat history storage
    6. Real-time risk alerts
    """
    
    def __init__(self, model, classifier, crisis_system):
        self.model = model
        self.classifier = classifier
        self.crisis_system = crisis_system
        
        # Heuristic keywords for demo purposes (since model weights are missing)
        self.keywords = {
            'depression': ['sad', 'empty', 'hopeless', 'tired', 'worthless', 'cry', 'sleep', 'pain', 'lonely', 'dark', 'give up'],
            'anxiety': ['worry', 'anxious', 'scared', 'nervous', 'panic', 'stress', 'fear', 'racing', 'heart', 'trembling'],
            'bpd': ['hate', 'love', 'abandon', 'unstable', 'mood', 'intense', 'impulsive', 'empty', 'anger', 'relationship'],
            'ptsd': ['flashback', 'trauma', 'nightmare', 'reliving', 'avoid', 'trigger', 'memory', 'event', 'past', 'scared']
        }

    def _calculate_heuristic_scores(self, text):
        """Calculate scores based on keywords when model is not available"""
        text = text.lower()
        scores = {k: 0.0 for k in self.keywords}
        
        # Count keyword matches
        total_hits = 0
        for category, words in self.keywords.items():
            for word in words:
                if word in text:
                    scores[category] += 1.0
                    total_hits += 1
        
        # Add small random noise for variety and baseline
        for k in scores:
            scores[k] += np.random.uniform(0.1, 0.3)
            
        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            for k in scores:
                scores[k] /= total_score
        
        return scores

    def analyze_message(self, message, session_id):
        """
        Analyze individual chat message
        """
        # Calculate scores (using heuristic since model is None)
        scores = self._calculate_heuristic_scores(message)
        
        # Map to array for consistency with original structure
        prediction = [
            scores['depression'],
            scores['anxiety'],
            scores['bpd'],
            scores['ptsd']
        ]
        
        # Get crisis level
        risk_level, keywords, confidence = self.crisis_system.detect_crisis_level(message)
        
        result = {
            'message_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'depression': float(prediction[0]),
                'anxiety': float(prediction[1]),
                'bpd': float(prediction[2]),
                'ptsd': float(prediction[3])
            },
            'predicted_class': int(np.argmax(prediction)),
            'confidence': float(np.max(prediction)),
            'crisis_detection': {
                'risk_level': risk_level,
                'confidence': confidence,
                'keywords': keywords
            }
        }
        
        # Store in session history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({
            'message': message,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result

# Linked Modules
from hybrid_classifier import HybridMentalHealthClassifier
from crisis_detection import CrisisDetectionSystem

# Initialize API
classifier = HybridMentalHealthClassifier() 
model = None 

crisis_system = CrisisDetectionSystem()
api = MentalHealthChatAPI(model, classifier, crisis_system)


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get mental health resources"""
    return jsonify({
        'resources': crisis_system.crisis_resources
    })


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create new chat session"""
    data = request.json
    session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = {
        'user_id': data.get('user_id'),
        'created_at': datetime.now().isoformat(),
        'metadata': data.get('metadata', {}),
        'message_count': 0
    }
    
    return jsonify({
        'session_id': session_id,
        'created_at': active_sessions[session_id]['created_at']
    }), 201


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyze single text input"""
    data = request.json
    text = data.get('text')
    session_id = data.get('session_id')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Use API class to analyze (consistent logic)
    result = api.analyze_message(text, session_id if session_id else 'temp')
    
    return jsonify(result), 200


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts at once"""
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'Texts array is required'}), 400
    
    results = []
    for text in texts:
        scores = api._calculate_heuristic_scores(text)
        result = {
            'text': text,
            'predictions': scores
        }
        results.append(result)
    
    return jsonify({
        'count': len(results),
        'results': results
    }), 200


@app.route('/api/session/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get chat history for session"""
    if session_id not in chat_histories:
        return jsonify({'error': 'Session not found'}), 404
    
    history = chat_histories[session_id]
    
    return jsonify({
        'session_id': session_id,
        'message_count': len(history),
        'history': history
    }), 200


@app.route('/api/session/<session_id>/summary', methods=['GET'])
def get_session_summary(session_id):
    """Get summary statistics for session"""
    if session_id not in chat_histories:
        return jsonify({'error': 'Session not found'}), 404
    
    history = chat_histories[session_id]
    
    if not history:
        return jsonify({
            'session_id': session_id,
            'message_count': 0,
            'average_scores': {'depression': 0, 'anxiety': 0, 'bpd': 0, 'ptsd': 0},
            'crisis_alerts': 0
        }), 200
    
    # Calculate summary statistics
    all_predictions = [msg['analysis']['predictions'] for msg in history]
    
    avg_depression = np.mean([p['depression'] for p in all_predictions])
    avg_anxiety = np.mean([p['anxiety'] for p in all_predictions])
    avg_bpd = np.mean([p['bpd'] for p in all_predictions])
    avg_ptsd = np.mean([p['ptsd'] for p in all_predictions])
    
    crisis_levels = [msg['analysis']['crisis_detection']['risk_level'] 
                    for msg in history]
    
    summary = {
        'session_id': session_id,
        'duration_minutes': 45,  # Mock
        'message_count': len(history),
        'average_scores': {
            'depression': float(avg_depression),
            'anxiety': float(avg_anxiety),
            'bpd': float(avg_bpd),
            'ptsd': float(avg_ptsd)
        },
        'crisis_alerts': crisis_levels.count('HIGH') + crisis_levels.count('CRITICAL'),
        'dominant_condition': max(['depression', 'anxiety', 'bpd', 'ptsd'], 
                                key=lambda k: {'depression': avg_depression, 'anxiety': avg_anxiety, 'bpd': avg_bpd, 'ptsd': avg_ptsd}[k]),
        'created_at': active_sessions[session_id]['created_at']
    }
    
    return jsonify(summary), 200


# ============================================================================
# WEBSOCKET EVENTS (Real-time Communication)
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connection_established', {
        'client_id': request.sid,
        'timestamp': datetime.now().isoformat()
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')


@socketio.on('join_session')
def handle_join_session(data):
    """Client joins a session"""
    session_id = data.get('session_id')
    print(f'Client {request.sid} joined session {session_id}')
    
    emit('session_joined', {
        'session_id': session_id,
        'status': 'active'
    })


@socketio.on('send_message')
def handle_message(data):
    """
    Handle incoming chat message
    Analyzes message and sends real-time feedback
    """
    session_id = data.get('session_id')
    message = data.get('message')
    
    print(f'Received message in session {session_id}: {message}')
    
    # Use standard analysis method
    analysis = api.analyze_message(message, session_id)
    
    # Send analysis back to client
    emit('message_analyzed', {
        'original_message': message,
        'analysis': analysis
    })
    
    # If crisis detected, send alert
    if analysis['crisis_detection']['risk_level'] in ['HIGH', 'CRITICAL']:
        emit('crisis_alert', {
            'level': analysis['crisis_detection']['risk_level'],
            'message': 'Immediate support recommended',
            'resources': crisis_system.crisis_resources
        })


@socketio.on('typing')
def handle_typing(data):
    """Handle typing indicator"""
    session_id = data.get('session_id')
    emit('user_typing', {
        'session_id': session_id
    }, broadcast=True, include_self=False)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MENTAL HEALTH CHAT API SERVER")
    print("="*70)
    print("\nAvailable Endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/resources")
    print("  POST /api/analyze")
    print("  POST /api/batch-analyze")
    print("  GET  /api/session/<id>/summary")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)