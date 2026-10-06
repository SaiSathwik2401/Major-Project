import re
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class CrisisDetectionSystem:
    """
    Real-time crisis detection and intervention system
    Features:
    1. Suicidal ideation detection
    2. Self-harm language detection
    3. Immediate crisis intervention
    4. Emergency contact notification
    5. Resource recommendation
    6. Risk level classification (Low/Medium/High/Critical)
    """
    
    def __init__(self):
        # Crisis keywords categorized by severity
        self.critical_keywords = [
            'kill myself', 'end my life', 'suicide', 'want to die',
            'better off dead', 'no reason to live', 'goodbye world',
            'final message', 'can\'t go on', 'ending it all'
        ]
        
        self.high_risk_keywords = [
            'self harm', 'cut myself', 'hurt myself', 'plan to',
            'overdose', 'jump off', 'pills', 'don\'t want to wake up',
            'burden to everyone', 'world without me'
        ]
        
        self.medium_risk_keywords = [
            'hopeless', 'worthless', 'give up', 'can\'t take it',
            'too much pain', 'no point', 'never get better',
            'why bother', 'nothing matters', 'tired of living'
        ]
        
        # Crisis resources
        self.crisis_resources = {
            'US': {
                'hotline': '988',
                'name': 'Suicide & Crisis Lifeline',
                'text': 'Text HOME to 741741',
                'website': 'https://988lifeline.org/'
            },
            'UK': {
                'hotline': '116 123',
                'name': 'Samaritans',
                'email': 'jo@samaritans.org',
                'website': 'https://www.samaritans.org/'
            },
            'International': {
                'website': 'https://www.opencounseling.com/suicide-hotlines',
                'name': 'International Suicide Hotlines'
            }
        }
    
    def detect_crisis_level(self, text):
        """
        Detect crisis level from text
        Returns: (risk_level, matched_keywords, confidence)
        """
        text_lower = text.lower()
        matched_critical = []
        matched_high = []
        matched_medium = []
        
        # Check for critical keywords
        for keyword in self.critical_keywords:
            if keyword in text_lower:
                matched_critical.append(keyword)
        
        # Check for high-risk keywords
        for keyword in self.high_risk_keywords:
            if keyword in text_lower:
                matched_high.append(keyword)
        
        # Check for medium-risk keywords
        for keyword in self.medium_risk_keywords:
            if keyword in text_lower:
                matched_medium.append(keyword)
        
        # Determine risk level
        if matched_critical:
            return 'CRITICAL', matched_critical, 1.0
        elif len(matched_high) >= 2:
            return 'HIGH', matched_high, 0.85
        elif matched_high:
            return 'HIGH', matched_high, 0.75
        elif len(matched_medium) >= 3:
            return 'MEDIUM', matched_medium, 0.6
        elif matched_medium:
            return 'LOW', matched_medium, 0.3
        else:
            return 'NONE', [], 0.0
    
    def analyze_urgency_indicators(self, text):
        """
        Analyze specific urgency indicators
        """
        indicators = {
            'has_plan': False,
            'has_timeline': False,
            'has_means': False,
            'expresses_isolation': False,
            'says_goodbye': False
        }
        
        text_lower = text.lower()
        
        # Check for plan
        plan_phrases = ['plan to', 'going to', 'will', 'tonight', 'tomorrow']
        if any(phrase in text_lower for phrase in plan_phrases):
            indicators['has_plan'] = True
        
        # Check for timeline
        time_phrases = ['today', 'tonight', 'soon', 'this week', 'tomorrow']
        if any(phrase in text_lower for phrase in time_phrases):
            indicators['has_timeline'] = True
        
        # Check for means
        means_phrases = ['pills', 'gun', 'rope', 'jump', 'overdose']
        if any(phrase in text_lower for phrase in means_phrases):
            indicators['has_means'] = True
        
        # Check for isolation
        isolation_phrases = ['nobody cares', 'no one would miss', 'all alone', 'no friends']
        if any(phrase in text_lower for phrase in isolation_phrases):
            indicators['expresses_isolation'] = True
        
        # Check for goodbye messages
        goodbye_phrases = ['goodbye', 'farewell', 'last message', 'final words']
        if any(phrase in text_lower for phrase in goodbye_phrases):
            indicators['says_goodbye'] = True
        
        urgency_score = sum(indicators.values()) / len(indicators)
        
        return indicators, urgency_score
    
    def generate_crisis_response(self, risk_level, country='US'):
        """
        Generate appropriate crisis response
        """
        resources = self.crisis_resources.get(country, self.crisis_resources['US'])
        
        if risk_level == 'CRITICAL':
            message = f"""
🚨 IMMEDIATE CRISIS SUPPORT NEEDED 🚨

You are not alone, and help is available RIGHT NOW.

📞 CALL: {resources.get('hotline', '911')} - {resources.get('name', 'Crisis Hotline')}
💬 TEXT: {resources.get('text', 'Available')}
🌐 VISIT: {resources.get('website', 'N/A')}

If you are in immediate danger:
• Call emergency services (911 in US)
• Go to nearest emergency room
• Call a trusted friend or family member

Your life matters. This crisis can pass. Please reach out.
            """
        
        elif risk_level == 'HIGH':
            message = f"""
⚠️ URGENT SUPPORT RECOMMENDED ⚠️

It sounds like you're going through a very difficult time.

Please consider reaching out:
📞 {resources.get('hotline', 'Crisis Line')}: {resources.get('name', 'Support Available')}
💬 {resources.get('text', 'Text support available')}

You don't have to face this alone. Professional help is available 24/7.
            """
        
        elif risk_level == 'MEDIUM':
            message = f"""
💙 SUPPORT AVAILABLE 💙

It seems you may be struggling. Please know that:

• You are not alone in feeling this way
• These feelings can improve with support
• Help is available whenever you need it

Resources:
📞 {resources.get('hotline', 'Support Line')}
🌐 {resources.get('website', 'Online Resources')}

Consider talking to a mental health professional.
            """
        
        else:
            message = """
💚 SUPPORT RESOURCES 💚

If you're feeling down or need someone to talk to:

• Mental health professionals can help
• Support groups are available
• Crisis lines are available 24/7

Remember: Taking care of your mental health is important.
            """
        
        return message
    
    def send_emergency_notification(self, user_id, text_snippet, risk_level, 
                                   emergency_contact_email):
        """
        Send emergency notification to designated contact
        """
        # THIS IS A DEMONSTRATION - Configure with real SMTP settings
        try:
            msg = MIMEMultipart()
            msg['From'] = 'crisis.system@mentalhealth.app'
            msg['To'] = emergency_contact_email
            msg['Subject'] = f'URGENT: Crisis Alert for User {user_id}'
            
            body = f"""
CRISIS ALERT - {risk_level} RISK

User ID: {user_id}
Timestamp: {datetime.now()}
Risk Level: {risk_level}

Text Analysis Detected Crisis Indicators:
{text_snippet[:200]}...

RECOMMENDED ACTIONS:
1. Contact user immediately
2. Check on their immediate safety
3. Encourage professional help
4. Call emergency services if needed

This is an automated alert from the Mental Health Monitoring System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: Configure SMTP server details
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login('your_email', 'your_password')
            # server.send_message(msg)
            # server.quit()
            
            print(f"[SIMULATION] Emergency notification sent to {emergency_contact_email}")
            return True
        
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False
    
    def create_safety_plan(self, user_id):
        """
        Generate personalized safety plan template
        """
        safety_plan = f"""
╔══════════════════════════════════════════════════════════╗
║              PERSONAL SAFETY PLAN                        ║
║              User ID: {user_id}                          ║
║              Created: {datetime.now().date()}            ║
╚══════════════════════════════════════════════════════════╝

1. WARNING SIGNS (When I need to use this plan):
   • ________________________________________________
   • ________________________________________________
   • ________________________________________________

2. INTERNAL COPING STRATEGIES (Things I can do alone):
   • ________________________________________________
   • ________________________________________________
   • ________________________________________________

3. SOCIAL DISTRACTIONS (People/places that help):
   • ________________________________________________
   • ________________________________________________
   • ________________________________________________

4. PEOPLE I CAN REACH OUT TO:
   Name: _________________ Phone: _________________
   Name: _________________ Phone: _________________

5. PROFESSIONAL CONTACTS:
   Therapist: ____________ Phone: _________________
   Crisis Line: 988 (US) / 116 123 (UK)

6. MAKING MY ENVIRONMENT SAFE:
   • Remove or secure items I could use to harm myself
   • ________________________________________________

7. REASONS FOR LIVING:
   • ________________________________________________
   • ________________________________________________
   • ________________________________________________

╔══════════════════════════════════════════════════════════╗
║  If in crisis: Call 988 (US) or go to nearest ER       ║
╚══════════════════════════════════════════════════════════╝
        """
        
        return safety_plan
    
    def comprehensive_crisis_assessment(self, text, user_id=None, 
                                       emergency_contact=None):
        """
        Complete crisis assessment and response
        """
        print("\n" + "="*70)
        print("CRISIS DETECTION ANALYSIS")
        print("="*70)
        
        # 1. Detect crisis level
        risk_level, keywords, confidence = self.detect_crisis_level(text)
        
        print(f"\nRisk Level: {risk_level}")
        print(f"Confidence: {confidence:.2%}")
        
        if keywords:
            print(f"\nDetected Crisis Indicators:")
            for keyword in keywords:
                print(f"  • {keyword}")
        
        # 2. Analyze urgency
        indicators, urgency_score = self.analyze_urgency_indicators(text)
        
        print(f"\nUrgency Score: {urgency_score:.2%}")
        print("\nUrgency Indicators:")
        for indicator, present in indicators.items():
            status = "✓ DETECTED" if present else "✗ Not detected"
            print(f"  {indicator.replace('_', ' ').title()}: {status}")
        
        # 3. Generate response
        response = self.generate_crisis_response(risk_level)
        
        print("\n" + "-"*70)
        print("RECOMMENDED RESPONSE:")
        print("-"*70)
        print(response)
        
        # 4. Take action for high-risk cases
        if risk_level in ['CRITICAL', 'HIGH']:
            print("\n" + "="*70)
            print("EMERGENCY ACTIONS INITIATED")
            print("="*70)
            
            # Send notification
            if emergency_contact:
                self.send_emergency_notification(
                    user_id, text, risk_level, emergency_contact
                )
            
            # Generate safety plan
            safety_plan = self.create_safety_plan(user_id or 'ANONYMOUS')
            
            # Save safety plan
            with open(f'safety_plan_{user_id or "anonymous"}.txt', 'w') as f:
                f.write(safety_plan)
            
            print("\n✅ Safety plan generated and saved")
            print("✅ Emergency protocols activated")
        
        print("\n" + "="*70)
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'keywords': keywords,
            'urgency_score': urgency_score,
            'indicators': indicators,
            'response': response
        }


# Example usage
if __name__ == "__main__":
    crisis_system = CrisisDetectionSystem()
    
    # Test with various severity levels
    test_cases = [
        {
            'text': "I can't take this anymore. I have a plan to end my life tonight.",
            'description': 'CRITICAL CASE'
        },
        {
            'text': "I've been thinking about hurting myself. I have pills saved up.",
            'description': 'HIGH RISK CASE'
        },
        {
            'text': "Everything is hopeless and worthless. I don't see the point anymore.",
            'description': 'MEDIUM RISK CASE'
        },
        {
            'text': "I'm feeling a bit down today but managing.",
            'description': 'LOW RISK CASE'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*70}")
        print(f"TEST CASE {i}: {case['description']}")
        print(f"{'#'*70}")
        
        result = crisis_system.comprehensive_crisis_assessment(
            text=case['text'],
            user_id=f'test_user_{i}',
            emergency_contact='emergency@example.com'
        )
    
    print("\n\n✅ Crisis detection system implementation complete!")