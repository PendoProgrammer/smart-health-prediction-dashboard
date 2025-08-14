import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Health Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high { background-color: #ff4444; color: white; }
    .risk-medium { background-color: #ffaa00; color: white; }
    .risk-low { background-color: #44ff44; color: black; }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class HealthDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.generate_sample_data()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'name': 'John Smith',
                'age': 35,
                'gender': 'Male',
                'height': 175,  # cm
                'weight': 75,   # kg
                'blood_type': 'O+'
            }
        
        if 'current_metrics' not in st.session_state:
            st.session_state.current_metrics = {
                'heart_rate': 72,
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'temperature': 36.7,  # Celsius
                'blood_sugar': 95,
                'oxygen_saturation': 98,
                'daily_steps': 8500,
                'sleep_hours': 7.2,
                'hydration_glasses': 6
            }
    
    def generate_sample_data(self):
        """Generate sample historical data"""
        dates = pd.date_range(end=datetime.date.today(), periods=7)
        
        # Create realistic variations in health metrics
        np.random.seed(42)
        
        self.historical_data = pd.DataFrame({
            'date': dates,
            'heart_rate': np.random.normal(72, 3, 7).astype(int),
            'systolic_bp': np.random.normal(120, 5, 7).astype(int),
            'diastolic_bp': np.random.normal(80, 3, 7).astype(int),
            'blood_sugar': np.random.normal(95, 8, 7).astype(int),
            'weight': np.random.normal(75, 0.5, 7).round(1),
            'daily_steps': np.random.normal(8500, 1000, 7).astype(int),
            'sleep_hours': np.random.normal(7.2, 0.5, 7).round(1),
            'hydration_glasses': np.random.randint(5, 9, 7)
        })
    
    def calculate_bmi(self):
        """Calculate BMI"""
        profile = st.session_state.user_profile
        height_m = profile['height'] / 100
        return round(profile['weight'] / (height_m ** 2), 1)
    
    def calculate_risk_scores(self):
        """Calculate health risk scores using simple ML logic"""
        profile = st.session_state.user_profile
        metrics = st.session_state.current_metrics
        
        bmi = self.calculate_bmi()
        age = profile['age']
        
        # Simplified risk calculations (in real implementation, use trained ML models)
        cardiovascular_risk = min(100, max(0, 
            (age * 0.8) + 
            (20 if bmi > 25 else 0) + 
            (15 if metrics['systolic_bp'] > 130 else 0) +
            (10 if metrics['heart_rate'] > 80 else 0)
        ))
        
        diabetes_risk = min(100, max(0,
            (age * 0.6) + 
            (25 if bmi > 30 else 0) + 
            (20 if metrics['blood_sugar'] > 100 else 0) +
            (15 if profile['gender'] == 'Male' and profile['age'] > 45 else 0)
        ))
        
        hypertension_risk = min(100, max(0,
            (30 if metrics['systolic_bp'] > 120 else 0) + 
            (age * 0.7) +
            (15 if bmi > 28 else 0)
        ))
        
        return {
            'cardiovascular': round(cardiovascular_risk),
            'diabetes': round(diabetes_risk),
            'hypertension': round(hypertension_risk)
        }
    
    def generate_predictions(self):
        """Generate 30-day health predictions"""
        # Simple linear regression for trend prediction
        X = np.arange(7).reshape(-1, 1)
        
        # Heart rate prediction
        hr_model = LinearRegression()
        hr_model.fit(X, self.historical_data['heart_rate'].values)
        future_hr = hr_model.predict([[30]])[0]
        
        # Weight prediction
        weight_model = LinearRegression()
        weight_model.fit(X, self.historical_data['weight'].values)
        future_weight = weight_model.predict([[30]])[0]
        
        # Health score calculation (0-100)
        current_health_score = self.calculate_health_score()
        predicted_improvement = 5 if future_hr < 70 and future_weight < 74 else 2
        
        return {
            'heart_rate': round(future_hr),
            'weight': round(future_weight, 1),
            'health_score': min(100, current_health_score + predicted_improvement)
        }
    
    def calculate_health_score(self):
        """Calculate overall health score"""
        metrics = st.session_state.current_metrics
        profile = st.session_state.user_profile
        
        score = 100
        
        # Heart rate (ideal: 60-80)
        if metrics['heart_rate'] < 60 or metrics['heart_rate'] > 80:
            score -= 10
        
        # Blood pressure (ideal: <120/80)
        if metrics['systolic_bp'] > 130 or metrics['diastolic_bp'] > 85:
            score -= 15
        
        # BMI (ideal: 18.5-25)
        bmi = self.calculate_bmi()
        if bmi > 25:
            score -= 10
        
        # Steps (ideal: >10000)
        if metrics['daily_steps'] < 10000:
            score -= 5
        
        # Sleep (ideal: 7-9 hours)
        if metrics['sleep_hours'] < 7 or metrics['sleep_hours'] > 9:
            score -= 5
        
        # Hydration (ideal: >8 glasses)
        if metrics['hydration_glasses'] < 8:
            score -= 5
        
        return max(0, score)
    
    def generate_recommendations(self):
        """Generate personalized recommendations"""
        metrics = st.session_state.current_metrics
        recommendations = []
        
        if metrics['daily_steps'] < 10000:
            recommendations.append({
                'category': 'Exercise',
                'message': 'Increase daily steps to 10,000+ for better cardiovascular health',
                'priority': 'High',
                'icon': '🚶'
            })
        
        if metrics['sleep_hours'] < 7:
            recommendations.append({
                'category': 'Sleep',
                'message': 'Aim for 7-9 hours of sleep for optimal recovery',
                'priority': 'High',
                'icon': '😴'
            })
        
        if metrics['hydration_glasses'] < 8:
            recommendations.append({
                'category': 'Hydration',
                'message': 'Drink more water - aim for 8+ glasses daily',
                'priority': 'Medium',
                'icon': '💧'
            })
        
        if metrics['systolic_bp'] > 130:
            recommendations.append({
                'category': 'Blood Pressure',
                'message': 'Monitor blood pressure closely and consider dietary changes',
                'priority': 'High',
                'icon': '❤️'
            })
        
        if self.calculate_bmi() > 25:
            recommendations.append({
                'category': 'Weight Management',
                'message': 'Consider a balanced diet and regular exercise for weight management',
                'priority': 'Medium',
                'icon': '⚖️'
            })
        
        return recommendations
    
    def create_metric_card(self, title, value, unit, delta=None):
        """Create a metric display card"""
        if delta:
            st.metric(label=title, value=f"{value} {unit}", delta=delta)
        else:
            st.metric(label=title, value=f"{value} {unit}")
    
    def create_risk_gauge(self, title, score, color):
        """Create a risk assessment gauge"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 30},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        return fig

def main():
    dashboard = HealthDashboard()
    
    # Sidebar for navigation and user profile
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        
        profile = st.session_state.user_profile
        name = st.text_input("Name", value=profile['name'])
        age = st.number_input("Age", min_value=1, max_value=120, value=profile['age'])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                             index=0 if profile['gender'] == 'Male' else 1)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=profile['height'])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=profile['weight'])
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                 index=6 if profile['blood_type'] == 'O+' else 0)
        
        # Update profile
        st.session_state.user_profile.update({
            'name': name, 'age': age, 'gender': gender,
            'height': height, 'weight': weight, 'blood_type': blood_type
        })
        
        st.markdown("---")
        st.markdown("## 📊 Navigation")
        page = st.radio("Go to", ["Overview", "Health Trends", "AI Predictions", "Recommendations"])
    
    # Main header
    st.markdown(f"<h1 class='main-header'>🏥 Smart Health Dashboard - {name}</h1>", 
                unsafe_allow_html=True)
    
    # Overview Page
    if page == "Overview":
        st.markdown("## 📈 Current Health Metrics")
        
        metrics = st.session_state.current_metrics
        
        # Create metric cards in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dashboard.create_metric_card("Heart Rate", metrics['heart_rate'], "bpm", "-2")
        with col2:
            dashboard.create_metric_card("Blood Pressure", 
                                       f"{metrics['systolic_bp']}/{metrics['diastolic_bp']}", "mmHg", "+1")
        with col3:
            dashboard.create_metric_card("Temperature", metrics['temperature'], "°C")
        with col4:
            dashboard.create_metric_card("Blood Sugar", metrics['blood_sugar'], "mg/dL", "-3")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            dashboard.create_metric_card("Oxygen Sat.", metrics['oxygen_saturation'], "%")
        with col6:
            dashboard.create_metric_card("Daily Steps", f"{metrics['daily_steps']:,}", "steps", "+5%")
        with col7:
            dashboard.create_metric_card("Sleep", metrics['sleep_hours'], "hours")
        with col8:
            dashboard.create_metric_card("Hydration", metrics['hydration_glasses'], "glasses")
        
        # Health Summary
        st.markdown("## 📋 Health Summary")
        
        bmi = dashboard.calculate_bmi()
        health_score = dashboard.calculate_health_score()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### BMI")
            st.metric("Body Mass Index", bmi, help="Normal: 18.5-24.9")
            if bmi < 18.5:
                st.warning("Underweight")
            elif bmi < 25:
                st.success("Normal weight")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
        
        with col2:
            st.markdown("### Health Score")
            st.metric("Overall Health", f"{health_score}/100")
            if health_score >= 80:
                st.success("Excellent health!")
            elif health_score >= 60:
                st.warning("Good health")
            else:
                st.error("Needs attention")
        
        with col3:
            st.markdown("### Status")
            if health_score >= 80:
                st.success("🟢 All systems normal")
            elif health_score >= 60:
                st.warning("🟡 Some areas need attention")
            else:
                st.error("🔴 Health risks detected")
    
    # Health Trends Page
    elif page == "Health Trends":
        st.markdown("## 📊 7-Day Health Trends")
        
        # Heart Rate Trend
        st.markdown("### Heart Rate Trend")
        fig_hr = px.line(dashboard.historical_data, x='date', y='heart_rate',
                        title='Heart Rate Over Time',
                        labels={'heart_rate': 'Heart Rate (bpm)', 'date': 'Date'})
        fig_hr.update_traces(line_color='red')
        st.plotly_chart(fig_hr, use_container_width=True)
        
        # Blood Pressure and Blood Sugar
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Blood Pressure Trend")
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(x=dashboard.historical_data['date'], 
                                      y=dashboard.historical_data['systolic_bp'],
                                      mode='lines+markers', name='Systolic', line=dict(color='blue')))
            fig_bp.add_trace(go.Scatter(x=dashboard.historical_data['date'], 
                                      y=dashboard.historical_data['diastolic_bp'],
                                      mode='lines+markers', name='Diastolic', line=dict(color='green')))
            fig_bp.update_layout(title='Blood Pressure Trend', xaxis_title='Date', yaxis_title='mmHg')
            st.plotly_chart(fig_bp, use_container_width=True)
        
        with col2:
            st.markdown("### Blood Sugar Trend")
            fig_bs = px.line(dashboard.historical_data, x='date', y='blood_sugar',
                           title='Blood Sugar Trend',
                           labels={'blood_sugar': 'Blood Sugar (mg/dL)', 'date': 'Date'})
            fig_bs.update_traces(line_color='purple')
            st.plotly_chart(fig_bs, use_container_width=True)
        
        # Weight and Activity
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Weight Trend")
            fig_weight = px.bar(dashboard.historical_data, x='date', y='weight',
                              title='Weight Trend',
                              labels={'weight': 'Weight (kg)', 'date': 'Date'})
            fig_weight.update_traces(marker_color='green')
            st.plotly_chart(fig_weight, use_container_width=True)
        
        with col4:
            st.markdown("### Daily Steps")
            fig_steps = px.area(dashboard.historical_data, x='date', y='daily_steps',
                              title='Daily Steps Trend',
                              labels={'daily_steps': 'Steps', 'date': 'Date'})
            fig_steps.update_traces(fill='tonexty', fillcolor='rgba(0,176,246,0.2)')
            st.plotly_chart(fig_steps, use_container_width=True)
    
    # AI Predictions Page
    elif page == "AI Predictions":
        st.markdown("## 🤖 AI-Powered Health Predictions")
        
        # Risk Assessment
        st.markdown("### Risk Assessment")
        risk_scores = dashboard.calculate_risk_scores()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_cardio = dashboard.create_risk_gauge("Cardiovascular Risk", 
                                                   risk_scores['cardiovascular'], "red")
            st.plotly_chart(fig_cardio, use_container_width=True)
        
        with col2:
            fig_diabetes = dashboard.create_risk_gauge("Diabetes Risk", 
                                                     risk_scores['diabetes'], "orange")
            st.plotly_chart(fig_diabetes, use_container_width=True)
        
        with col3:
            fig_hypertension = dashboard.create_risk_gauge("Hypertension Risk", 
                                                         risk_scores['hypertension'], "purple")
            st.plotly_chart(fig_hypertension, use_container_width=True)
        
        # 30-Day Predictions
        st.markdown("### 30-Day Health Forecast")
        predictions = dashboard.generate_predictions()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Predicted Changes")
            st.info(f"🫀 Heart Rate: {predictions['heart_rate']} bpm (Target: <70)")
            st.info(f"⚖️ Weight: {predictions['weight']} kg")
            st.info(f"📊 Health Score: {predictions['health_score']}/100")
            
            if predictions['heart_rate'] < 70:
                st.success("✅ Heart rate improvement expected!")
            if predictions['weight'] < dashboard.historical_data['weight'].mean():
                st.success("✅ Weight loss trend detected!")
        
        with col2:
            st.markdown("#### AI Insights")
            health_score = dashboard.calculate_health_score()
            
            if health_score >= 80:
                st.success("🟢 Your health trends are excellent! Continue current habits.")
            elif health_score >= 60:
                st.warning("🟡 Some improvements possible. Focus on exercise and sleep.")
            else:
                st.error("🔴 Multiple risk factors detected. Consult healthcare provider.")
            
            st.info("📈 Predicted 7-point health score improvement with recommended changes")
            st.info("🎯 Focus areas: Daily steps, hydration, and sleep quality")
    
    # Recommendations Page
    elif page == "Recommendations":
        st.markdown("## 💡 Personalized Health Recommendations")
        
        recommendations = dashboard.generate_recommendations()
        
        if recommendations:
            st.markdown("### 🎯 Priority Actions")
            
            for rec in recommendations:
                priority_color = {
                    'High': 'error',
                    'Medium': 'warning',
                    'Low': 'info'
                }
                
                with st.container():
                    if rec['priority'] == 'High':
                        st.error(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
                    elif rec['priority'] == 'Medium':
                        st.warning(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
                    else:
                        st.info(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
        else:
            st.success("🎉 Great job! All your health metrics are within optimal ranges!")
        
        # Weekly Action Plan
        st.markdown("### 📅 This Week's Action Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Daily Goals")
            st.markdown("""
            - 🚶 Walk 10,000+ steps
            - 💧 Drink 8+ glasses of water
            - 😴 Sleep 7-8 hours
            - ❤️ Monitor blood pressure daily
            - 🥗 Eat 5 servings of fruits/vegetables
            """)
        
        with col2:
            st.markdown("#### Weekly Targets")
            st.markdown("""
            - 🏃 3 cardio exercise sessions (30 min each)
            - 💪 2 strength training sessions
            - 🧘 Daily meditation (10 minutes)
            - 📱 Track all meals and snacks
            - 🩺 Complete health metric logging
            """)
        
        # Nutrition Recommendations
        st.markdown("### 🥗 Nutrition Recommendations")
        
        bmi = dashboard.calculate_bmi()
        metrics = st.session_state.current_metrics
        
        if bmi > 25:
            st.markdown("#### Weight Management Diet")
            st.info("🍽️ **Caloric Deficit**: Reduce daily intake by 300-500 calories")
            st.info("🥦 **Vegetables**: Fill half your plate with non-starchy vegetables")
            st.info("🍗 **Protein**: Include lean protein with every meal")
        
        if metrics['blood_sugar'] > 100:
            st.markdown("#### Blood Sugar Management")
            st.warning("🍞 **Carbohydrates**: Choose complex carbs over simple sugars")
            st.warning("⏰ **Meal Timing**: Eat smaller, more frequent meals")
        
        if metrics['systolic_bp'] > 130:
            st.markdown("#### Blood Pressure Diet")
            st.error("🧂 **Sodium**: Limit sodium intake to <2300mg per day")
            st.error("🍌 **Potassium**: Increase potassium-rich foods (bananas, spinach)")

if __name__ == "__main__":
    main()