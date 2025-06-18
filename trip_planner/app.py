import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
load_dotenv()
from trip_planner.telemetry import setup_telemetry
from langchain_openai import ChatOpenAI
from .agents import TripAgents, TravelInput, CityInput
from .guardrails import GuardrailManager
from .tools import TravelTools
from crewai import Task, Crew
from opentelemetry import trace

# Initialize telemetry (will only initialize once due to singleton pattern)
tracer_provider = setup_telemetry()
if tracer_provider is None:
    st.warning("""
    Telemetry is not available. The application will continue without tracing.
    To enable tracing, set the following environment variables:
    - PHOENIX_ENABLED=true
    - PHOENIX_CLIENT_HEADERS=api_key=your_phoenix_api_key
    """)
else:
    st.success("Telemetry initialized successfully!")
    # Test trace
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("app_initialization") as span:
        span.set_attribute("app.name", "AI Travel Planner")
        span.set_attribute("app.version", "1.0.0")
        st.info("Test trace created successfully!")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
if not os.environ["OPENAI_API_KEY"]:
    st.error("Please set the OPENAI_API_KEY environment variable")
    st.stop()

# Initialize OpenAI
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7
)


# Initialize agents
agents = TripAgents(llm)

# Initialize guardrails
guardrails = GuardrailManager()

def initialize_session_state():
    """Initialize session state variables"""
    if 'travel_plan' not in st.session_state:
        st.session_state.travel_plan = None
    if 'selected_cities' not in st.session_state:
        st.session_state.selected_cities = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'city_selection'

def display_header():
    """Display the application header"""
    st.title("✈️ AI Travel Planner")
    st.markdown("""
    Plan your perfect trip with AI-powered travel agents. 
    Get personalized recommendations, detailed itineraries, and comprehensive travel plans.
    """)

def display_weather_forecast(destination: str, date: str):
    """Display weather forecast for a destination"""
    with st.spinner("Getting weather forecast..."):
        weather = TravelTools.get_weather_forecast(destination, date)
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        col1.metric("Temperature", f"{weather['temperature']}°C")
        # Use markdown for condition to avoid truncation
        col2.markdown(f"<span style='font-size:1.5em'>{weather['condition']}</span>", unsafe_allow_html=True)
        col3.metric("Humidity", f"{weather['humidity']}%")
        col4.metric("Wind Speed", f"{weather['wind_speed']} km/h")
    st.markdown("<br>", unsafe_allow_html=True)

def display_safety_info(destination: str):
    """Display safety information for a destination"""
    with st.spinner("Getting safety information..."):
        safety_info = json.loads(TravelTools.get_safety_information(destination))
        st.subheader("Safety Information")
        st.info(f"General Safety: {safety_info['general_safety']}")
        st.info(f"Health Concerns: {safety_info['health_concerns']}")
        st.info(f"Crime Rate: {safety_info['crime_rate']}")
        st.info(f"Natural Disasters: {safety_info['natural_disasters']}")
    st.markdown("<br>", unsafe_allow_html=True)

def display_local_events(destination: str, date_range: dict):
    """Display local events for a destination"""
    with st.spinner("Getting local events..."):
        events = TravelTools.get_local_events(destination, date_range)
        st.subheader("Local Events")
        for event in events:
            with st.expander(f"🎉 {event['name']} - {event['date']}"):
                st.markdown(f"**Location:** {event['location']}")
                st.markdown(f"**Description:** {event['description']}")

def display_budget_analysis(budget_breakdown: dict):
    """Display budget analysis with visualizations"""
    st.subheader("Budget Analysis")
    
    # Create pie chart for budget distribution
    fig = px.pie(
        values=list(budget_breakdown.values()),
        names=list(budget_breakdown.keys()),
        title="Budget Distribution"
    )
    st.plotly_chart(fig)
    
    # Display budget metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accommodation", f"${budget_breakdown['accommodation']}")
    col2.metric("Food", f"${budget_breakdown['food']}")
    col3.metric("Activities", f"${budget_breakdown['activities']}")
    col4.metric("Transportation", f"${budget_breakdown['transportation']}")
    col5.metric("Total", f"${budget_breakdown['total']}")

def display_city_comparison(cities: list):
    """Display comparison of recommended cities only if 3 or more cities are present."""
    if len(cities) < 3:
        return  # Skip plots for 1-2 cities
    st.subheader("City Comparison")
    metrics = {
        "Match Score": [city["match_score"] for city in cities],
        "Daily Cost": [city["estimated_cost"]["total_per_day"] for city in cities],
        "City": [city["name"] for city in cities]
    }
    fig = px.bar(
        x="City",
        y="Match Score",
        data_frame=pd.DataFrame(metrics),
        title="City Match Scores"
    )
    st.plotly_chart(fig)
    fig = px.bar(
        x="City",
        y="Daily Cost",
        data_frame=pd.DataFrame(metrics),
        title="Daily Costs Comparison"
    )
    st.plotly_chart(fig)

def display_city_recommendations(cities):
    """Display city recommendations as summary cards for all cities."""
    st.subheader("Recommended Cities")
    display_city_comparison(cities)
    # Show all cities as cards in a grid
    cols = st.columns(min(3, len(cities)))
    for idx, city in enumerate(cities):
        with cols[idx % len(cols)]:
            st.markdown(f"### 🌆 {city['name']}, {city['country']} (Score: {city['match_score']:.2f})")
            st.markdown(f"**Description:** {city['description']}")
            st.markdown(f"**Estimated Daily Cost:** ${city['estimated_cost']['total_per_day']}")
            st.markdown(f"**Highlights:** {' | '.join(city['highlights'])}")
            # Weather summary
            weather = TravelTools.get_weather_forecast(city['name'], datetime.now().strftime("%Y-%m-%d"))
            st.markdown(f"**Weather:** {weather['temperature']}°C, {weather['condition']}")
            # Events summary
            events = TravelTools.get_local_events(city['name'])
            if events:
                st.markdown(f"**Events:** {events[0]['name']} ({events[0]['date'][:10]})")
            # Safety summary
            st.markdown("[Travel Advisory](https://www.travel-advisory.info/) for safety info.")
            st.markdown("---")
    # Optionally, expand for details
    for city in cities:
        with st.expander(f"More about {city['name']}"):
            st.markdown(f"**Description:** {city['description']}")
            st.markdown("**Highlights:**")
            for highlight in city['highlights']:
                st.markdown(f"- :star: {highlight}")
            st.markdown("**Weather Forecast:**")
            display_weather_forecast(city['name'], datetime.now().strftime("%Y-%m-%d"))
            display_safety_info(city['name'])
            st.markdown("**Events:**")
            for event in TravelTools.get_local_events(city['name']):
                st.markdown(f"- {event['name']} ({event['date'][:10]})")
            st.markdown("**Estimated Daily Costs:**")
            costs = city['estimated_cost']
            st.metric("Accommodation", f"${costs['accommodation']}")
            st.metric("Food", f"${costs['food']}")
            st.metric("Activities", f"${costs['activities']}")
            st.metric("Total per day", f"${costs['total_per_day']}", 
                     delta=f"${costs['total_per_day'] - 200:.2f} vs budget")
            st.markdown("<br>", unsafe_allow_html=True)

def display_travel_plan(plan):
    """Display travel plan in a visually appealing way"""
    st.subheader("Your Travel Plan")
    
    # Display budget breakdown with analysis
    display_budget_analysis(plan['budget_breakdown'])
    
    # Display itinerary
    st.markdown("### 📅 Daily Itinerary")
    for day in plan['itinerary']:
        with st.expander(f"Day {day['day']} - {day['date']}"):
            # Display weather forecast for the day
            display_weather_forecast(plan['destination'], day['date'])
            
            # Display activities
            st.markdown("#### 🎯 Activities")
            for activity in day['activities']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{activity['time']} - {activity['activity']}**")
                    st.markdown(f"📍 {activity['location']}")
                    st.markdown(f"⏱️ {activity['duration']}")
                with col2:
                    st.metric("Cost", f"${activity['cost']}")
                st.markdown(f"_{activity['description']}_")
                st.divider()
            
            # Display meals
            st.markdown("#### 🍽️ Meals")
            for meal in day['meals']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{meal['time']} - {meal['type']}**")
                    st.markdown(f"🍴 {meal['suggestion']}")
                with col2:
                    st.metric("Cost", f"${meal['cost']}")
    
    # Display recommendations
    st.markdown("### 💡 Recommendations")
    for i, rec in enumerate(plan['recommendations'], 1):
        st.markdown(f"{i}. {rec}")
    
    # Add export options
    st.markdown("### 📤 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export as JSON"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(plan, indent=2),
                file_name="travel_plan.json",
                mime="application/json"
            )
    with col2:
        if st.button("Export as CSV"):
            # Convert itinerary to CSV format
            itinerary_data = []
            for day in plan['itinerary']:
                for activity in day['activities']:
                    itinerary_data.append({
                        'Day': day['day'],
                        'Date': day['date'],
                        'Time': activity['time'],
                        'Activity': activity['activity'],
                        'Location': activity['location'],
                        'Duration': activity['duration'],
                        'Cost': activity['cost']
                    })
            df = pd.DataFrame(itinerary_data)
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="travel_plan.csv",
                mime="text/csv"
            )

def city_selection_form():
    """Display the city selection form and handle city recommendations"""
    st.subheader("Step 1: Select Your Destination")
    
    with st.form("city_selection"):
        # Get user preferences
        preferences = st.multiselect(
            "What are you looking for in your destination?",
            ["Beach", "Mountains", "City Life", "Culture", "Food", "Adventure", "Relaxation", "Nightlife"],
            default=["Beach", "Culture"]
        )
        
        budget = st.slider(
            "What's your budget per day (in USD)?",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )
        
        duration = st.slider(
            "How many days do you want to travel?",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )
        
        season = st.selectbox(
            "When do you plan to travel?",
            ["Spring", "Summer", "Fall", "Winter"]
        )
        
        submitted = st.form_submit_button("Get City Recommendations")
        
        if submitted:
            # Validate input using guardrails
            input_data = {
                "preferences": preferences,
                "budget": budget,
                "duration": duration,
                "season": season
            }
            
            is_valid, error_message = guardrails.validate_input(input_data)
            if not is_valid:
                st.error(error_message)
                return
            
            # Create input for city selection
            city_input = CityInput(
                preferences=preferences,
                budget=budget,
                duration=duration,
                season=season
            )
            
            # Get city recommendations
            with st.spinner("Getting city recommendations..."):
                city_expert = agents.city_selection_expert()
                task = Task(
                    description=f"""Based on these preferences: {city_input.dict()}, recommend cities for travel.\nReturn at least 5 cities in the response.\nYour response MUST be a valid JSON object with the following structure:\n{{\n    \"recommended_cities\": [\n        {{\n            \"name\": \"City Name\",\n            \"country\": \"Country Name\",\n            \"description\": \"Brief description of the city\",\n            \"match_score\": 0.95,\n            \"highlights\": [\"Highlight 1\", \"Highlight 2\"],\n            \"estimated_cost\": {{\n                \"accommodation\": 100,\n                \"food\": 50,\n                \"activities\": 75,\n                \"total_per_day\": 225\n            }}\n        }}\n    ]\n}}""",
                    agent=city_expert
                )
                crew = Crew(
                    agents=[city_expert],
                    tasks=[task]
                )
                with tracer.start_as_current_span("city_recommendation_task") as span:  # ✅ Tracing starts here
                    span.set_attribute("season", season)
                    span.set_attribute("budget", budget)
                    span.set_attribute("preferences", str(preferences))
                    result = crew.kickoff()
                
                # Debug logging
                st.write("Raw result:", result)
                
                # Validate output using guardrails
                try:
                    # Try to parse the result as JSON
                    if isinstance(result, str):
                        result_data = json.loads(result)
                    else:
                        result_data = result
                    
                    st.write("Parsed result data:", result_data)
                    
                    # Validate the structure
                    if not isinstance(result_data, dict):
                        st.error("Invalid response format: expected a dictionary")
                        return
                    
                    if 'recommended_cities' not in result_data:
                        st.error("Invalid response format: missing 'recommended_cities' key")
                        return
                    
                    if not isinstance(result_data['recommended_cities'], list):
                        st.error("Invalid response format: 'recommended_cities' should be a list")
                        return
                    
                    # Validate each city's structure
                    for i, city in enumerate(result_data['recommended_cities']):
                        required_fields = {
                            "name", "country", "description", "match_score",
                            "highlights", "estimated_cost"
                        }
                        missing_fields = required_fields - set(city.keys())
                        if missing_fields:
                            st.error(f"City {i+1} is missing required fields: {missing_fields}")
                            return
                        
                        if not isinstance(city['estimated_cost'], dict):
                            st.error(f"City {i+1}: 'estimated_cost' should be a dictionary")
                            return
                        
                        cost_fields = {
                            "accommodation", "food", "activities", "total_per_day"
                        }
                        missing_costs = cost_fields - set(city['estimated_cost'].keys())
                        if missing_costs:
                            st.error(f"City {i+1}: 'estimated_cost' is missing fields: {missing_costs}")
                            return
                    
                    # Store in session state
                    st.session_state.selected_cities = result_data
                    st.write("Session state after update:", st.session_state.selected_cities)
                    
                    # Display recommendations
                    try:
                        display_city_recommendations(result_data['recommended_cities'])
                    except Exception as e:
                        st.error(f"Error displaying recommendations: {str(e)}")
                        st.write("Debug - Cities data:", result_data['recommended_cities'])
                
                except json.JSONDecodeError as e:
                    st.error("Invalid JSON response from the AI. Please try again.")
                    st.write("Raw result that failed to parse:", result)
                    # Provide a fallback response
                    fallback_response = {
                        "recommended_cities": [
                            {
                                "name": "Barcelona",
                                "country": "Spain",
                                "description": "A vibrant city known for its beaches and rich cultural heritage.",
                                "match_score": 0.9,
                                "highlights": ["Sagrada Familia", "Beach", "Local Cuisine"],
                                "estimated_cost": {
                                    "accommodation": 80,
                                    "food": 40,
                                    "activities": 30,
                                    "total_per_day": 150
                                }
                            }
                        ]
                    }
                    st.session_state.selected_cities = fallback_response
                    display_city_recommendations(fallback_response['recommended_cities'])
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    st.write("Debug - Full error:", e)

def travel_planning_form():
    """Display the travel planning form and handle travel plan generation"""
    st.subheader("Step 2: Plan Your Trip")
    
    with st.form("travel_planning"):
        # Get selected city
        selected_city = st.selectbox(
            "Select your destination",
            [city["name"] for city in st.session_state.selected_cities['recommended_cities']]
        )
        
        # Get travel dates
        start_date = st.date_input(
            "Start Date",
            min_value=datetime.now().date(),
            value=datetime.now().date() + timedelta(days=7)
        )
        
        end_date = st.date_input(
            "End Date",
            min_value=start_date,
            value=start_date + timedelta(days=6)
        )
        
        # Get additional preferences
        activities = st.multiselect(
            "What activities interest you?",
            ["Sightseeing", "Museums", "Shopping", "Local Food", "Adventure Sports", "Relaxation", "Nightlife"],
            default=["Sightseeing", "Local Food"]
        )
        
        accommodation = st.selectbox(
            "Preferred accommodation type",
            ["Budget", "Mid-range", "Luxury"]
        )
        
        submitted = st.form_submit_button("Generate Travel Plan")
        
        if submitted:
            # Validate input using guardrails
            input_data = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "activities": activities
            }
            
            is_valid, error_message = guardrails.validate_input(input_data)
            if not is_valid:
                st.error(error_message)
                return
            
            # Create input for travel planning
            travel_input = TravelInput(
                destination=selected_city,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                activities=activities,
                accommodation=accommodation
            )
            
            # Generate travel plan
            with st.spinner("Generating your travel plan..."):
                travel_expert = agents.travel_planning_expert()
                task = Task(
                    description=f"""Create a detailed travel plan based on these preferences: {travel_input.dict()}
                    Your response MUST be a valid JSON object with the following structure:
                    {{
                        "itinerary": [
                            {{
                                "day": 1,
                                "date": "YYYY-MM-DD",
                                "activities": [
                                    {{
                                        "time": "09:00",
                                        "activity": "Activity name",
                                        "description": "Activity description",
                                        "location": "Location name",
                                        "duration": "2 hours",
                                        "cost": 50
                                    }}
                                ],
                                "meals": [
                                    {{
                                        "time": "12:00",
                                        "type": "Lunch",
                                        "suggestion": "Restaurant name",
                                        "cost": 30
                                    }}
                                ]
                            }}
                        ],
                        "budget_breakdown": {{
                            "accommodation": 500,
                            "food": 300,
                            "activities": 400,
                            "transportation": 200,
                            "total": 1400
                        }},
                        "recommendations": [
                            "Recommendation 1",
                            "Recommendation 2"
                        ]
                    }}""",
                    agent=travel_expert
                )
                travel_expert = agents.travel_planning_expert()
                task = Task(description=f"Plan travel for: {travel_input.dict()}", agent=travel_expert)
                crew = Crew(
                    agents=[travel_expert],
                    tasks=[task]
                )
                with tracer.start_as_current_span("travel_plan_generation_task") as span:  # ✅ Tracing block added
                    span.set_attribute("destination", travel_input.destination)
                    span.set_attribute("duration_days", (end_date - start_date).days)
                    span.set_attribute("activities", str(activities))
                    result = crew.kickoff()
                
                # Validate output using guardrails
                try:
                    result_data = json.loads(result)
                    is_valid, error_message = guardrails.validate_output(result_data, "travel_plan")
                    if not is_valid:
                        st.error(error_message)
                        return
                    
                    # Validate business rules
                    is_valid, error_message = guardrails.validate_business_rules(result_data)
                    if not is_valid:
                        st.error(error_message)
                        return
                    
                    st.session_state.travel_plan = result_data
                    display_travel_plan(st.session_state.travel_plan)
                except json.JSONDecodeError:
                    st.error("Invalid response format from the AI. Please try again.")

def main():
    """Main function to run the Streamlit app"""
    initialize_session_state()
    display_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Start Over"):
        st.session_state.current_step = 'city_selection'
        st.session_state.travel_plan = None
        st.session_state.selected_cities = None
        st.rerun()
    
    # Main content
    if st.session_state.current_step == 'city_selection':
        city_selection_form()
    elif st.session_state.current_step == 'travel_planning':
        travel_planning_form()
        if st.session_state.travel_plan:
            st.success("Your travel plan is ready! 🎉")
            st.balloons()

if __name__ == "__main__":
    main() 