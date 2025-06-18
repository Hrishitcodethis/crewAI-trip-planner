from crewai import Crew
from textwrap import dedent
from trip_planner.agents import TripAgents
from trip_planner.tasks import TravelTasks
from datetime import datetime
import sys
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv
load_dotenv()
import os
from trip_planner.telemetry import setup_telemetry

# Initialize telemetry
try:
    tracer_provider = setup_telemetry()
except Exception as e:
    print(f"Warning: Failed to initialize telemetry: {str(e)}")

class TripCrew:
    def __init__(self, origin: str, cities: List[str], date_range: Dict[str, str], 
                 interests: List[str], budget: Optional[float] = None):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests
        self.budget = budget
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate all input parameters."""
        if not self.origin:
            raise ValueError("Origin city is required")
        
        if not self.cities:
            raise ValueError("At least one destination city is required")
        
        if not self.date_range or 'start' not in self.date_range or 'end' not in self.date_range:
            raise ValueError("Date range must include start and end dates")
        
        try:
            start_date = datetime.strptime(self.date_range['start'], "%Y-%m-%d")
            end_date = datetime.strptime(self.date_range['end'], "%Y-%m-%d")
            if end_date < start_date:
                raise ValueError("End date must be after start date")
            if (end_date - start_date).days > 90:
                raise ValueError("Trip duration cannot exceed 90 days")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")
        
        if not self.interests:
            raise ValueError("At least one interest is required")
        
        if self.budget is not None and self.budget <= 0:
            raise ValueError("Budget must be a positive number")

    def run(self):
        """Run the travel planning crew."""
        try:
            # Initialize agents and tasks
            agents = TripAgents()
            tasks = TravelTasks()

            # Create specialized agents
            expert_travel_agent = agents.expert_travel_agent()
            city_selection_expert = agents.city_selection_expert()
            local_tour_guide = agents.local_tour_guide()
            transportation_specialist = agents.transportation_specialist()
            accommodation_expert = agents.accommodation_expert()
            budget_planner = agents.budget_planner()

            # Create tasks with all necessary information
            plan_itinerary = tasks.plan_itinerary(
                expert_travel_agent,
                self.cities,
                self.date_range,
                self.interests,
                self.budget
            )

            identify_city = tasks.identify_city(
                city_selection_expert,
                self.origin,
                self.cities,
                self.interests,
                self.date_range,
                self.budget
            )

            gather_city_info = tasks.gather_city_info(
                local_tour_guide,
                self.cities,
                self.date_range,
                self.interests
            )

            plan_transportation = tasks.plan_transportation(
                transportation_specialist,
                self.origin,
                self.cities,
                self.date_range
            )

            find_accommodation = tasks.find_accommodation(
                accommodation_expert,
                self.cities,
                self.date_range,
                self.budget
            )

            create_budget = tasks.create_budget(
                budget_planner,
                self.cities,
                self.date_range,
                self.interests,
                self.budget
            )

            # Create and run the crew
            crew = Crew(
                agents=[
                    expert_travel_agent,
                    city_selection_expert,
                    local_tour_guide,
                    transportation_specialist,
                    accommodation_expert,
                    budget_planner
                ],
                tasks=[
                    identify_city,
                    gather_city_info,
                    plan_transportation,
                    find_accommodation,
                    create_budget,
                    plan_itinerary
                ],
                verbose=True,
            )

            result = crew.kickoff()
            return self._format_result(result)

        except Exception as e:
            print(f"Error during travel planning: {str(e)}")
            raise

    def _format_result(self, result: str) -> Dict:
        """Format the result into a structured dictionary."""
        try:
            # Try to parse the result as JSON if it's in JSON format
            return json.loads(result)
        except json.JSONDecodeError:
            # If not JSON, return as a formatted string
            return {"itinerary": result}


def get_user_input() -> Dict:
    """Get and validate user input for travel planning."""
    print("\n## Welcome to Trip Planner Crew")
    print('-------------------------------')
    
    # Get origin
    origin = input(dedent("""
    From where will you be traveling from?
    """)).strip()
    
    # Get cities
    cities_input = input(dedent("""
    What are the cities you are interested in visiting? (comma-separated)
    """)).strip()
    cities = [city.strip() for city in cities_input.split(',')]
    
    # Get date range
    start_date = input(dedent("""
    What is your start date? (YYYY-MM-DD)
    """)).strip()
    end_date = input(dedent("""
    What is your end date? (YYYY-MM-DD)
    """)).strip()
    
    # Get interests
    interests_input = input(dedent("""
    What are your interests and hobbies? (comma-separated)
    """)).strip()
    interests = [interest.strip() for interest in interests_input.split(',')]
    
    # Get budget
    budget_input = input(dedent("""
    What is your budget in USD? (press Enter to skip)
    """)).strip()
    budget = float(budget_input) if budget_input else None
    
    return {
        "origin": origin,
        "cities": cities,
        "date_range": {
            "start": start_date,
            "end": end_date
        },
        "interests": interests,
        "budget": budget
    }


def save_itinerary(itinerary: Dict, filename: str = "trip_plan.json"):
    """Save the itinerary to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(itinerary, f, indent=2)
        print(f"\nItinerary saved to {filename}")
    except Exception as e:
        print(f"Error saving itinerary: {str(e)}")


if __name__ == "__main__":
    try:
        # Get user input
        user_input = get_user_input()
        
        # Create and run trip crew
        trip_crew = TripCrew(**user_input)
        result = trip_crew.run()
        
        # Display results
        print("\n########################")
        print("## Here is your Trip Plan")
        print("########################\n")
        print(json.dumps(result, indent=2))
        
        # Save itinerary
        save_itinerary(result)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)