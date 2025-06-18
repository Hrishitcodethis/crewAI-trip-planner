from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import requests
import json
from datetime import datetime, timedelta
import os

class TravelTools:
    def __init__(self):
        # Initialize API keys
        self.opentripmap_api_key = os.getenv('OPENTRIPMAP_API_KEY')
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.currency_api_key = os.getenv('CURRENCY_API_KEY')
        self.eventbrite_api_key = os.getenv('EVENTBRITE_API_KEY')
        self.aviation_stack_api_key = os.getenv('AVIATION_STACK_API_KEY')
        self.transitland_api_key = os.getenv('TRANSITLAND_API_KEY')

    @staticmethod
    def calculate_match_score(city: Dict[str, Any], preferences: List[str], budget: float, season: str) -> float:
        """Calculate how well a city matches the user's preferences."""
        score = 0.0
        max_score = 1.0
        
        # Preference matching (40% of total score)
        preference_weights = {
            "Beach": ["coastal", "beach", "seaside", "ocean"],
            "Mountains": ["mountain", "hiking", "skiing", "alpine"],
            "City Life": ["urban", "metropolitan", "city", "downtown"],
            "Culture": ["museum", "art", "history", "cultural", "heritage"],
            "Food": ["cuisine", "restaurant", "gastronomy", "culinary"],
            "Adventure": ["adventure", "outdoor", "sports", "activities"],
            "Relaxation": ["spa", "wellness", "peaceful", "tranquil"],
            "Nightlife": ["nightlife", "entertainment", "bars", "clubs"]
        }
        
        description = city.get("description", "").lower()
        for pref in preferences:
            if pref in preference_weights:
                for keyword in preference_weights[pref]:
                    if keyword in description:
                        score += 0.1  # 0.1 points per matching keyword
        
        # Budget matching (30% of total score)
        daily_cost = city.get("estimated_cost", {}).get("total_per_day", 0)
        if daily_cost <= budget:
            score += 0.3  # Full points if within budget
        else:
            # Reduce score based on how much over budget
            budget_diff = (daily_cost - budget) / budget
            score += max(0, 0.3 * (1 - budget_diff))
        
        # Season matching (30% of total score)
        season_weather = {
            "Spring": ["mild", "spring", "pleasant", "temperate"],
            "Summer": ["hot", "summer", "warm", "sunny"],
            "Fall": ["autumn", "fall", "cool", "mild"],
            "Winter": ["cold", "winter", "snow", "chilly"]
        }
        
        if season in season_weather:
            for keyword in season_weather[season]:
                if keyword in description:
                    score += 0.1  # 0.1 points per matching season keyword
        
        # Normalize score to be between 0 and 1
        return min(max(score, 0), 1)

    @staticmethod
    def calculate_travel_budget(destination: str, duration: int, preferences: list) -> Dict[str, float]:
        """Calculate estimated travel budget based on destination, duration, and preferences, using real currency conversion."""
        try:
            geo = TravelTools.geocode_city(destination)
            country = geo.get('country', 'US')
            # Use restcountries API to get currency code
            currency_code = 'USD'
            try:
                rest_url = f'https://restcountries.com/v3.1/alpha/{country}'
                rest_resp = requests.get(rest_url)
                rest_data = rest_resp.json()
                currency_code = list(rest_data[0]['currencies'].keys())[0]
            except Exception:
                pass
            # Get currency rates
            currency_api_key = os.getenv('CURRENCY_API_KEY')
            rates_url = f'https://v6.exchangerate-api.com/v6/{currency_api_key}/latest/USD'
            rates_resp = requests.get(rates_url)
            rates = rates_resp.json().get('conversion_rates', {})
            rate = rates.get(currency_code, 1.0)
            # Base costs in USD
            base_costs = {
                "accommodation": 100,
                "food": 50,
                "activities": 75,
                "transportation": 200
            }
            # Convert to destination currency
            converted_costs = {key: round(value * rate, 2) for key, value in base_costs.items()}
            total = sum(converted_costs.values()) * duration
            return {
                **converted_costs,
                "total": total
            }
        except Exception as e:
            return {
                "accommodation": 100 * duration,
                "food": 50 * duration,
                "activities": 75 * duration,
                "transportation": 200,
                "total": (100 + 50 + 75) * duration + 200
            }

    @staticmethod
    def geocode_city(city_name: str) -> dict:
        """Get latitude, longitude, and country for a city using OpenTripMap."""
        try:
            url = f"https://api.opentripmap.com/0.1/en/places/geoname"
            params = {
                'name': city_name,
                'apikey': os.getenv('OPENTRIPMAP_API_KEY')
            }
            response = requests.get(url, params=params)
            data = response.json()
            return {
                'lat': data.get('lat', 0),
                'lon': data.get('lon', 0),
                'country': data.get('country', ''),
                'name': data.get('name', city_name)
            }
        except Exception as e:
            return {'lat': 0, 'lon': 0, 'country': '', 'name': city_name}

    @staticmethod
    def get_safety_information(destination: str) -> str:
        """Get safety information for a destination using OpenTripMap tags."""
        try:
            geo = TravelTools.geocode_city(destination)
            url = f"https://api.opentripmap.com/0.1/en/places/radius"
            params = {
                'radius': 1000,
                'lon': geo['lon'],
                'lat': geo['lat'],
                'apikey': os.getenv('OPENTRIPMAP_API_KEY')
            }
            response = requests.get(url, params=params)
            data = response.json()
            # Try to extract tags or info from the first POI
            pois = data.get('features', [])
            tags = []
            if pois:
                for poi in pois:
                    tags.extend(poi.get('properties', {}).get('kinds', '').split(','))
            # Simple logic: if 'danger' or 'safety' in tags, flag it
            general_safety = "Generally safe for tourists"
            if any('danger' in tag or 'safety' in tag for tag in tags):
                general_safety = "Some safety concerns reported. Check local advisories."
            return json.dumps({
                "general_safety": general_safety,
                "health_concerns": "Check local health advisories",
                "crime_rate": "Check local crime statistics",
                "natural_disasters": "Check local disaster risk"
            })
        except Exception as e:
            return json.dumps({
                "general_safety": "Generally safe for tourists",
                "health_concerns": "No major health concerns",
                "crime_rate": "Low",
                "natural_disasters": "Low risk"
            })

    @staticmethod
    def get_weather_forecast(destination: str, date: str = None) -> Dict[str, Any]:
        """Get weather forecast for a destination and date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                'key': os.getenv('WEATHER_API_KEY'),
                'q': destination,
                'dt': date,
                'aqi': 'no'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            return {
                "temperature": data['current']['temp_c'],
                "condition": data['current']['condition']['text'],
                "humidity": data['current']['humidity'],
                "wind_speed": data['current']['wind_kph']
            }
        except Exception as e:
            return {
                "temperature": 25,
                "condition": "Sunny",
                "humidity": 60,
                "wind_speed": 10
            }

    @staticmethod
    def get_local_events(destination: str, date_range: Dict[str, str] = None) -> list:
        """Get local events for a destination."""
        try:
            geo = TravelTools.geocode_city(destination)
            url = "https://www.eventbriteapi.com/v3/events/search/"
            params = {
                'location.latitude': geo['lat'],
                'location.longitude': geo['lon'],
                'location.within': '10km',
                'expand': 'venue',
                'token': os.getenv('EVENTBRITE_API_KEY')
            }
            if date_range:
                params['start_date.range_start'] = date_range.get('start')
                params['start_date.range_end'] = date_range.get('end')
            
            response = requests.get(url, params=params)
            data = response.json()
            
            events = []
            for event in data.get('events', [])[:5]:
                events.append({
                    'name': event['name']['text'],
                    'date': event['start']['local'],
                    'venue': event['venue']['name'],
                    'url': event['url']
                })
            if not events:
                events.append({
                    "name": "No major events found",
                    "date": date_range["start"],
                    "description": f"No major events found for {city_name} in this period.",
                    "location": city_name
                })
            return events
        except Exception as e:
            return [
                {
                    "name": "Local Festival",
                    "date": date_range["start"],
                    "description": "Annual cultural festival",
                    "location": destination
                }
            ]

    @staticmethod
    def get_transportation_routes(origin: str, destination: str, date: str = None) -> Dict[str, Any]:
        """Get transportation routes between two locations using real coordinates."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        try:
            # Geocode origin and destination
            origin_geo = TravelTools.geocode_city(origin)
            dest_geo = TravelTools.geocode_city(destination)
            # Use Aviation Stack API for flight information
            url = "http://api.aviationstack.com/v1/flights"
            params = {
                'access_key': os.getenv('AVIATION_STACK_API_KEY'),
                'dep_iata': origin[:3].upper(),  # Placeholder: ideally use a real IATA lookup
                'arr_iata': destination[:3].upper(),
                'flight_date': date
            }
            response = requests.get(url, params=params)
            flight_data = response.json()
            # Use TransitLand API for ground transportation
            transit_url = f"https://transit.land/api/v2/routes"
            transit_params = {
                'api_key': os.getenv('TRANSITLAND_API_KEY'),
                'lat': dest_geo['lat'],
                'lon': dest_geo['lon'],
                'radius': 1000
            }
            transit_response = requests.get(transit_url, params=transit_params)
            transit_data = transit_response.json()
            return {
                "flights": flight_data.get('data', []),
                "transit_routes": transit_data.get('routes', [])
            }
        except Exception as e:
            return {
                "flights": [],
                "transit_routes": []
            } 