from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from crewai import Agent
from textwrap import dedent
from langchain.llms import Ollama

from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools


class TripAgents:
    def __init__(self):
        self.Ollama = Ollama(model="openhermes")

        # Initialize LLaMA using transformers
        self.llama_model_name = "decapoda-research/llama-7b-hf"  # Replace with the correct model path
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(self.llama_model_name)
        self.llama_pipeline = pipeline("text-generation", model=self.llama_model, tokenizer=self.llama_tokenizer)

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f"""Expert in travel planning and logistics. 
                             I have decades of experience making travel iteneraries"""),
            goal=dedent(f"""
                        Create a 7-day travel itenerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose=True,
            llm=self.llama_pipeline,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(f"""I specialize in helping travelers select the best cities to visit based on their preferences and interests."""),
            goal=dedent(f"""Help the traveler select the best city for their vacation, taking into account their interests, budget, and time available."""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.llama_pipeline,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""I am an experienced local tour guide who knows all the hidden gems and must-see spots in the city."""),
            goal=dedent(f"""Create a detailed itinerary for a day tour in the city, including food recommendations, cultural experiences, and shopping spots."""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.llama_pipeline,
        )
