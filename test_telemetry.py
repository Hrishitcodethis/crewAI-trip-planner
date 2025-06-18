import os
from dotenv import load_dotenv
from opentelemetry import trace
from trip_planner.telemetry import setup_telemetry
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def test_telemetry():
    print("Starting telemetry test for crewAI-trip-planner...")
    
    # Initialize telemetry
    try:
        tracer_provider = setup_telemetry()
        print("✓ Telemetry initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize telemetry: {str(e)}")
        return
    
    # Get a tracer
    tracer = trace.get_tracer(__name__)
    print("✓ Tracer obtained successfully")
    
    # Create a test span
    with tracer.start_as_current_span("test_telemetry_span") as span:
        span.set_attribute("test.attribute", "test_value")
        span.set_attribute("service.name", "crewAI-trip-planner")
        print("✓ Test span created with attributes")
        
        try:
            # Create a simple CrewAI test
            llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
            print("✓ LLM initialized")
            
            # Create a test agent
            test_agent = Agent(
                role="Test Agent",
                goal="Test telemetry integration",
                backstory="I am a test agent for telemetry verification",
                llm=llm
            )
            print("✓ Test agent created")
            
            # Create a test task
            test_task = Task(
                description="Test task for telemetry verification",
                agent=test_agent
            )
            print("✓ Test task created")
            
            # Create and run a test crew
            crew = Crew(
                agents=[test_agent],
                tasks=[test_task],
                verbose=True
            )
            print("✓ Test crew created")
            
            # Run the crew
            result = crew.kickoff()
            print("✓ Crew execution completed")
            
            print("\nTest completed successfully!")
            print("Result:", result)
            print("\nCheck your Arize AI dashboard for traces.")
            print("You should see:")
            print("1. A test_telemetry_span")
            print("2. CrewAI operation spans")
            print("3. The test attribute in the span attributes")
            print("4. Service name: crewAI-trip-planner")
            
        except Exception as e:
            print(f"✗ Error during test execution: {str(e)}")
            span.record_exception(e)
            raise

if __name__ == "__main__":
    test_telemetry() 