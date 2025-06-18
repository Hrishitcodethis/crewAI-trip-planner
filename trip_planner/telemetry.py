import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_telemetry():
    """Setup OpenTelemetry with Phoenix integration"""
    try:
        # Get the current tracer provider (should be set by Phoenix in streamlit_app.py)
        tracer_provider = trace.get_tracer_provider()
        
        if tracer_provider is None:
            # Fallback to console tracing if Phoenix setup failed
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
            trace.set_tracer_provider(tracer_provider)
            print("Using console tracing (Phoenix not configured)")
        
        return tracer_provider

    except Exception as e:
        print(f"Failed to setup telemetry: {str(e)}")
        return None 