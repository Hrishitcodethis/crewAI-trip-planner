from typing import Optional
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(..., description="The mathematical expression to evaluate")

class CalculatorTools:
    @staticmethod
    def calculate(expression: str) -> str:
        """Calculate the result of a mathematical expression."""
        try:
            # Safely evaluate the expression
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
            return str(result)
        except Exception as e:
            return f"Error calculating expression: {str(e)}" 