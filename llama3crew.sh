model_name="llama3"
custom_model_name="crewai-llama3"

ollama pull $model_name

ollama create $custom_model_name -f ./Llama3Modelfile