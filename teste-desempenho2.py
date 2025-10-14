from locust import HttpUser, task, between
import logging
import json
import time 

model_name = "llama3.2"

system_message = "You are a long and high-quality story teller. Make the story engaging and detailed."

messages = [
    {
        "role": "user", 
        "content": """
        Rex and Charlie were best friends who did everything together. 
        They lived next door to each other with their human families and spent all day playing in the backyard. 
        Rex was a golden retriever, always happy and eager for fun. Charlie was a German shepherd, more serious but very loyal.  
        Every morning, Rex and Charlie would wake up and bark excitedly, ready to start the day's adventures. 
        Their families would let them out into the backyard and they'd run around chasing each other and sniffing for interesting smells. 
        After tiring themselves out, they'd nap in the shade of the big oak tree, Rex's tail still thumping contentedly even in his sleep. 
        Continue this story...
        """
    },
    {"role":"user",
    "content":"tell me a baseball history"}

]
token_metrics = []

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:11434"  # URL base do ollama
    @task
    def generation(self):
        start_time = time.time()

        # Preparar o payload
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                *messages
            ],
            "stream": False,
            "options": {
                "num_predict": 300,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        # Fazer a requisicao usando o cliente http do locust
        with self.client.post(
            "/api/chat",
            json=payload,
            catch_response=True,
            name="Llama Generation",
            timeout=180
        ) as response:
            end_time = time.time()  # Medir tempo final
            request_duration = end_time - start_time
            # Se respondido corretamente, podemos analisar o throughput e a latencia
            if response.status_code == 200:
                try:

                    response_body = response.json()
                    completion = response_body.get('message', {}).get('content', '')
                    if completion:
                        tokens_generated = len(completion.split())  # Estimativa de tokens
                        tokens_per_second = tokens_generated / request_duration if request_duration > 0 else 0
                        
                      
                        token_metrics.append({
                            'tokens_generated': tokens_generated,
                            'tokens_per_second': tokens_per_second,
                            'response_time': request_duration,
                            'timestamp': time.time()
                        })
                        response.success()
                        print(completion)
                        logging.info(f"Generated {len(completion)} characters")
                    else:
                        response.failure("Empty response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
        
        logging.info("Finished generation!")