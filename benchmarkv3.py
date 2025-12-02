import argparse
import json
import time
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import sys

import requests

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationError
)

INPUT_DATASET_PATH = "prompts_dataset.jsonl"
OUTPUT_DATASET_PATH = 'output_dataset.jsonl'
OLLAMA_API_BASE_URL = "http://localhost:11434"
TIMEOUT = 180

# Caso test, ainda tenho q colocar os outros modelos
DEFAULT_MODELS = ["llama3.2"]

MODEL_CONFIG = {
    "stream": False,
    "options": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_ctx": 4096
    }
}


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True) # Tokens do prompt
    prompt_eval_duration: int
    eval_count: int # Tokens da resposta
    eval_duration: int # Tempo para gerar a resposta

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value):
        if isinstance(value, str):
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    return datetime.now()
        return value

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print("\nWarning: prompt token count was not provided, potentially due to prompt caching.")
            return 0
        return value


class DatasetEntry(BaseModel):
    id: str
    prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OutputEntry(BaseModel):
    id: str
    model: str
    prompt: str
    response: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class OllamaAPIClient:
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.timeout = TIMEOUT
    
    def list_models(self) -> List[str]:
        # Lista modelos disponiveis
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = DEFAULT_MODELS #data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def chat(self, model: str, messages: List[Dict[str, str]], 
        stream: bool = False, options: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {}
        }
        
        try:
            if stream:
                response = requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                full_content = ""
                last_chunk = None
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            last_chunk = chunk
                            if chunk.get("message") and chunk["message"].get("content"):
                                content = chunk["message"]["content"]
                                full_content += content
                        except json.JSONDecodeError:
                            continue
                
                # Construir resposta final
                if last_chunk:
                    final_response = last_chunk.copy()
                    final_response["message"]["content"] = full_content
                    return final_response
                else:
                    return {"error": "No response received"}
                    
            else:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Timeout after {self.timeout} seconds")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request error: {e}")
            return {"error": str(e)}
    
    def generate(self, model: str, prompt: str, stream: bool = False,
        options: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {}
        }
        
        try:
            if stream:
                response = requests.post(
                    url,
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                full_response = ""
                last_chunk = None
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            last_chunk = chunk
                            if chunk.get("response"):
                                full_response += chunk["response"]
                        except json.JSONDecodeError:
                            continue
                
                if last_chunk:
                    last_chunk["response"] = full_response
                    return last_chunk
                else:
                    return {"error": "No response received"}
                    
            else:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Timeout after {self.timeout} seconds")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request error: {e}")
            return {"error": str(e)}


def convert_chat_to_ollama_response(api_response: Dict[str, Any], model_name: str) -> Optional[OllamaResponse]:
    if "error" in api_response:
        print(f"API Error: {api_response['error']}")
        return None
    
    try:
        # Extrair dados da resposta de chat
        message_content = api_response.get("message", {}).get("content", "")
        if not message_content:
            message_content = api_response.get("response", "")
        
        response_data = {
            "model": model_name,
            "created_at": api_response.get("created_at", datetime.now().isoformat()),
            "message": {
                "role": "assistant",
                "content": message_content
            },
            "done": api_response.get("done", True),
            "total_duration": api_response.get("total_duration", 0),
            "load_duration": api_response.get("load_duration", 0),
            "prompt_eval_count": api_response.get("prompt_eval_count", -1),
            "prompt_eval_duration": api_response.get("prompt_eval_duration", 0),
            "eval_count": api_response.get("eval_count", 0),
            "eval_duration": api_response.get("eval_duration", 0)
        }
        
        return OllamaResponse.model_validate(response_data)
        
    except Exception as e:
        print(f"Error converting chat response: {e}")
        print(f"API Response: {json.dumps(api_response, indent=2)[:500]}...")
        return None


def load_prompts_from_dataset(file_path: str) -> List[DatasetEntry]:
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = DatasetEntry(**data)
                    prompts.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                except ValidationError as e:
                    print(f"Validation error on line {line_num}: {e}")
        print(f"Loaded {len(prompts)} prompts from {file_path}")
    except FileNotFoundError:
        print(f"Dataset file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    return prompts


def save_output_to_dataset(entry: OutputEntry, file_path: str):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = entry.model_dump_json()
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Error saving output: {e}")


def run_benchmark_direct_api(
    model_name: str, 
    prompt: str, 
    verbose: bool = False,
    use_chat_api: bool = True
) -> Optional[OllamaResponse]:
    
    client = OllamaAPIClient(OLLAMA_API_BASE_URL)
    
    try:
        if use_chat_api:
            messages = [
                {"role": "user", "content": prompt}
            ]
            api_response = client.chat(
                model_name, 
                messages, 
                stream=False,
                options=MODEL_CONFIG.get("options", {})
            )
        else:
            api_response = client.generate(
                model_name,
                prompt,
                stream=verbose,
                options=MODEL_CONFIG.get("options", {})
            )
        
        if verbose and not use_chat_api:
            # Para generate API, mostrar resposta
            if "response" in api_response:
                print(api_response["response"])
        
        # Converter para OllamaResponse
        return convert_chat_to_ollama_response(api_response, model_name)
        
    except Exception as e:
        print(f"Error running benchmark for model {model_name}: {e}")
        return None


def nanosec_to_sec(nanosec: int) -> float:
    return nanosec / 1_000_000_000


def inference_stats(model_response: OllamaResponse):
    
    if not model_response:
        print("No response data available for statistics")
        return
    
    prompt_eval_sec = nanosec_to_sec(model_response.prompt_eval_duration)
    eval_sec = nanosec_to_sec(model_response.eval_duration)
    total_eval_sec = nanosec_to_sec(
        model_response.prompt_eval_duration + model_response.eval_duration
    )
    
    prompt_ts = model_response.prompt_eval_count / prompt_eval_sec if prompt_eval_sec > 0 else 0
    response_ts = model_response.eval_count / eval_sec if eval_sec > 0 else 0
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / total_eval_sec if total_eval_sec > 0 else 0

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def extract_metrics(model_response: OllamaResponse) -> Dict[str, Any]:    
    if not model_response:
        return {}
    
    prompt_eval_sec = nanosec_to_sec(model_response.prompt_eval_duration)
    eval_sec = nanosec_to_sec(model_response.eval_duration)
    total_eval_sec = nanosec_to_sec(
        model_response.prompt_eval_duration + model_response.eval_duration
    )
    
    prompt_ts = model_response.prompt_eval_count / prompt_eval_sec if prompt_eval_sec > 0 else 0
    response_ts = model_response.eval_count / eval_sec if eval_sec > 0 else 0
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / total_eval_sec if total_eval_sec > 0 else 0
    
    # Retorna dicionario com todas as metricas
    return {
        "prompt_tokens_per_second": prompt_ts,
        "response_tokens_per_second": response_ts,
        "total_tokens_per_second": total_ts,
        "prompt_tokens": model_response.prompt_eval_count,
        "response_tokens": model_response.eval_count,
        "total_tokens": model_response.prompt_eval_count + model_response.eval_count,
        "model_load_time_seconds": nanosec_to_sec(model_response.load_duration),
        "prompt_eval_time_seconds": nanosec_to_sec(model_response.prompt_eval_duration),
        "response_time_seconds": nanosec_to_sec(model_response.eval_duration),
        "total_time_seconds": nanosec_to_sec(model_response.total_duration),
        "done": model_response.done
    }


def average_stats(responses: List[Optional[OllamaResponse]]):
    valid_responses = [r for r in responses if r is not None]
    if len(valid_responses) == 0:
        print("No valid responses to average")
        return
    
    res = OllamaResponse(
        model=valid_responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(valid_responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in valid_responses),
        load_duration=sum(r.load_duration for r in valid_responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in valid_responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in valid_responses),
        eval_count=sum(r.eval_count for r in valid_responses),
        eval_duration=sum(r.eval_duration for r in valid_responses),
    )
    print(f"\nAverage stats across {len(valid_responses)} runs:")
    inference_stats(res)


def get_benchmark_models(
    skip_models: List[str] = [], 
    specified_models: List[str] = None
) -> List[str]:
    
    if specified_models:
        model_names = specified_models
    else:
        try:
            client = OllamaAPIClient(OLLAMA_API_BASE_URL)
            model_names = client.list_models()
            if not model_names:
                print("No models found via API, using default models")
                model_names = DEFAULT_MODELS
        except Exception as e:
            print(f"Error getting models from ollama API: {e}")
            model_names = DEFAULT_MODELS
    
    if len(skip_models) > 0:
        model_names = [
            model for model in model_names if model not in skip_models
        ]
    
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=None,
        help="List of models to evaluate. If not specified, uses DEFAULT_MODELS or all installed models.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="store_true",
        default=False,
        help="Use prompts from dataset file instead of command line prompts.",
    )
    parser.add_argument(
        "-g",
        "--generate-api",
        action="store_true",
        default=False,
        help="Use generate API instead of chat API (legacy).",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=INPUT_DATASET_PATH,
        help=f"Path to input dataset JSONL file (default: {INPUT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default=OUTPUT_DATASET_PATH,
        help=f"Path to output dataset JSONL file (default: {OUTPUT_DATASET_PATH})",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without actually running benchmarks.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    use_generate_api = args.generate_api
    use_dataset = args.dataset
    input_dataset_path = args.input_dataset
    output_dataset_path = args.output_dataset
    max_prompts = args.max_prompts
    dry_run = args.dry_run
    
    print(f"\nVerbose: {verbose}")
    print(f"Skip models: {skip_models}")
    print(f"Use generate API (legacy): {use_generate_api}")
    print(f"Use dataset: {use_dataset}")
    print(f"Input dataset: {input_dataset_path}")
    print(f"Output dataset: {output_dataset_path}")
    if max_prompts:
        print(f"Max prompts: {max_prompts}")
    print(f"Dry run: {dry_run}")

    # Load prompts do dataset
    if use_dataset:
        dataset_entries = load_prompts_from_dataset(input_dataset_path)
        if not dataset_entries:
            print("No prompts loaded from dataset. Exiting.")
            return
        prompts_data = dataset_entries
    else:
        # Usar prompts padrao
        prompts_data = [
            DatasetEntry(id="default_1", prompt="Why is the sky blue?", metadata={"source": "default"}),
            DatasetEntry(id="default_2", prompt="Write a report on the financials of Apple Inc.", metadata={"source": "default"})
        ]
    
    if max_prompts and len(prompts_data) > max_prompts:
        print(f"Limiting to {max_prompts} prompts (from {len(prompts_data)})")
        prompts_data = prompts_data[:max_prompts]
    
    print(f"Using {len(prompts_data)} prompts")

    model_names = get_benchmark_models(
        skip_models=skip_models, 
        specified_models=args.models
    )
    
    if not model_names:
        print("No models to evaluate. Exiting.")
        return
    
    if dry_run:
        print("\nDry run mode - would process:")
        print(f"  Models: {model_names}")
        print(f"  Prompts: {len(prompts_data)}")
        print(f"  Total runs: {len(model_names) * len(prompts_data)}")
        print("\nExiting dry run.")
        return

    try:
        open(output_dataset_path, 'w').close()
        print(f"Cleared output file: {output_dataset_path}")
    except:
        print(f"Creating new output file: {output_dataset_path}")

    benchmarks = {}

    for model_idx, model_name in enumerate(model_names, 1):
        responses: List[Optional[OllamaResponse]] = []
        
        print(f"\n{'='*60}")
        print(f"Model {model_idx}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        for prompt_idx, prompt_entry in enumerate(prompts_data, 1):
            prompt_id = prompt_entry.id
            prompt_text = prompt_entry.prompt
            prompt_metadata = prompt_entry.metadata
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Model: {model_name} ({model_idx}/{len(model_names)})")
                print(f"Prompt: {prompt_idx}/{len(prompts_data)}")
                print(f"Prompt ID: {prompt_id}")
                print(f"Prompt: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"Prompt: {prompt_text}")
                print(f"{'='*60}")
            else:
                # Mostrar progresso basico
                print(f"  [{prompt_idx}/{len(prompts_data)}] Processing prompt '{prompt_id}'...", end="", flush=True)
            
            response = run_benchmark_direct_api(
                model_name, 
                prompt_text, 
                verbose=verbose,
                use_chat_api=not use_generate_api
            )
            
            if response:
                responses.append(response)
                
                metrics = extract_metrics(response)
                
                output_entry = OutputEntry(
                    id=prompt_id,
                    model=model_name,
                    prompt=prompt_text,
                    response=response.message.content,
                    metrics=metrics,
                    metadata={
                        **prompt_metadata,
                        "model_config": MODEL_CONFIG,
                        "api_used": "generate" if use_generate_api else "chat"
                    }
                )
                
                save_output_to_dataset(output_entry, output_dataset_path)
                
                if not verbose:
                    print(f" ✓ ({metrics.get('total_tokens', 0)} tokens, {metrics.get('total_time_seconds', 0):.2f}s)")
                
                if verbose:
                    print(f"\nResponse saved to dataset")
                    inference_stats(response)
            else:
                print(f" ✗ Failed")
                if not verbose:
                    print(f"  Failed to get response for prompt '{prompt_id}'")
        
        benchmarks[model_name] = responses
        
        if responses:
            print(f"\nSummary for {model_name}:")
            average_stats(responses)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*60}")
    print(f"Total prompts processed: {len(prompts_data)}")
    print(f"Models evaluated: {len(model_names)}")
    print(f"Total runs: {sum(len(r) for r in benchmarks.values())}")
    print(f"Output saved to: {output_dataset_path}")
    
    # Summary por modelo
    for model_name, responses in benchmarks.items():
        valid_responses = [r for r in responses if r is not None]
        print(f"  {model_name}: {len(valid_responses)}/{len(responses)} successful")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()