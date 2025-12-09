import csv
import math
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from dataclasses import dataclass
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import requests
from pydantic import BaseModel, Field, field_validator, ValidationError
from tqdm import tqdm
#from config import ENVIRONMENT_FACTORS, DESIGN_SPECS


INPUT_DATASET_PATH = "prompts_dataset.jsonl"
OUTPUT_DATASET_PATH = 'output_dataset.jsonl'
OLLAMA_API_BASE_URL = "http://localhost:11434"
TIMEOUT = 180
DEFAULT_MODELS = ["llama3.2"]
MAX_WORKERS = 3  # Número máximo de threads paralelas

MODEL_CONFIG = {
    "stream": False,
    "options": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_ctx": 4096
    }
}


@dataclass
class BenchmarkMetrics:
    model_name: str
    prompt_id: str
    prompt_text: str
    response_text: str
    
    # Métricas temporais (nanosegundos)
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    
    # Contagem de tokens
    prompt_tokens: int
    response_tokens: int
    
    # Timestamps
    created_at: datetime
    processed_at: datetime
    
    # Métricas calculadas
    prompt_tokens_per_sec: float = 0.0
    response_tokens_per_sec: float = 0.0
    total_tokens_per_sec: float = 0.0
    
    def __post_init__(self):
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calcula métricas por segundo"""
        # Converter nanosegundos para segundos
        prompt_eval_sec = self.prompt_eval_duration_ns / 1_000_000_000
        eval_sec = self.eval_duration_ns / 1_000_000_000
        total_eval_sec = (self.prompt_eval_duration_ns + self.eval_duration_ns) / 1_000_000_000
        
        # Calcular tokens por segundo
        self.prompt_tokens_per_sec = self.prompt_tokens / prompt_eval_sec if prompt_eval_sec > 0 else 0
        self.response_tokens_per_sec = self.response_tokens / eval_sec if eval_sec > 0 else 0
        self.total_tokens_per_sec = (self.prompt_tokens + self.response_tokens) / total_eval_sec if total_eval_sec > 0 else 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens
    
    @property
    def load_duration_sec(self) -> float:
        return self.load_duration_ns / 1_000_000_000
    
    @property
    def prompt_eval_duration_sec(self) -> float:
        return self.prompt_eval_duration_ns / 1_000_000_000
    
    @property
    def eval_duration_sec(self) -> float:
        return self.eval_duration_ns / 1_000_000_000
    
    @property
    def total_duration_sec(self) -> float:
        return self.total_duration_ns / 1_000_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prompt_id": self.prompt_id,
            "timing_nanoseconds": {
                "total_duration": self.total_duration_ns,
                "load_duration": self.load_duration_ns,
                "prompt_eval_duration": self.prompt_eval_duration_ns,
                "eval_duration": self.eval_duration_ns
            },
            "timing_seconds": {
                "total_duration": self.total_duration_sec,
                "load_duration": self.load_duration_sec,
                "prompt_eval_duration": self.prompt_eval_duration_sec,
                "eval_duration": self.eval_duration_sec
            },
            "token_counts": {
                "prompt_tokens": self.prompt_tokens,
                "response_tokens": self.response_tokens,
                "total_tokens": self.total_tokens
            },
            "throughput_tokens_per_second": {
                "prompt": self.prompt_tokens_per_sec,
                "response": self.response_tokens_per_sec,
                "total": self.total_tokens_per_sec
            },
            "timestamps": {
                "created_at": self.created_at.isoformat(),
                "processed_at": self.processed_at.isoformat()
            }
        }


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
    benchmark_metrics: Dict[str, Any]  # Métricas detalhadas do benchmark
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class OllamaAPIClient:
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.timeout = TIMEOUT
        self.session = requests.Session()
    
    def list_models(self) -> List[str]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", 
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except requests.exceptions.RequestException as e:
            print(f"Erro ao listar modelos: {e}")
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
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Timeout após {self.timeout} segundos")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            print(f"Erro HTTP: {e}")
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
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.Timeout:
            print(f"Timeout após {self.timeout} segundos")
            return {"error": "timeout"}
        except requests.exceptions.RequestException as e:
            print(f"Erro HTTP: {e}")
            return {"error": str(e)}


def extract_benchmark_metrics(api_response: Dict[str, Any], model_name: str, 
                            prompt_id: str, prompt_text: str) -> Optional[BenchmarkMetrics]:
    
    if "error" in api_response:
        print(f"Erro na API: {api_response['error']}")
        return None
    
    try:
        # Extrair dados da resposta
        if "message" in api_response:  # API de chat
            response_text = api_response.get("message", {}).get("content", "")
            # Para chat API, usar eval_count como response_tokens
            response_tokens = api_response.get("eval_count", 0)
        else:  # API de generate
            response_text = api_response.get("response", "")
            response_tokens = api_response.get("eval_count", 0)
        
        # Extrair timestamps
        created_at_str = api_response.get("created_at", "")
        if created_at_str:
            try:
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                created_at = datetime.fromisoformat(created_at_str)
            except:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        # Extrair métricas
        prompt_tokens = api_response.get("prompt_eval_count", 0)
        if prompt_tokens == -1:  # Cache de prompt
            prompt_tokens = 0
        
        # Criar objeto de métricas
        metrics = BenchmarkMetrics(
            model_name=model_name,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            response_text=response_text,
            total_duration_ns=api_response.get("total_duration", 0),
            load_duration_ns=api_response.get("load_duration", 0),
            prompt_eval_duration_ns=api_response.get("prompt_eval_duration", 0),
            eval_duration_ns=api_response.get("eval_duration", 0),
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            created_at=created_at,
            processed_at=datetime.now()
        )
        
        return metrics
        
    except Exception as e:
        print(f"Erro ao extrair métricas: {e}")
        return None


def run_single_benchmark(
    client: OllamaAPIClient,
    model_name: str,
    prompt_entry: DatasetEntry,
    use_chat_api: bool = True
) -> Optional[BenchmarkMetrics]:
    
    prompt_id = prompt_entry.id
    prompt_text = prompt_entry.prompt
    
    try:
        if use_chat_api:
            # Usar API de chat
            messages = [{"role": "user", "content": prompt_text}]
            api_response = client.chat(
                model_name, 
                messages, 
                stream=False,
                options=MODEL_CONFIG.get("options", {})
            )
        else:
            # Usar API de generate
            api_response = client.generate(
                model_name,
                prompt_text,
                stream=False,
                options=MODEL_CONFIG.get("options", {})
            )
        
        # Extrair métricas
        metrics = extract_benchmark_metrics(
            api_response, model_name, prompt_id, prompt_text
        )
        
        return metrics
        
    except Exception as e:
        print(f"Erro no benchmark para prompt '{prompt_id}': {e}")
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
                    print(f"Erro no JSON linha {line_num}: {e}")
                except ValidationError as e:
                    print(f"Erro de validação linha {line_num}: {e}")
        print(f"Carregados {len(prompts)} prompts de {file_path}")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        sys.exit(1)
    return prompts


def save_output_entry(entry: OutputEntry, file_path: str):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json_line = entry.model_dump_json()
            f.write(json_line + '\n')
    except Exception as e:
        print(f"Erro ao salvar saída: {e}")

def calculate_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    
    sorted_vals = sorted(values)
    index = math.ceil(0.95 * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(index, len(sorted_vals) - 1))]


def create_output_entry(
    prompt_entry: DatasetEntry,
    metrics: BenchmarkMetrics,
    use_chat_api: bool = True
) -> OutputEntry:
    
    # Metricas básicas para compatibilidade
    basic_metrics = {
        "prompt_tokens": metrics.prompt_tokens,
        "response_tokens": metrics.response_tokens,
        "total_tokens": metrics.total_tokens,
        "model_load_time_seconds": metrics.load_duration_sec,
        "prompt_eval_time_seconds": metrics.prompt_eval_duration_sec,
        "response_time_seconds": metrics.eval_duration_sec,
        "total_time_seconds": metrics.total_duration_sec,
        "prompt_tokens_per_second": metrics.prompt_tokens_per_sec,
        "response_tokens_per_second": metrics.response_tokens_per_sec,
        "total_tokens_per_second": metrics.total_tokens_per_sec
    }
    
    return OutputEntry(
        id=prompt_entry.id,
        model=metrics.model_name,
        prompt=metrics.prompt_text,
        response=metrics.response_text,
        metrics=basic_metrics,
        benchmark_metrics=metrics.to_dict(),  # Todas as métricas detalhadas
        metadata={
            **prompt_entry.metadata,
            "model_config": MODEL_CONFIG,
            "api_used": "chat" if use_chat_api else "generate",
        }
    )



def main():
    print("BENCHMARK DE MODELOS OLLAMA")

    use_chat_api = True
    api_type = "chat"

    print(f"API utilizada: {api_type}")
    print(f"Dataset de entrada: {INPUT_DATASET_PATH}")
    print(f"Dataset de saída: {OUTPUT_DATASET_PATH}")

    # Carregar prompts
    prompts = load_prompts_from_dataset(INPUT_DATASET_PATH)
    print(f"Total de prompts a processar: {len(prompts)}")

    # Inicializar cliente API
    client = OllamaAPIClient(OLLAMA_API_BASE_URL)

    # Buscar modelos disponíveis
    model_names = client.list_models()
    if not model_names:
        print("Nenhum modelo encontrado, usando modelos padrão")
        model_names = DEFAULT_MODELS

    print(f"Modelos a avaliar: {model_names}")

    # Limpar arquivo de saída
    try:
        open(OUTPUT_DATASET_PATH, 'w').close()
        print(f"Arquivo de saída limpo: {OUTPUT_DATASET_PATH}")
    except:
        print(f"Criando novo arquivo de saída: {OUTPUT_DATASET_PATH}")

    # Executar benchmarks
    all_metrics: List[BenchmarkMetrics] = []

    for model_name in model_names:
        print(f" Modelo: {model_name}")

        model_metrics: List[BenchmarkMetrics] = []

        for prompt_entry in tqdm(prompts, desc=f"Modelo: {model_name}"):
            metrics = run_single_benchmark(
                client, model_name, prompt_entry, use_chat_api
            )

            if metrics:
                model_metrics.append(metrics)

                output_entry = create_output_entry(
                    prompt_entry, metrics, use_chat_api
                )
                save_output_entry(output_entry, OUTPUT_DATASET_PATH)

        all_metrics.extend(model_metrics)

        if model_metrics:
            print(f"\n Modelo {model_name}: {len(model_metrics)}/{len(prompts)} completos")
        else:
            print(f"\n Modelo {model_name}: 0/{len(prompts)} completos")

    # Resumo final
    print("BENCHMARK FINALIZADO")

    total_runs = len(all_metrics)
    total_expected = len(model_names) * len(prompts)
    success_rate = (total_runs / total_expected * 100) if total_expected > 0 else 0

    print(f"Total de execuções: {total_runs}/{total_expected} ({success_rate:.1f}%)")
    print(f"Dataset de saída: {OUTPUT_DATASET_PATH}")

    if all_metrics:
        print_metrics_summary(all_metrics)


    # Salvar resumo
    try:
        summary_file = OUTPUT_DATASET_PATH.replace('.jsonl', '_summary.json')
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "models_evaluated": model_names,
            "api_used": api_type,
            "total_successful_runs": total_runs,
            "success_rate_percent": success_rate,
            "model_config": MODEL_CONFIG
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\nResumo salvo em: {summary_file}")
    except Exception as e:
        print(f"\nErro ao salvar resumo: {e}")


if __name__ == "__main__":
    main()