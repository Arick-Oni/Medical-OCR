import base64
import io
import time
import requests
from typing import Optional, Dict, Any
from PIL import Image

DEFAULT_NGROK_URL = "https://banjo-mammal-gradient.ngrok-free.dev"

class NgrokJobClient:
    def __init__(self, base_url: str = DEFAULT_NGROK_URL):
        self.base_url = base_url.rstrip("/")

    def set_base_url(self, new_url: str):
        if new_url:
            self.base_url = new_url.rstrip("/")

    @staticmethod
    def encode_image(image_input) -> str:
        """Converts PIL.Image, numpy ndarray, or file path/bytes to base64 string."""
        if isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def check_health(self, test_url: Optional[str] = None) -> Dict[str, Any]:
        """Tests the ngrok tunnel endpoint availability via POST /submit."""
        target_base = (test_url or self.base_url).rstrip("/")
        try:
            start = time.time()
            url = f"{target_base}/submit"
            payload = {
                "model": "qwen2.5vl:7b",
                "prompt": "ping",
                "options": {"num_predict": 1}
            }
            res = requests.post(url, json=payload, timeout=12)
            latency = round((time.time() - start) * 1000, 2)
            if res.status_code == 200 and "job_id" in res.json():
                return {
                    "status": "online",
                    "url": target_base,
                    "latency_ms": latency,
                    "message": "Connected to Colab Ngrok job queue"
                }
            return {
                "status": "warning",
                "url": target_base,
                "latency_ms": latency,
                "message": f"Server responded with HTTP {res.status_code}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "offline",
                "url": target_base,
                "latency_ms": None,
                "message": f"Cannot connect to Ngrok tunnel: {str(e)}"
            }

    def submit_job(
        self,
        model: str,
        prompt: str,
        image_b64: Optional[str] = None,
        system: Optional[str] = None,
        json_schema: Optional[dict] = None,
        options: Optional[dict] = None
    ) -> str:
        """Submits a job to /submit endpoint and returns job_id."""
        url = f"{self.base_url}/submit"
        payload = {
            "model": model,
            "prompt": prompt,
            "image_b64": image_b64,
            "system": system,
            "json_schema": json_schema,
            "options": options or {"temperature": 0.0}
        }
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        job_id = data.get("job_id")
        if not job_id:
            raise ValueError(f"Missing job_id in response: {data}")
        return job_id

    def poll_result(self, job_id: str, timeout_sec: int = 600, poll_interval: float = 1.0) -> str:
        """Polls /result/{job_id} until status is done or error."""
        url = f"{self.base_url}/result/{job_id}"
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            try:
                res = requests.get(url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status")
                    if status == "done":
                        return data.get("result", "")
                    elif status == "error":
                        raise RuntimeError(f"Colab Job Error: {data.get('result')}")
                    elif status == "not_found":
                        pass
            except requests.exceptions.RequestException:
                pass
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} timed out after {timeout_sec} seconds")

    def run_inference(
        self,
        model: str,
        prompt: str,
        image_b64: Optional[str] = None,
        system: Optional[str] = None,
        json_schema: Optional[dict] = None,
        options: Optional[dict] = None,
        timeout_sec: int = 600
    ) -> tuple[str, float]:
        """Convenience method to submit and poll in one step."""
        start_time = time.time()
        job_id = self.submit_job(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            system=system,
            json_schema=json_schema,
            options=options
        )
        result_text = self.poll_result(job_id=job_id, timeout_sec=timeout_sec)
        elapsed = round(time.time() - start_time, 3)
        return result_text, elapsed
