# run_vllm_server.py
"""
vLLM 서버를 GPU 0에 띄우는 코드
모델 경로: huggingface/a
"""

import os
import subprocess
import sys

def run_vllm_server(
    model_path: str = "huggingface/a",
    gpu_id: int = 0,
    port: int = 8000,
    host: str = "0.0.0.0"
):
    """vLLM OpenAI 호환 서버를 실행합니다."""
    
    # GPU 지정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # vLLM 서버 실행 명령어
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]
    
    print(f"vLLM 서버를 시작합니다...")
    print(f"모델: {model_path}")
    print(f"GPU: {gpu_id}")
    print(f"주소: http://{host}:{port}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 서버 실행")
    parser.add_argument("--model", type=str, default="huggingface/a", help="모델 경로")
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU ID")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    
    args = parser.parse_args()
    
    run_vllm_server(
        model_path=args.model,
        gpu_id=args.gpu,
        port=args.port
    )