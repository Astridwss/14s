import os
import requests

class SceneDownloader:
    @staticmethod
    def download_file(url: str, task_id: str) -> str:
        """
        将 URL 下载到本地 ./scenarios/{task_id}/scene.json 中
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        base_dir = os.path.join(os.environ.get("PROJECT_ROOT", "."), "scenarios")
        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        local_scene_path = os.path.join(task_dir, "scene.json")
            
        try:
            print(f"[SceneDownloader] 正在从 {url} 下载场景文件...")
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status() 
            
            with open(local_scene_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: 
                        f.write(chunk)
                
            print(f"[SceneDownloader] 下载完成: {local_scene_path}")
            
        # --- 精准捕获：仅拦截网络 HTTP 报错和磁盘 I/O 写入报错 ---
        except (requests.RequestException, OSError) as e:
            print(f"[SceneDownloader] 下载或保存场景文件失败 [网络/磁盘异常]: {e}")
            raise  # 强烈推荐：直接用裸 raise，完美保留原始异常报错行号和堆栈
            
        return local_scene_path