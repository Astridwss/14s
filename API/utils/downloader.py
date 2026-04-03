# API/utils/downloader.py
import os
import requests

class SceneDownloader:
    @staticmethod
    def download_file(url: str, task_id: str) -> str:
        """
        将 URL 下载到本地 ./scenarios/{task_id}/scene.json 中
        """
        # 1. 安全防护：如果平台下发的 URL 没带协议头，自动补全 http://
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        base_dir = os.path.join(os.environ.get("PROJECT_ROOT", "."), "scenarios")
        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        local_scene_path = os.path.join(task_dir, "scene.json")
            
        try:
            print(f"[SceneDownloader] 正在从 {url} 下载场景文件...")
            
            # 2. 开启 stream=True 进行流式请求，并将超时放宽到 60 秒
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()  # 检查 HTTP 状态码，如果接口报错 404/500 会跳转到 except
            
            # 3. 分块读取并写入文件（极其安全的工业级大文件下载写法）
            with open(local_scene_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # 过滤掉保持连接的 keep-alive 新行
                        f.write(chunk)
                
            print(f"[SceneDownloader] 下载完成: {local_scene_path}")
            
        except Exception as e:
            print(f"[SceneDownloader] 下载场景文件失败: {e}")
            raise e  # 下载失败必须抛出异常，阻止后续的训练/推演启动
            
        return local_scene_path