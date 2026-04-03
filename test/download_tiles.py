#离线下载地球皮肤
import os
import urllib.request
import time

# 存放目录
BASE_DIR = os.path.join("static", "offline_maps")

#  替换为 NASA 官方的 Blue Marble XYZ 接口
URL_TEMPLATE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg"

def download_tiles(max_zoom=4):  
    # 💡 建议先下 0~4 级（共340张图片，几秒钟下完）测试一下。
    # 想要更清晰的全球图，可以把 max_zoom 改成 8（大约 87000 张图，需要下半小时）。
    print(f"开始下载 0 到 {max_zoom} 级的 NASA Blue Marble 卫星瓦片...")
    total_downloaded = 0
    
    for z in range(max_zoom + 1):
        num_tiles = 2 ** z
        for x in range(num_tiles):
            for y in range(num_tiles):
                dir_path = os.path.join(BASE_DIR, str(z), str(x))
                os.makedirs(dir_path, exist_ok=True)
                
                #  注意后缀改成了 .jpeg
                file_path = os.path.join(dir_path, f"{y}.jpeg")
                
                if not os.path.exists(file_path):
                    url = URL_TEMPLATE.format(z=z, x=x, y=y)
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                            out_file.write(response.read())
                        total_downloaded += 1
                        print(f"成功下载: {z}/{x}/{y}.jpeg")
                        time.sleep(0.05) 
                    except Exception as e:
                        print(f"下载失败 {url}: {e}")

    print(f"\n🎉 下载完成共下载了 {total_downloaded} 张瓦片。")

if __name__ == "__main__":
    download_tiles(max_zoom=4)