# utils/tv_display.py
import pygame
import sys
import os

if not hasattr(sys, 'frozen'):
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class TwoDimDisplay:
    def __init__(self, width=1200, height=800, fps=30, fixed_trajectories=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("防空多智能体 RL 战术态势监控中心")
        
        self.clock = pygame.time.Clock()
        self.fps = fps
        font_name = "simhei" if "simhei" in pygame.font.get_fonts() else pygame.font.get_default_font()
        self.font = pygame.font.SysFont(font_name, 16)
        self.large_font = pygame.font.SysFont(font_name, 24)
        
        self.fixed_trajectories = fixed_trajectories if fixed_trajectories else {}
        
        self.colors = {
            "bg": (10, 15, 25),
            "grid": (30, 40, 60),
            "radar": (0, 255, 255),
            "target": (255, 50, 50),
            "target_path": (120, 40, 40),
            "link": (50, 255, 50),
            "text": (200, 200, 200)
        }

        # 视口和雷达的“地桩”标志位
        self.view_initialized = False
        self.bbox = {"min_lon": 100, "max_lon": 130, "min_lat": 10, "max_lat": 40}
        self.fixed_radars = [] # 缓存第一帧的雷达位置

    def _lock_viewport_and_radars(self, radars):
        """ 修复：在第一帧打地桩，过滤异常坐标 (如 0,0)，并自适应视口比例"""
        # 1. 过滤掉绝对值小于 0.1 的异常坐标（避免 0,0 扯烂画布）
        all_lons = [r['lon'] for r in radars if abs(r['lon']) > 0.1]
        all_lats = [r['lat'] for r in radars if abs(r['lat']) > 0.1]
        
        # 把轨迹的极值也加进来
        for track in self.fixed_trajectories.values():
            for pt in track:
                if abs(pt[0]) > 0.1 and abs(pt[1]) > 0.1:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])
                
        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            
            # 2. 自适应边距：取真实经纬度跨度的 10% 作为留白，而不是死板的 1.0 度
            margin_lon = max(0.01, (max_lon - min_lon) * 0.1) # 最小留 0.01 度(约 1km)
            margin_lat = max(0.01, (max_lat - min_lat) * 0.1)
            
            self.bbox["min_lon"] = min_lon - margin_lon
            self.bbox["max_lon"] = max_lon + margin_lon
            self.bbox["min_lat"] = min_lat - margin_lat
            self.bbox["max_lat"] = max_lat + margin_lat
            
        # 永远缓存第一帧的雷达位置
        self.fixed_radars = radars.copy()
        self.view_initialized = True

    def _geo_to_pixel(self, lon, lat):
        lon_range = self.bbox["max_lon"] - self.bbox["min_lon"]
        lat_range = self.bbox["max_lat"] - self.bbox["min_lat"]
        if lon_range == 0: lon_range = 1
        if lat_range == 0: lat_range = 1

        pad = 50 
        x = pad + (lon - self.bbox["min_lon"]) / lon_range * (self.width - 2 * pad)
        y = self.height - pad - (lat - self.bbox["min_lat"]) / lat_range * (self.height - 2 * pad)
        return int(x), int(y)

    def render_step(self, time_step: int, radars: list, targets: list, links: list):
        #  第一帧初始化：打下地桩，锁死相机和雷达
        if not self.view_initialized:
            self._lock_viewport_and_radars(radars)

        # 如果有目标飞出了当前屏幕边界，动态把相机框撑大
        for t in targets:
            if abs(t['lon']) > 0.1 and abs(t['lat']) > 0.1: # 忽略失效目标的0,0坐标
                if t['lon'] < self.bbox["min_lon"]: self.bbox["min_lon"] = t['lon'] - 0.05
                if t['lon'] > self.bbox["max_lon"]: self.bbox["max_lon"] = t['lon'] + 0.05
                if t['lat'] < self.bbox["min_lat"]: self.bbox["min_lat"] = t['lat'] - 0.05
                if t['lat'] > self.bbox["max_lat"]: self.bbox["max_lat"] = t['lat'] + 0.05
        # ==================================

        self.screen.fill(self.colors["bg"])

        # ==================== 画网格 ====================
        for i in range(5):
            x = self.width * (i / 4.0)
            y = self.height * (i / 4.0)
            pygame.draw.line(self.screen, self.colors["grid"], (x, 0), (x, self.height), 1)
            pygame.draw.line(self.screen, self.colors["grid"], (0, y), (self.width, y), 1)

        # ==================== Layer 1: 画航迹 (背景) ====================
        for tid, points in self.fixed_trajectories.items():
            if len(points) < 2: continue
            pixel_points = [self._geo_to_pixel(p[0], p[1]) for p in points]
            pygame.draw.lines(self.screen, self.colors["target_path"], False, pixel_points, 1)

        pos_dict = {}

        # ==================== Layer 2: 画固定的地基雷达 ====================
        #  无论传入的 radars 怎么变，我们只画一开始缓存的 fixed_radars
        for r in self.fixed_radars:
            px, py = self._geo_to_pixel(r['lon'], r['lat'])
            pos_dict[r['id']] = (px, py)
            pygame.draw.rect(self.screen, self.colors["radar"], (px-6, py-6, 12, 12), 2)
            label = self.font.render(f"R:{r['id']}", True, self.colors["radar"])
            self.screen.blit(label, (px + 10, py - 10))

        # ==================== Layer 3: 画移动的导弹 ====================
        for t in targets:
            px, py = self._geo_to_pixel(t['lon'], t['lat'])
            pos_dict[t['id']] = (px, py)
            pygame.draw.circle(self.screen, self.colors["target"], (px, py), 5)
            alt_km = t.get('alt', 0) / 1000.0
            label = self.font.render(f"T:{t['id']} (H:{alt_km:.1f}km)", True, self.colors["target"])
            self.screen.blit(label, (px + 8, py + 8))

        # ==================== Layer 4: 画动作连线 ====================
        for radar_id, target_id in links:
            if radar_id in pos_dict and target_id in pos_dict:
                pygame.draw.line(self.screen, self.colors["link"], pos_dict[radar_id], pos_dict[target_id], 2)

        # 画时间
        time_text = self.large_font.render(f"仿真时间: {time_step} s", True, self.colors["text"])
        self.screen.blit(time_text, (20, 20))
        
        pygame.display.flip()
        self.clock.tick(self.fps)

class LiveObserver:
    """强化学习训练时的态势tv_display观察者，彻底解耦渲染逻辑"""
    def __init__(self, watch_freq=20, fps=30):
        self.watch_freq = watch_freq
        self.fps = fps
        self.display = None
        self.is_watching = False

    def check_and_start(self, episode_idx, epsilon, fixed_trajectories=None):
        """回合开始前：判断是否需要创建屏幕并拉起tv_display"""
        #  接受外部注入的固定轨迹数据
        self.is_watching = (episode_idx % self.watch_freq == 0)
        if self.is_watching:
            print(f"\n📺 [Live] 正在为您tv_display第 {episode_idx} 局训练实况 (eps={epsilon:.2f})...")
            self.display = TwoDimDisplay(fps=self.fps, fixed_trajectories=fixed_trajectories)
        return self.is_watching

    def render_callback(self, info):
        """推演过程中：塞给 Rollout 的回调函数，负责解剖数据"""
        # 处理退出事件，防止看tv_display时卡死
        if self.is_watching:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
        if not self.is_watching or self.display is None:
            return
            
        raw_obs = info.get('raw_obs')
        actions = info.get('agent_actions_list', [])
        if not raw_obs: 
            return
            
        time_step = int(raw_obs.current_time)
        radars = [{'id': k, 'lon': v.longitude, 'lat': v.latitude} for k, v in raw_obs.dict_equip_state.items()]
        targets = [{'id': k, 'lon': v.longitude, 'lat': v.latitude, 'alt': v.altitude} for k, v in raw_obs.dict_system_track.items()]
        links = [(cmd.str_equip_id, cmd.str_target_id) for cmd in actions if cmd.str_target_id != ""]
        
        self.display.render_step(time_step, radars, targets, links)

    def close(self):
        """回合结束后：销毁屏幕，释放资源"""
        if self.is_watching:
            pygame.quit()
            self.display = None
            self.is_watching = False    