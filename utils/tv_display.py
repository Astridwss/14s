# utils/tv_display.py
"""2D 战术态势可视化 —— 防空多智能体 RL 训练/推演实时监控。

性能优化:
  - 雷达标签预渲染 + Surface 缓存，避免每帧 font.render()
  - 轨迹点像素坐标一次性转换并缓存
  - 视口外实体跳过渲染
  - 连线数上限截断，防止 250 agent 时画面混乱

交互:
  - 滚轮       缩放（以鼠标为中心）
  - 左键拖动    平移
  - R          复位视图
  - SPACE      暂停/恢复
  - +/-        加速/减速
  - T          切换目标尾迹
  - L          切换动作连线
  - G          切换分组着色
  - ESC        退出可视化
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

if not hasattr(sys, 'frozen'):
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


# ============================================================
# 调色板
# ============================================================
PALETTE = {
    "bg":            (8, 12, 22),
    "panel_bg":      (14, 20, 35),
    "panel_border":  (40, 50, 70),
    "grid":          (22, 30, 48),
    "grid_major":    (35, 45, 65),
    "radar":         (0, 220, 240),
    "radar_dim":     (0, 120, 140),
    "target":        (255, 55, 55),
    "target_glow":   (255, 100, 100, 80),
    "target_trail":  (140, 50, 50),
    "link_active":   (60, 255, 80),
    "link_idle":     (80, 160, 90),
    "text":          (200, 210, 220),
    "text_dim":      (120, 130, 145),
    "text_highlight":(255, 255, 255),
    "warning":       (255, 180, 40),
    "accent_blue":   (70, 140, 255),
    "accent_green":  (70, 220, 120),
}

# H-QMIX 分组色板 (最多支持 30 组)
GROUP_COLORS = [
    (0, 220, 240), (240, 180, 0), (180, 100, 255),
    (0, 200, 130), (255, 130, 50), (100, 180, 255),
    (220, 80, 180), (80, 220, 220), (255, 200, 80),
    (140, 220, 80), (200, 80, 120), (80, 160, 255),
    (200, 160, 60), (60, 200, 160), (220, 100, 200),
    (100, 200, 100), (200, 140, 80), (80, 140, 220),
    (180, 200, 60), (160, 80, 200), (80, 200, 180),
    (220, 120, 140), (140, 180, 220), (200, 220, 100),
    (160, 120, 200), (120, 200, 140), (220, 160, 100),
    (100, 160, 200), (180, 140, 160), (140, 200, 200),
]


# ============================================================
# TwoDimDisplay
# ============================================================

class TwoDimDisplay:
    """2D 战术态势渲染器。

    Args:
        width, height:   窗口尺寸
        fps:             目标帧率
        fixed_trajectories: {target_id: [(lon, lat), ...]}  目标预设航迹
        group_assignments: H-QMIX 分组, List[List[int]],
                           用于按组着色雷达 (可选)
    """

    # 右侧信息面板宽度
    PANEL_WIDTH = 220

    def __init__(
        self,
        width: int = 1400,
        height: int = 800,
        fps: int = 30,
        fixed_trajectories: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        group_assignments: Optional[List[List[int]]] = None,
        radar_keys: Optional[List[str]] = None,
    ):
        pygame.init()
        self.width = width
        self.height = height
        self.map_width = width - self.PANEL_WIDTH
        self.map_height = height
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("防空多智能体 RL 战术态势监控中心")

        self.clock = pygame.time.Clock()
        self.fps = fps
        self._base_fps = fps

        # ---- 字体 ----
        font_name = (
            "simhei" if "simhei" in pygame.font.get_fonts()
            else pygame.font.get_default_font()
        )
        self.font_sm = pygame.font.SysFont(font_name, 13)
        self.font_md = pygame.font.SysFont(font_name, 16)
        self.font_lg = pygame.font.SysFont(font_name, 22)
        self.font_mono = pygame.font.SysFont("consolas", 13)

        # ---- 固定航迹数据 ----
        self.fixed_trajectories = fixed_trajectories or {}
        self._traj_pixel_cache: Dict[str, List[Tuple[int, int]]] = {}

        # ---- 分组着色 ----
        self.group_assignments = group_assignments  # List[List[int]]
        self.radar_keys = radar_keys or []           # List[str], 与 group 索引对齐
        self._radar_to_group: Dict[str, int] = {}    # radar_id → group_index
        self._radar_to_group_color: Dict[str, Tuple[int, int, int]] = {}
        if group_assignments and radar_keys:
            for g_idx, group in enumerate(group_assignments):
                color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
                for agent_idx in group:
                    if agent_idx >= 0 and agent_idx < len(radar_keys):
                        rid = radar_keys[agent_idx]
                        self._radar_to_group[rid] = g_idx
                        self._radar_to_group_color[rid] = color

        # ---- 视口 & 地桩 ----
        self.view_initialized = False
        self.bbox = {"min_lon": 100, "max_lon": 130, "min_lat": 10, "max_lat": 40}
        self.fixed_radars: List[dict] = []

        # ---- 预渲染缓存 ----
        self._radar_label_cache: Dict[str, pygame.Surface] = {}
        self._target_label_cache: Dict[str, pygame.Surface] = {}
        self._panel_surface: Optional[pygame.Surface] = None

        # ---- 目标尾迹 (最近 N 个位置) ----
        self._target_trail: Dict[str, List[Tuple[float, float]]] = {}
        self._max_trail_len = 20

        # ---- 交互状态 ----
        self._paused = False
        self._show_trails = True
        self._show_links = True
        self._show_groups = True
        self._frame_count = 0
        self._total_frames = 0

        # ---- 性能统计 ----
        self._render_time_ms = 0.0

        # ---- 缩放 / 平移 (滚轮缩放、左键拖动平移) ----
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._dragging = False
        self._drag_start = (0, 0)
        self._pan_start = (0.0, 0.0)

    # ============================================================
    # 视口锁定
    # ============================================================

    def _lock_viewport_and_radars(self, radars: List[dict]) -> None:
        """第一帧：固定视口范围 + 缓存雷达位置 + 预计算轨迹像素坐标。"""
        all_lons = [r['lon'] for r in radars if abs(r['lon']) > 0.1]
        all_lats = [r['lat'] for r in radars if abs(r['lat']) > 0.1]

        for track in self.fixed_trajectories.values():
            for pt in track:
                if abs(pt[0]) > 0.1 and abs(pt[1]) > 0.1:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])

        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            margin_lon = max(0.02, (max_lon - min_lon) * 0.12)
            margin_lat = max(0.02, (max_lat - min_lat) * 0.12)
            self.bbox["min_lon"] = min_lon - margin_lon
            self.bbox["max_lon"] = max_lon + margin_lon
            self.bbox["min_lat"] = min_lat - margin_lat
            self.bbox["max_lat"] = max_lat + margin_lat

        self.fixed_radars = radars.copy()
        self.view_initialized = True

        # ---- 预计算：轨迹点 → 像素坐标 ----
        self._rebuild_traj_cache()

        # ---- 预渲染：所有固定雷达标签 ----
        self._radar_label_cache.clear()
        for r in self.fixed_radars:
            rid = str(r.get('id', ''))
            color = self._radar_to_group_color.get(rid, PALETTE["radar"])
            self._radar_label_cache[rid] = self.font_sm.render(
                rid, True, color,
            )

    # ============================================================
    # 坐标转换
    # ============================================================

    def _geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        lon_range = self.bbox["max_lon"] - self.bbox["min_lon"] or 1.0
        lat_range = self.bbox["max_lat"] - self.bbox["min_lat"] or 1.0
        pad = 40
        base_x = pad + (lon - self.bbox["min_lon"]) / lon_range * (self.map_width - 2 * pad)
        base_y = (self.map_height - pad
                  - (lat - self.bbox["min_lat"]) / lat_range * (self.map_height - 2 * pad))
        # 缩放（绕地图中心）+ 平移
        cx = self.map_width / 2.0
        cy = self.map_height / 2.0
        x = (base_x - cx) * self._zoom + cx + self._pan_x
        y = (base_y - cy) * self._zoom + cy + self._pan_y
        return int(x), int(y)

    def _is_on_screen(self, px: int, py: int, margin: int = 20) -> bool:
        return (-margin <= px <= self.map_width + margin
                and -margin <= py <= self.map_height + margin)

    # ============================================================
    # 主渲染入口
    # ============================================================

    def render_step(
        self,
        time_step: int,
        radars: list,
        targets: list,
        links: list,
        extra_info: Optional[dict] = None,
        satellites: Optional[list] = None,
    ) -> None:
        """每步调用一次，绘制当前帧。

        Args:
            time_step: 仿真时间 (秒)
            radars:    [{'id': str, 'lon': float, 'lat': float}, ...] 地面雷达 (type==1)
            targets:   [{'id': str, 'lon': float, 'lat': float, 'alt': float}, ...]
            links:     [(radar_id, target_id), ...]  当前动作连线
            extra_info: 可选扩展信息 (episode, reward, n_actions 等)
            satellites: [{'id': str, 'lon': float, 'lat': float}, ...] 卫星 (不参与视口锁定)
        """
        t0 = pygame.time.get_ticks()

        # ---- 处理键盘事件 ----
        self._handle_events()

        if self._paused:
            self.clock.tick(self.fps)
            return

        # ---- 首帧初始化 ----
        if not self.view_initialized:
            self._lock_viewport_and_radars(radars)

        # ---- 动态扩展视口 ----
        self._expand_viewport_if_needed(targets)

        # ---- 更新目标尾迹 ----
        if self._show_trails:
            for t in targets:
                tid = str(t.get('id', ''))
                lon, lat = t.get('lon', 0), t.get('lat', 0)
                if abs(lon) > 0.1 and abs(lat) > 0.1:
                    if tid not in self._target_trail:
                        self._target_trail[tid] = []
                    self._target_trail[tid].append((lon, lat))
                    if len(self._target_trail[tid]) > self._max_trail_len:
                        self._target_trail[tid] = (
                            self._target_trail[tid][-self._max_trail_len:]
                        )

        # ---- 绘制 ----
        self._draw_background()
        self._draw_grid()
        self._draw_trajectories()
        if self._show_trails:
            self._draw_target_trails()
        pos_dict = self._draw_radars()
        if satellites:
            pos_dict.update(self._draw_satellites(satellites))
        target_pos = self._draw_targets(targets)
        pos_dict.update(target_pos)
        if self._show_links:
            self._draw_links(links, pos_dict)
        self._draw_time_overlay(time_step)
        self._draw_info_panel(time_step, radars, targets, links, extra_info,
                              n_satellites=len(satellites or []))

        pygame.display.flip()
        self.clock.tick(self.fps)

        self._total_frames += 1
        self._render_time_ms = (pygame.time.get_ticks() - t0)

    def render_idle(self, episode_idx: int, max_episodes: int) -> None:
        """非渲染局占位画面 —— 只绘制等待提示，防止窗口黑屏。

        事件处理统一由 LiveObserver._pump_events() 负责（避免在此 sys.exit）。
        """
        self._draw_background()
        msg = f"等待渲染局  {episode_idx}/{max_episodes}"
        txt = self.font_lg.render(msg, True, PALETTE["text_highlight"])
        self.screen.blit(
            txt,
            (self.map_width // 2 - txt.get_width() // 2,
             self.map_height // 2 - 20),
        )
        sub = "SPACE 暂停 / ESC 退出可视化"
        stxt = self.font_md.render(sub, True, PALETTE["text_dim"])
        self.screen.blit(
            stxt,
            (self.map_width // 2 - stxt.get_width() // 2,
             self.map_height // 2 + 10),
        )
        pygame.display.flip()

    # ============================================================
    # 事件处理
    # ============================================================

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if self.handle_event(event):
                pygame.quit()
                sys.exit()

    def handle_event(self, event) -> bool:
        """处理单个 pygame 事件，返回 True 表示应退出程序。

        统一由 TwoDimDisplay 处理，LiveObserver._pump_events 复用，
        保证渲染局 / 非渲染局的交互（缩放、平移、暂停等）一致。
        """
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return True
            elif event.key == pygame.K_SPACE:
                self._paused = not self._paused
                print(f"[Display] {'暂停' if self._paused else '运行'}")
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.fps = min(120, self.fps + 10)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.fps = max(1, self.fps - 10)
            elif event.key == pygame.K_t:
                self._show_trails = not self._show_trails
            elif event.key == pygame.K_l:
                self._show_links = not self._show_links
            elif event.key == pygame.K_g:
                self._show_groups = not self._show_groups
                # 重建雷达标签缓存（颜色变化）
                self._radar_label_cache.clear()
                group_on = self._show_groups and bool(self._radar_to_group_color)
                for r in self.fixed_radars:
                    rid = str(r.get('id', ''))
                    color = (
                        self._radar_to_group_color.get(rid, PALETTE["radar"])
                        if group_on else PALETTE["radar"]
                    )
                    self._radar_label_cache[rid] = self.font_sm.render(
                        rid, True, color,
                    )
            elif event.key == pygame.K_r:
                self._reset_view()

        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.1 if event.y > 0 else (1.0 / 1.1)
            # MOUSEWHEEL 事件无 .pos，用 pygame.mouse.get_pos() 取当前鼠标位置
            self._apply_zoom(factor, pygame.mouse.get_pos())

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._dragging = True
            self._drag_start = event.pos
            self._pan_start = (self._pan_x, self._pan_y)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False

        elif event.type == pygame.MOUSEMOTION and self._dragging:
            dx = event.pos[0] - self._drag_start[0]
            dy = event.pos[1] - self._drag_start[1]
            self._pan_x = self._pan_start[0] + dx
            self._pan_y = self._pan_start[1] + dy
            self._rebuild_traj_cache()

        return False

    def _apply_zoom(self, factor: float, mouse_pos: Tuple[int, int]) -> None:
        """以鼠标位置为中心缩放，保持鼠标指向的地理点不动。"""
        old = self._zoom
        new = max(0.2, min(20.0, old * factor))
        if abs(new - old) < 1e-9:
            return
        cx = self.map_width / 2.0
        cy = self.map_height / 2.0
        ratio = new / old
        mx, my = mouse_pos
        self._pan_x = mx - cx - (mx - cx - self._pan_x) * ratio
        self._pan_y = my - cy - (my - cy - self._pan_y) * ratio
        self._zoom = new
        self._rebuild_traj_cache()

    def _reset_view(self) -> None:
        """复位缩放与平移。"""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._rebuild_traj_cache()

    def _rebuild_traj_cache(self) -> None:
        """按当前缩放/平移重建轨迹像素缓存。"""
        self._traj_pixel_cache.clear()
        for tid, points in self.fixed_trajectories.items():
            if len(points) >= 2:
                self._traj_pixel_cache[tid] = [
                    self._geo_to_pixel(p[0], p[1]) for p in points
                ]

    # ============================================================
    # 视口动态扩展
    # ============================================================

    def _expand_viewport_if_needed(self, targets: list) -> None:
        changed = False
        for t in targets:
            lon, lat = t.get('lon', 0), t.get('lat', 0)
            if abs(lon) < 0.1 or abs(lat) < 0.1:
                continue
            if lon < self.bbox["min_lon"]:
                self.bbox["min_lon"] = lon - 0.05; changed = True
            if lon > self.bbox["max_lon"]:
                self.bbox["max_lon"] = lon + 0.05; changed = True
            if lat < self.bbox["min_lat"]:
                self.bbox["min_lat"] = lat - 0.05; changed = True
            if lat > self.bbox["max_lat"]:
                self.bbox["max_lat"] = lat + 0.05; changed = True
        if changed:
            # 视口变化时重建轨迹像素缓存
            self._rebuild_traj_cache()

    # ============================================================
    # 各图层绘制
    # ============================================================

    def _draw_background(self) -> None:
        self.screen.fill(PALETTE["bg"])
        # 右侧面板背景
        panel_rect = pygame.Rect(self.map_width, 0, self.PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, PALETTE["panel_bg"], panel_rect)
        pygame.draw.line(
            self.screen, PALETTE["panel_border"],
            (self.map_width, 0), (self.map_width, self.height), 2,
        )

    def _draw_grid(self) -> None:
        """自适应经纬度网格。"""
        # 经线
        lon_span = self.bbox["max_lon"] - self.bbox["min_lon"]
        lat_span = self.bbox["max_lat"] - self.bbox["min_lat"]
        n_lon_lines = max(4, int(lon_span / 2))
        n_lat_lines = max(3, int(lat_span / 2))

        for i in range(n_lon_lines + 1):
            frac = i / n_lon_lines
            x = int(40 + frac * (self.map_width - 80))
            color = PALETTE["grid_major"] if i % max(1, n_lon_lines // 4) == 0 else PALETTE["grid"]
            pygame.draw.line(self.screen, color, (x, 0), (x, self.map_height), 1)
            # 标注经度
            lon_val = self.bbox["min_lon"] + frac * lon_span
            label = self.font_sm.render(f"{lon_val:.1f}°", True, PALETTE["text_dim"])
            self.screen.blit(label, (x + 2, self.map_height - 18))

        for i in range(n_lat_lines + 1):
            frac = i / n_lat_lines
            y = int(self.map_height - 40 - frac * (self.map_height - 80))
            color = PALETTE["grid_major"] if i % max(1, n_lat_lines // 4) == 0 else PALETTE["grid"]
            pygame.draw.line(self.screen, color, (40, y), (self.map_width - 40, y), 1)
            lat_val = self.bbox["min_lat"] + frac * lat_span
            label = self.font_sm.render(f"{lat_val:.1f}°", True, PALETTE["text_dim"])
            self.screen.blit(label, (4, y - 8))

    def _draw_trajectories(self) -> None:
        """绘制预设目标航迹（使用缓存的像素坐标）。"""
        for _tid, pixel_points in self._traj_pixel_cache.items():
            if len(pixel_points) < 2:
                continue
            # 虚线效果：每隔一个点绘制线段
            for i in range(0, len(pixel_points) - 1, 2):
                if i + 1 < len(pixel_points):
                    pygame.draw.line(
                        self.screen, PALETTE["target_trail"],
                        pixel_points[i], pixel_points[i + 1], 1,
                    )

    def _draw_target_trails(self) -> None:
        """绘制目标实时尾迹（最近 N 个位置）。"""
        for _tid, trail in self._target_trail.items():
            if len(trail) < 2:
                continue
            pixel_pts = [self._geo_to_pixel(p[0], p[1]) for p in trail]
            # 尾迹逐点变淡
            n = len(pixel_pts)
            for i in range(n - 1):
                alpha = int(60 + 140 * i / n)  # 越近越亮
                r, g, b = PALETTE["target_glow"][:3]
                color = (min(255, r), min(255, g * alpha // 255), min(255, b * alpha // 255))
                pygame.draw.line(
                    self.screen, color,
                    pixel_pts[i], pixel_pts[i + 1], max(1, 3 * i // n),
                )

    def _draw_radars(self) -> Dict[str, Tuple[int, int]]:
        """绘制固定雷达（使用预渲染标签缓存）。"""
        pos_dict: Dict[str, Tuple[int, int]] = {}
        group_on = self._show_groups and bool(self._radar_to_group_color)

        for r in self.fixed_radars:
            rid = str(r.get('id', ''))
            px, py = self._geo_to_pixel(r['lon'], r['lat'])
            pos_dict[rid] = (px, py)

            color = (
                self._radar_to_group_color.get(rid, PALETTE["radar"])
                if group_on else PALETTE["radar"]
            )

            # 雷达本体
            if self._is_on_screen(px, py):
                pygame.draw.rect(self.screen, color, (px - 5, py - 5, 10, 10), 2)
                pygame.draw.rect(self.screen, (*color, 60), (px - 5, py - 5, 10, 10))  # 半透明填充
                # 使用缓存的标签
                label = self._radar_label_cache.get(rid)
                if label is None:
                    label = self.font_sm.render(rid, True, color)
                    self._radar_label_cache[rid] = label
                self.screen.blit(label, (px + 8, py - 8))

        return pos_dict

    def _draw_targets(self, targets: list) -> Dict[str, Tuple[int, int]]:
        """绘制移动目标（导弹）。"""
        pos_dict: Dict[str, Tuple[int, int]] = {}
        for t in targets:
            tid = str(t.get('id', ''))
            lon, lat = t.get('lon', 0), t.get('lat', 0)
            if abs(lon) < 0.1 and abs(lat) < 0.1:
                continue

            px, py = self._geo_to_pixel(lon, lat)
            pos_dict[tid] = (px, py)

            if not self._is_on_screen(px, py):
                continue

            # 光晕
            pygame.draw.circle(self.screen, PALETTE["target_glow"], (px, py), 8, 1)
            # 实体
            pygame.draw.circle(self.screen, PALETTE["target"], (px, py), 5)
            # 高度标签
            alt_km = t.get('alt', 0) / 1000.0
            label = self.font_sm.render(
                f"{tid} {alt_km:.0f}km", True, PALETTE["text"]
            )
            self.screen.blit(label, (px + 9, py + 6))

        return pos_dict

    def _draw_satellites(self, satellites: list) -> Dict[str, Tuple[int, int]]:
        """绘制卫星（菱形标记，不参与视口锁定）。"""
        pos_dict: Dict[str, Tuple[int, int]] = {}
        for s in satellites:
            sid = str(s.get('id', ''))
            lon, lat = s.get('lon', 0), s.get('lat', 0)
            if abs(lon) < 0.1 and abs(lat) < 0.1:
                continue
            px, py = self._geo_to_pixel(lon, lat)
            pos_dict[sid] = (px, py)
            if not self._is_on_screen(px, py):
                continue
            # 菱形
            pts = [(px, py - 5), (px + 5, py), (px, py + 5), (px - 5, py)]
            pygame.draw.polygon(self.screen, PALETTE["accent_blue"], pts, 1)
        return pos_dict

    def _draw_links(
        self, links: list, pos_dict: Dict[str, Tuple[int, int]],
        max_links: int = 100,
    ) -> None:
        """绘制雷达 → 目标的动作连线（截断上限防画面混乱）。"""
        drawn = 0
        for radar_id, target_id in links:
            if drawn >= max_links:
                break
            r_pos = pos_dict.get(radar_id)
            t_pos = pos_dict.get(target_id)
            if r_pos and t_pos:
                # 已锁定连线颜色更亮
                color = PALETTE["link_active"]
                pygame.draw.line(self.screen, color, r_pos, t_pos, 2)
                drawn += 1

    def _draw_time_overlay(self, time_step: int) -> None:
        """左上角仿真时间叠加层。"""
        # 半透明背景条
        overlay = pygame.Surface((280, 40), pygame.SRCALPHA)
        overlay.fill((8, 12, 22, 160))
        self.screen.blit(overlay, (10, 8))

        time_text = self.font_lg.render(
            f"仿真时间: {time_step} s", True, PALETTE["text_highlight"]
        )
        self.screen.blit(time_text, (18, 14))   

        # 帧率 + 缩放
        fps_text = self.font_sm.render(
            f"FPS: {self.clock.get_fps():.0f}  ·  ZOOM: {self._zoom:.1f}x",
            True, PALETTE["text_dim"],
        )
        self.screen.blit(fps_text, (18, 32))

        # 暂停提示
        if self._paused:
            pause_text = self.font_lg.render("⏸ PAUSED", True, PALETTE["warning"])
            self.screen.blit(pause_text, (self.map_width // 2 - 60, 14))

    # ============================================================
    # 信息面板
    # ============================================================

    def _draw_info_panel(
        self, time_step: int, _radars: list, targets: list,
        links: list, extra_info: Optional[dict],
        n_satellites: int = 0,
    ) -> None:
        """右侧信息面板：统计 + 图例 + 热键提示。"""
        panel_x = self.map_width + 12
        y = 16
        line_h = 22

        def _row(label: str, value: str, color: Tuple = PALETTE["text"], v_color=None):
            nonlocal y
            txt = self.font_md.render(f"{label}:", True, PALETTE["text_dim"])
            self.screen.blit(txt, (panel_x, y))
            val = self.font_md.render(value, True, v_color or color)
            self.screen.blit(val, (panel_x + 90, y))
            y += line_h

        def _sep():
            nonlocal y
            y += 4
            pygame.draw.line(
                self.screen, PALETTE["panel_border"],
                (panel_x, y), (panel_x + self.PANEL_WIDTH - 24, y), 1,
            )
            y += 8

        def _title(text: str):
            nonlocal y
            txt = self.font_lg.render(text, True, PALETTE["text_highlight"])
            self.screen.blit(txt, (panel_x, y))
            y += line_h + 6

        # ---- 标题 ----
        _title("📊 实时态势")

        # 活跃目标数
        active_targets = sum(
            1 for t in targets if abs(t.get('lon', 0)) > 0.1
        )
        _row("仿真时间", f"{time_step} s", PALETTE["accent_blue"])
        _row("活跃雷达", str(len(self.fixed_radars)), PALETTE["radar"])
        _row("卫星", str(n_satellites), PALETTE["accent_blue"])
        _row("活跃目标", str(active_targets), PALETTE["target"])
        _row("动作连线", str(len(links)))
        _row("缩放", f"{self._zoom:.1f}x")
        _row("FPS", f"{self.clock.get_fps():.0f}")

        _sep()

        # ---- H-QMIX 分组 ----
        if self.group_assignments:
            n_groups = len(self.group_assignments)
            k_max = max(len(g) for g in self.group_assignments)
            _title("🔗 H-QMIX 分组")
            _row("分组数", str(n_groups), PALETTE["accent_green"])
            _row("每组大小", f"~{k_max}")
            # 实际分布
            sizes = [sum(1 for idx in g if idx >= 0) for g in self.group_assignments]
            _row("分布", f"{min(sizes)}-{max(sizes)}")

        _sep()

        # ---- 额外信息 ----
        if extra_info:
            for key, val in extra_info.items():
                if isinstance(val, float):
                    _row(key, f"{val:.3f}")
                else:
                    _row(key, str(val))

        _sep()

        # ---- 图例 ----
        _title("🎨 图例")
        legend_items = [
            (PALETTE["radar"], "■ 雷达"),
            (PALETTE["target"], "● 目标"),
            (PALETTE["link_active"], "— 锁定连线"),
            (PALETTE["target_trail"], "- - 航迹"),
        ]
        for color, label in legend_items:
            txt = self.font_sm.render(label, True, color)
            self.screen.blit(txt, (panel_x, y))
            y += line_h - 2

        _sep()

        # ---- 热键 ----
        _title("⌨ 热键")
        hotkeys = [
            "滚轮    缩放",
            "左键拖动 平移",
            "R       复位",
            "SPACE   暂停",
            "+/-     加减速",
            "T       尾迹",
            "L       连线",
            "G       分组色",
            "ESC     退出",
        ]
        for hk in hotkeys:
            txt = self.font_sm.render(hk, True, PALETTE["text_dim"])
            self.screen.blit(txt, (panel_x, y))
            y += line_h - 4

        _sep()

        # 渲染耗时
        rt_text = self.font_sm.render(
            f"渲染: {self._render_time_ms:.1f} ms",
            True, PALETTE["text_dim"],
        )
        self.screen.blit(rt_text, (panel_x, y))

    # ============================================================
    # 清理
    # ============================================================

    def close(self) -> None:
        self._traj_pixel_cache.clear()
        self._radar_label_cache.clear()
        self._target_label_cache.clear()
        self._target_trail.clear()


# ============================================================
# LiveObserver —— 持久窗口，训练全程常驻
# ============================================================

class LiveObserver:
    """RL 训练态势观察者 —— 持久化窗口。

    训练开始时打开一个窗口，全程常驻，按 ``watch_freq`` 决定哪些局
    进行实时渲染；非渲染局窗口保持等待画面并响应键盘事件。

    用法::

        obs = LiveObserver(watch_freq=10)
        obs.start()                                    # 训练开始时一次

        for ep in range(1, max_episodes + 1):
            obs.begin_episode(ep, eps)                 # 每局开始
            ... rollout(render_callback=obs.render_callback) ...
            obs.end_episode()                           # 每局结束

        obs.shutdown()                                 # 训练结束时一次

    Args:
        watch_freq:          每隔多少局渲染一局 (0 = 永不弹窗)
        fps:                 目标帧率
        group_assignments:   H-QMIX 分组信息 → 分组着色
        radar_keys:          雷达 ID 列表 (与 group_assignments 索引对齐)
        max_episodes:        总训练局数 (用于等待画面显示进度)
    """

    def __init__(
        self,
        watch_freq: int = 10,
        fps: int = 30,
        group_assignments: Optional[List[List[int]]] = None,
        radar_keys: Optional[List[str]] = None,
        max_episodes: int = 5000,
    ):
        self.watch_freq = watch_freq
        self.fps = fps
        self.group_assignments = group_assignments
        self.radar_keys = radar_keys
        self.max_episodes = max_episodes
        self.display: Optional[TwoDimDisplay] = None
        self.active = False           # 窗口是否已创建
        self._rendering = False       # 当前这局是否在渲染
        self._episode_idx = 0
        self._epsilon = 0.0

    # ============================================================
    # 生命周期
    # ============================================================

    def start(self) -> None:
        """训练开始时调用一次：创建持久化窗口。"""
        if self.watch_freq <= 0:
            return
        if self.display is not None:
            return

        # Linux 无头环境（无 DISPLAY / WAYLAND_DISPLAY）直接跳过，避免 GLX 报错。
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print("[Live] 检测到无头 Linux 环境，已跳过态势可视化")
            return

        try:
            self.display = TwoDimDisplay(
                fps=self.fps,
                group_assignments=self.group_assignments,
                radar_keys=self.radar_keys,
            )
        except Exception as e:
            # 无 GPU / 缺 Mesa swrast / 无法创建 OpenGL 上下文（GLXBadContext）时，
            # 降级为不启用可视化，训练照常继续。
            print(f"[Live] 可视化窗口初始化失败（无显示或 OpenGL 不可用），已自动禁用: {e}")
            self.display = None
            self.active = False
            try:
                pygame.quit()
            except Exception:
                pass
            return

        self.active = True
        print(f"\n📺 [Live] 态势可视化窗口已打开 (每 {self.watch_freq} 局渲染一局)")

    def begin_episode(self, episode_idx: int, epsilon: float) -> bool:
        """每局开始前调用。

        Returns:
            bool: 本局是否需要进行逐帧渲染
        """
        if not self.active:
            return False

        self._episode_idx = episode_idx
        self._epsilon = epsilon
        # 首局必渲染（避免 watch_freq 较大时，小 max_episodes 全程看不到态势），
        # 其余按 watch_freq 间隔抽样渲染。
        self._rendering = (episode_idx == 1) or (episode_idx % self.watch_freq == 0)

        if self._rendering:
            # 重置视口以便重新锁定（不同局的实体位置可能不同）
            if self.display is not None:
                self.display.view_initialized = False
        else:
            # 非渲染局：泵事件 + 画占位，防止窗口假死/黑屏
            self._pump_events()
            if self.display is not None:
                self.display.render_idle(episode_idx, self.max_episodes)

        return self._rendering

    def render_callback(self, info: dict) -> None:
        """逐帧渲染回调 —— 由 RolloutWorker._run_episode 调用。"""
        if self.display is None:
            return

        if not self._rendering:
            # 非渲染局：只泵事件保持窗口响应（占位画面已在 begin_episode 绘制）
            self._pump_events()
            return

        # 先处理事件（防止窗口假死）
        self._pump_events()

        raw_obs = info.get('raw_obs')
        actions = info.get('agent_actions_list', [])
        if not raw_obs:
            return

        time_step = int(raw_obs.current_time)
        # 区分雷达(type==1)与卫星(type!=1)：卫星不参与视口锁定，
        # 避免 GEO 卫星(赤道/高经度)把视口撑得过大导致什么都看不清。
        radars = []
        satellites = []
        for k, v in raw_obs.dict_equip_state.items():
            entry = {'id': k, 'lon': v.longitude, 'lat': v.latitude}
            if getattr(v, 'type', 1) == 1:
                radars.append(entry)
            else:
                satellites.append(entry)
        targets = [
            {'id': k, 'lon': v.longitude, 'lat': v.latitude, 'alt': v.altitude}
            for k, v in raw_obs.dict_system_track.items()
        ]
        links = [
            (cmd.str_equip_id, cmd.str_target_id)
            for cmd in actions if cmd.str_target_id != ""
        ]

        extra = {
            "episode": f"{self._episode_idx}/{self.max_episodes}",
            "epsilon": f"{self._epsilon:.3f}",
            "n_actions": str(len(links)),
            "active_targets": str(len(targets)),
        }

        self.display.render_step(time_step, radars, targets, links, extra,
                                 satellites=satellites)

    def end_episode(self) -> None:
        """每局结束后调用：渲染局结束后短暂保持画面，非渲染局不做任何事。"""
        if not self.active or self.display is None:
            return

        if self._rendering:
            # 渲染局结束：保持最后一帧 0.5 秒，然后标记渲染结束
            pygame.time.wait(500)
            self._rendering = False

        # 处理事件队列，防止窗口卡死
        self._pump_events()

    def shutdown(self) -> None:
        """训练结束时调用一次：关闭窗口，释放资源。"""
        if self.display is not None:
            self.display.close()
        self.display = None
        self.active = False
        self._rendering = False
        pygame.quit()
        print("[Live] 态势可视化窗口已关闭")

    # ============================================================
    # 内部
    # ============================================================

    def _pump_events(self) -> None:
        """处理 pygame 事件队列，防止窗口被系统标记为'无响应'。

        复用 TwoDimDisplay.handle_event，渲染局 / 非渲染局交互一致
        （缩放、平移、暂停、退出可视化等）。
        """
        if self.display is None:
            return
        for event in pygame.event.get():
            if self.display.handle_event(event):
                self.shutdown()
                return
