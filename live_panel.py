"""Live Isaac Sim GUI for the conveyor demo: presentation camera in the viewport, and an omni.ui
"Inspection" panel docked on the right (live inspection camera with verdicts, belt state, counters,
recent decisions). Used when DEMO_LIVE=1."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    FONT = FONT_S = ImageFont.load_default()

C_BG, C_TXT, C_DIM, C_ACC, C_OK, C_BAD, C_WARN = (0xFF1E1812, 0xFFFFFFFF, 0xFFC3B7AA, 0xFF78DCFF,
                                                  0xFF8CE6A0, 0xFF5A5AF0, 0xFF3CC8FF)


class LivePanel:
    def __init__(self, presentation_camera):
        import omni.ui as ui
        import carb.settings
        import isaacsim.core.experimental.utils.app as app_utils
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.actions.actions import toggle_camera_visibility
        self.ui = ui
        vp = get_active_viewport()
        vp.camera_path = presentation_camera
        toggle_camera_visibility(vp, visible=False)
        vp.fill_frame = True
        st = carb.settings.get_settings()
        for k in ["deviceMemory", "hostMemory", "processMemory", "renderResolution", "renderProgress", "cameraSpeed"]:
            st.set(f"/persistent/app/viewport/{vp.id}/hud/{k}/visible", False)
        st.set(f"/persistent/app/viewport/{vp.id}/hud/visible", True)          # FPS stays: proof of real time
        for name in ["Stage", "Property", "Content", "Console", "Layer", "Render Settings", "Robot Inspector",
                     "Semantics Schema Editor", "Isaac Sim Assets", "Simulation Data Visualizer"]:
            w = ui.Workspace.get_window(name)
            if w:
                w.visible = False
        self.win = ui.Window("Inspection  -  RGB-D camera + RT-DETR (cap / no-cap)", width=660, height=1000)
        self.provider = ui.ByteImageProvider()
        L = {}
        with self.win.frame:
            with ui.ZStack():
                ui.Rectangle(style={"background_color": C_BG})
                with ui.VStack(spacing=5, style={"margin": 8}):
                    ui.Label("Inspection camera  |  detector + depth check must both agree",
                             height=24, style={"font_size": 18, "color": C_ACC})
                    ui.ImageWithProvider(self.provider, height=340, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    ui.Label("green: cap OK    red: NO CAP (open mouth confirmed by depth)", height=20,
                             style={"font_size": 14, "color": C_DIM})
                    L["belt"] = ui.Label("", height=34, style={"font_size": 26, "color": C_TXT})
                    L["detail"] = ui.Label("", height=22, word_wrap=True, style={"font_size": 16, "color": C_DIM})
                    ui.Spacer(height=4)
                    with ui.HStack(height=62, spacing=10):
                        for key, title, col in (("insp", "inspected", C_TXT), ("pass", "passed", C_OK),
                                                ("rej", "rejected", C_BAD)):
                            with ui.VStack():
                                L[key] = ui.Label("0", alignment=ui.Alignment.CENTER, style={"font_size": 34, "color": col})
                                ui.Label(title, alignment=ui.Alignment.CENTER, height=18, style={"font_size": 15, "color": C_DIM})
                    L["audit"] = ui.Label("", height=22, style={"font_size": 15, "color": C_DIM})
                    ui.Spacer(height=6)
                    ui.Label("Latest decisions", height=24, style={"font_size": 20, "color": C_TXT})
                    L["log"] = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP,
                                        style={"font_size": 15, "color": C_DIM})
        self.L = L
        self.lines = []
        app_utils.update_app(steps=3)
        vw = ui.Workspace.get_window("Viewport")
        if vw:
            self.win.dock_in(vw, ui.DockPosition.RIGHT, 0.34)
        app_utils.update_app(steps=3)

    def set_image(self, rgb, dets):
        """dets: list of (xyxy, verdict_text, is_bad or None)"""
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        for xyxy, txt, bad in dets:
            col = (230, 45, 45) if bad else (40, 200, 70) if bad is not None else (240, 200, 60)
            d.rectangle(xyxy, outline=col, width=5)
            tw = d.textlength(txt, font=FONT_S)
            d.rectangle([xyxy[0], xyxy[1] - 28, xyxy[0] + tw + 10, xyxy[1]], fill=col)
            d.text((xyxy[0] + 5, xyxy[1] - 26), txt, font=FONT_S, fill=(255, 255, 255))
        small = np.asarray(img.resize((640, 360)).convert("RGBA"), dtype=np.uint8)
        self.provider.set_data_array(np.ascontiguousarray(small), [640, 360])

    def update(self, belt_text, detail, n_insp, n_pass, n_rej, audit):
        self.L["belt"].text = belt_text
        self.L["detail"].text = detail
        self.L["insp"].text = str(n_insp)
        self.L["pass"].text = str(n_pass)
        self.L["rej"].text = str(n_rej)
        self.L["audit"].text = audit

    def add_decision(self, text):
        self.lines.append(text)
        self.L["log"].text = "\n".join(self.lines[-7:][::-1])
