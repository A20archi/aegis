"""
manim_aegis.py -- AEGIS architecture animation (white background).

Explains the dual-locus design on a FROZEN SmolVLA:
  RGB -> Vision Encoder -> [RIB @ connector: PERCEPTION locus]
       -> Action Expert -> [RASF @ action chunk: ACTION locus] -> TE -> env
with animated explainer insets for each locus (RIB bottleneck sheds a corrupted
token; RASF spectral bars pull down an anomalous band), and the identity-residual
"pass-through at init" guarantee.

Render (forgedeck env has manim 0.20.1):
  /home/user/anaconda3/envs/forgedeck/bin/manim -qh --format=mp4 \
      --media_dir presentation/_manim -o aegis_architecture \
      presentation/manim_aegis.py AegisArchitecture
"""
from manim import *

INK   = "#151820"
BLUE  = "#1F5FD0"
TEAL  = "#1B9E4B"
AMBER = "#E68A00"
RED   = "#C0392B"
GREY  = "#6B7280"
LGREY = "#E0E3E9"


def box(label, color, w=2.6, h=1.2, fs=22, sub=None, fill=WHITE, fopac=1.0):
    rect = RoundedRectangle(corner_radius=0.12, width=w, height=h,
                            stroke_color=color, stroke_width=3.5,
                            fill_color=fill, fill_opacity=fopac)
    txt = Text(label, color=INK, font_size=fs, weight=BOLD).move_to(rect)
    if txt.width > w - 0.3:
        txt.scale((w - 0.3) / txt.width)
    grp = VGroup(rect, txt)
    if sub:
        s = Text(sub, color=GREY, font_size=13).next_to(txt, DOWN, buff=0.08)
        if s.width > w - 0.2:
            s.scale((w - 0.2) / s.width)
        grp.add(s)
    return grp


def badge(tag, name, color, w=3.0):
    rect = RoundedRectangle(corner_radius=0.1, width=w, height=0.9,
                            stroke_color=color, stroke_width=3,
                            fill_color=color, fill_opacity=0.10)
    t1 = Text(tag, color=color, font_size=18, weight=BOLD)
    t2 = Text(name, color=INK, font_size=13)
    if t2.width > w - 0.2:
        t2.scale((w - 0.2) / t2.width)
    VGroup(t1, t2).arrange(DOWN, buff=0.05).move_to(rect)
    return VGroup(rect, t1, t2)


class AegisArchitecture(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # ---------- title ----------
        title = Text("AEGIS — Dual-Locus Robustness for a Frozen VLA",
                     color=INK, font_size=34, weight=BOLD).to_edge(UP, buff=0.35)
        underline = Line(title.get_left(), title.get_right(),
                         color=BLUE, stroke_width=5).next_to(title, DOWN, buff=0.1)
        sub = Text("One rate-limited information bottleneck, realised at two interfaces inside a frozen SmolVLA",
                   color=GREY, font_size=18).next_to(underline, DOWN, buff=0.14)
        self.play(Write(title), GrowFromCenter(underline))
        self.play(FadeIn(sub, shift=UP*0.2))
        self.wait(0.3)

        # ---------- frozen backbone pipeline ----------
        rgb  = box("RGB obs", INK, w=1.7, sub="camera")
        venc = box("Vision Enc.", GREY, w=2.0, sub="frozen")
        aexp = box("Action Expert", GREY, w=2.3, sub="flow-match · frozen")
        env  = box("env step", INK, w=1.7, sub="execute")
        backbone = VGroup(rgb, venc, aexp, env).arrange(RIGHT, buff=2.6)
        backbone.scale_to_fit_width(13.2).move_to(ORIGIN).shift(UP*0.35)

        arr = VGroup()
        for a, b in zip(backbone[:-1], backbone[1:]):
            arr.add(Arrow(a.get_right(), b.get_left(), buff=0.1, color=GREY,
                          stroke_width=4, max_tip_length_to_length_ratio=0.16))
        self.play(FadeIn(rgb, shift=RIGHT*0.3))
        for i in range(1, 4):
            self.play(GrowArrow(arr[i-1]), FadeIn(backbone[i], shift=RIGHT*0.3), run_time=0.4)

        # frozen banner
        frozen = Text("backbone FROZEN — no gradient to pretrained weights",
                      color=BLUE, font_size=16, weight=BOLD)
        frz_box = SurroundingRectangle(VGroup(venc, aexp), color=BLUE, buff=0.25,
                                       corner_radius=0.1, stroke_width=2)
        frz_box.set_stroke(opacity=0.5)
        frozen.next_to(frz_box, UP, buff=0.12).shift(RIGHT*0.0)
        self.play(Create(frz_box), FadeIn(frozen))
        self.wait(0.4)
        self.play(FadeOut(frozen))

        # ---------- insert the two loci ----------
        rib  = box("RIB", BLUE, w=1.25, h=1.2, fs=21, fill=BLUE, fopac=0.10)
        rib.move_to((venc.get_center()+aexp.get_center())/2)
        rasf = box("RASF", BLUE, w=1.3, h=1.2, fs=21, fill=BLUE, fopac=0.10)
        rasf.move_to((aexp.get_center()+env.get_center())/2)

        # re-route arrows through the loci
        self.play(FadeOut(arr[1]), FadeOut(arr[2]))
        a_vr = Arrow(venc.get_right(), rib.get_left(), buff=0.1, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        a_ra = Arrow(rib.get_right(), aexp.get_left(), buff=0.1, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        a_af = Arrow(aexp.get_right(), rasf.get_left(), buff=0.1, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        a_fe = Arrow(rasf.get_right(), env.get_left(), buff=0.1, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        self.play(GrowFromCenter(rib), GrowArrow(a_vr), GrowArrow(a_ra))
        b_rib = badge("PERCEPTION locus", "RIB · VIB @ connector · ~2.27M", BLUE, w=3.6).next_to(rib, UP, buff=0.5)
        cn1 = DashedLine(rib.get_top(), b_rib.get_bottom(), color=BLUE, stroke_width=2)
        self.play(GrowFromCenter(b_rib), Create(cn1), run_time=0.5)
        self.wait(0.2)
        self.play(GrowFromCenter(rasf), GrowArrow(a_af), GrowArrow(a_fe))
        b_rasf = badge("ACTION locus", "RASF · DCT spectral residual", BLUE, w=3.6).next_to(rasf, DOWN, buff=0.5)
        cn2 = DashedLine(rasf.get_bottom(), b_rasf.get_top(), color=BLUE, stroke_width=2)
        self.play(GrowFromCenter(b_rasf), Create(cn2), run_time=0.5)
        self.wait(0.3)

        # ---------- token flows through ----------
        self.play(FadeOut(b_rib), FadeOut(cn1), FadeOut(b_rasf), FadeOut(cn2))
        token = Dot(color=AMBER, radius=0.13).move_to(rgb.get_center())
        glow = Circle(radius=0.24, color=AMBER, stroke_width=3).move_to(token)
        self.play(FadeIn(token), FadeIn(glow))
        for pt in [venc, rib, aexp, rasf, env]:
            self.play(token.animate.move_to(pt.get_center()),
                      glow.animate.move_to(pt.get_center()), run_time=0.5)
        self.play(FadeOut(token), FadeOut(glow))
        self.wait(0.2)

        # ================= EXPLAINER INSETS =================
        self.play(*[FadeOut(m) for m in [title, underline, sub, frz_box, backbone,
                    rib, rasf, a_vr, a_ra, a_af, a_fe, arr[0]]])
        sec = Text("How each locus removes its corruption", color=INK,
                   font_size=26, weight=BOLD).to_edge(UP, buff=0.5)
        self.play(FadeIn(sec))

        # ----- RIB inset: bottleneck sheds a corrupted token -----
        rib_t = Text("RIB — perception bottleneck", color=BLUE, font_size=22, weight=BOLD)
        rib_t.move_to(LEFT*3.4 + UP*1.0)
        # funnel: wide -> neck -> wide
        neck = Line(LEFT*4.6+DOWN*0.3, LEFT*2.2+DOWN*0.3, color=GREY, stroke_width=0)
        funnel = Polygon(LEFT*4.7+UP*0.4, LEFT*3.6+DOWN*0.1, LEFT*3.6+DOWN*0.9, LEFT*4.7+DOWN*1.4,
                         stroke_color=BLUE, stroke_width=3, fill_color=BLUE, fill_opacity=0.06)
        funnel2 = Polygon(LEFT*2.1+UP*0.4, LEFT*3.2+DOWN*0.1, LEFT*3.2+DOWN*0.9, LEFT*2.1+DOWN*1.4,
                          stroke_color=BLUE, stroke_width=3, fill_color=BLUE, fill_opacity=0.06)
        good = VGroup(*[Dot(color=TEAL, radius=0.08) for _ in range(4)])
        good.arrange(DOWN, buff=0.18).move_to(LEFT*4.9+DOWN*0.5)
        bad = Dot(color=RED, radius=0.10).move_to(LEFT*4.9+DOWN*0.05)
        self.play(FadeIn(rib_t), Create(funnel), Create(funnel2))
        self.play(FadeIn(good), FadeIn(bad))
        # good tokens pass; bad token is shed at the neck
        self.play(good.animate.move_to(LEFT*2.0+DOWN*0.5), run_time=0.9)
        self.play(bad.animate.move_to(LEFT*3.4+DOWN*0.4).scale(0.1).set_opacity(0.0), run_time=0.9)
        shed = Text("corruption-sensitive subspace shed;\nbenign info passes (pass-through at init)",
                    color=GREY, font_size=15, line_spacing=0.8).move_to(LEFT*3.4+DOWN*2.05)
        self.play(FadeIn(shed))
        self.wait(0.5)

        # ----- RASF inset: spectral bars, anomalous band pulled down -----
        rasf_t = Text("RASF — action spectral filter", color=BLUE, font_size=22, weight=BOLD)
        rasf_t.move_to(RIGHT*3.4 + UP*1.0)
        base_y = -1.0
        heights = [0.5, 0.8, 0.6, 1.7, 0.5, 0.7, 0.45]  # band 3 anomalous
        bars = VGroup()
        for i, hh in enumerate(heights):
            b = Rectangle(width=0.34, height=hh, stroke_color=BLUE, stroke_width=2,
                          fill_color=(RED if i == 3 else BLUE),
                          fill_opacity=(0.7 if i == 3 else 0.25))
            b.move_to(RIGHT*(2.0+i*0.42) + UP*(base_y + hh/2))
            bars.add(b)
        axis = Line(RIGHT*1.75+UP*base_y, RIGHT*5.0+UP*base_y, color=GREY, stroke_width=2)
        floor = DashedLine(RIGHT*1.75+UP*(base_y+0.45), RIGHT*5.0+UP*(base_y+0.45), color=GREY, stroke_width=1.5)
        self.play(FadeIn(rasf_t), Create(axis), Create(floor), *[GrowFromEdge(b, DOWN) for b in bars])
        # pull the anomalous band down to the floor
        target = Rectangle(width=0.34, height=0.5, stroke_color=BLUE, stroke_width=2,
                           fill_color=BLUE, fill_opacity=0.25).move_to(RIGHT*(2.0+3*0.42)+UP*(base_y+0.25))
        self.play(Transform(bars[3], target), run_time=1.0)
        rcap = Text("anomalous band pulled to the gain floor;\nbenign bands all-pass (bounded residual)",
                    color=GREY, font_size=15, line_spacing=0.8).move_to(RIGHT*3.4+DOWN*2.05)
        self.play(FadeIn(rcap))
        self.wait(0.6)

        # ---------- closing guarantee ----------
        self.play(*[FadeOut(m) for m in [sec, rib_t, funnel, funnel2, good, shed,
                                          rasf_t, axis, floor, bars, rcap]])
        g1 = Text("Both loci are exact pass-throughs at initialisation",
                  color=INK, font_size=30, weight=BOLD)
        g2 = Text("clean success protected by construction  ·  robustness strictly additive",
                  color=BLUE, font_size=22, weight=BOLD)
        g3 = Text("+ Temporal Ensembling (consensus over overlapping chunks) in both arms",
                  color=GREY, font_size=18)
        VGroup(g1, g2, g3).arrange(DOWN, buff=0.35).move_to(ORIGIN)
        self.play(Write(g1))
        self.play(FadeIn(g2, shift=UP*0.2))
        self.play(FadeIn(g3))
        self.wait(1.2)
