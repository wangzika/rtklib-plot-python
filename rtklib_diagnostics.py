"""Modular RTKLIB diagnostic plots (01-15) and overview dashboard.

Metrics are computed from the documented RTKLIB .pos columns, matching GPST
to reference ECEF, and $SAT records in .stat. Unsupported diagnostics are
explicitly marked instead of being fabricated.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POS = ROOT / "results_pos" / "6_RTK_Multi_L1L2.pos"
DEFAULT_REF = ROOT.parent / "Tokyo_Data" / "Tokyo_Data" / "Odaiba" / "reference.csv"
DEFAULT_OUT = ROOT / "plots_15_diagnostics"
EXPORT_FORMATS = ("png",)
EXPORT_DPI = 180

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 130,
                     "savefig.dpi": 180, "font.size": 9})
COLORS = {"E": "#2878B5", "N": "#E07A2D", "U": "#36965A",
          "GPS": "#2878B5", "GLONASS": "#E07A2D", "Galileo": "#36965A",
          "BeiDou": "#A34FA3", "QZSS": "#D3A11E", "SBAS": "#777777"}


def _geodetic_to_ecef(lat_deg: float | np.ndarray, lon_deg: float | np.ndarray, h: float | np.ndarray):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    return x, y, z


def _ecef_to_enu(xyz: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    p, l = np.deg2rad([lat_deg, lon_deg])
    rot = np.array([[-np.sin(l), np.cos(l), 0],
                    [-np.sin(p)*np.cos(l), -np.sin(p)*np.sin(l), np.cos(p)],
                    [np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)]])
    return (xyz @ rot.T)


def load_data(pos_path: Path, reference_path: Path | None = None,
              stat_path: Path | None = None) -> dict:
    """Load RTKLIB .pos and optional reference/.stat inputs into aligned frames."""
    from datetime import datetime
    pos_path = Path(pos_path)
    stat_path = Path(stat_path) if stat_path else Path(str(pos_path) + ".stat")
    rows = []
    for line in pos_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        f = line.split()
        if len(f) < 15:
            continue
        try:
            # 1. 兼容日期时间与 week/tow
            if "/" in f[0] or "-" in f[0]:
                t_str = f[0] + " " + f[1]
                try:
                    dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(t_str, "%Y/%m/%d %H:%M:%S")
                diff = (dt - datetime(1980, 1, 6)).total_seconds()
                week = int(diff // 604800)
                tow = float(diff % 604800)
                v1, v2, v3 = float(f[2]), float(f[3]), float(f[4])
                q, ns = int(f[5]), int(f[6])
                s1, s2, s3 = float(f[7]), float(f[8]), float(f[9])
                s12, s23, s31 = float(f[10]), float(f[11]), float(f[12])
                age, ratio = float(f[13]), float(f[14])
            else:
                week = int(f[0])
                tow = float(f[1])
                v1, v2, v3 = float(f[2]), float(f[3]), float(f[4])
                q, ns = int(f[5]), int(f[6])
                s1, s2, s3 = float(f[7]), float(f[8]), float(f[9])
                s12, s23, s31 = float(f[10]), float(f[11]), float(f[12])
                age, ratio = float(f[13]), float(f[14])

            # 2. 兼容 LLH (纬度/经度/高度) 还是 ECEF (x/y/z)
            is_llh = abs(v1) <= 90.0 and abs(v2) <= 360.0
            if is_llh:
                lat, lon, h = v1, v2, v3
                x, y, z = _geodetic_to_ecef(lat, lon, h)
                sdn, sde, sdu = s1, s2, s3
                sdne, sdeu, sdun = s12, s23, s31
                sdx, sdy, sdz = s1, s2, s3
                sdxy, sdyz, sdzx = s12, s23, s31
            else:
                x, y, z = v1, v2, v3
                lat, lon, h = np.nan, np.nan, np.nan
                sdx, sdy, sdz = s1, s2, s3
                sdxy, sdyz, sdzx = s12, s23, s31
                sdn, sde, sdu = np.nan, np.nan, np.nan
                sdne, sdeu, sdun = np.nan, np.nan, np.nan

            rows.append([week, tow, x, y, z, lat, lon, h, q, ns,
                         sdx, sdy, sdz, sdxy, sdyz, sdzx,
                         sdn, sde, sdu, sdne, sdeu, sdun,
                         age, ratio, is_llh])
        except (ValueError, IndexError):
            continue

    cols = ["week", "tow", "x", "y", "z", "lat", "lon", "h", "q", "ns",
            "sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx",
            "sdn", "sde", "sdu", "sdne", "sdeu", "sdun",
            "age", "ratio", "is_llh"]
    pos = pd.DataFrame(rows, columns=cols).sort_values("tow").reset_index(drop=True)
    if pos.empty:
        raise ValueError(f"No position epochs parsed from {pos_path}")
    pos["time"] = pos.tow - pos.tow.iloc[0]

    ref = None
    if reference_path and Path(reference_path).exists():
        ref = pd.read_csv(reference_path, skipinitialspace=True)
        needed = ["GPS TOW (s)", "Latitude (deg)", "Longitude (deg)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]
        if all(c in ref for c in needed):
            ref = ref[needed].dropna().sort_values("GPS TOW (s)")
            xyz0 = ref[["ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]].to_numpy()
            ref_lat0 = float(ref.iloc[0]["Latitude (deg)"])
            ref_lon0 = float(ref.iloc[0]["Longitude (deg)"])
            enu_ref = _ecef_to_enu(xyz0 - xyz0[0], ref_lat0, ref_lon0)
            xyz = pos[["x", "y", "z"]].to_numpy()
            enu_pos = _ecef_to_enu(xyz - xyz0[0], ref_lat0, ref_lon0)
            pos[["E", "N", "U"]] = enu_pos
            for i, c in enumerate(["E", "N", "U"]):
                pos[f"gt_{c}"] = np.interp(pos.tow, ref["GPS TOW (s)"], enu_ref[:, i], left=np.nan, right=np.nan)
                pos[f"d{c.lower()}"] = pos[c] - pos[f"gt_{c}"]
            pos["eh"] = np.hypot(pos.de, pos.dn)
            pos["e3d"] = np.sqrt(pos.de**2 + pos.dn**2 + pos.du**2)
        else:
            ref = None

    if "E" not in pos:
        ref_lat0 = float(pos.iloc[0]["lat"]) if np.isfinite(pos.iloc[0]["lat"]) else 0.0
        ref_lon0 = float(pos.iloc[0]["lon"]) if np.isfinite(pos.iloc[0]["lon"]) else 0.0
        xyz0 = pos[["x", "y", "z"]].iloc[0].to_numpy()
        enu_pos = _ecef_to_enu(pos[["x", "y", "z"]].to_numpy() - xyz0, ref_lat0, ref_lon0)
        pos[["E", "N", "U"]] = enu_pos

    if pos["is_llh"].iloc[0]:
        pos["sigE"] = pos["sde"]
        pos["sigN"] = pos["sdn"]
        pos["sigU"] = pos["sdu"]
        pos["covEN"] = np.sign(pos["sdne"]) * pos["sdne"]**2
        pos["rhoEN"] = pos["covEN"] / np.maximum(pos["sigE"] * pos["sigN"], 1e-15)
    else:
        cov_ecef = np.zeros((len(pos), 3, 3))
        cov_ecef[:, 0, 0], cov_ecef[:, 1, 1], cov_ecef[:, 2, 2] = pos.sdx**2, pos.sdy**2, pos.sdz**2
        cov_ecef[:, 0, 1] = cov_ecef[:, 1, 0] = np.sign(pos.sdxy)*pos.sdxy**2
        cov_ecef[:, 1, 2] = cov_ecef[:, 2, 1] = np.sign(pos.sdyz)*pos.sdyz**2
        cov_ecef[:, 2, 0] = cov_ecef[:, 0, 2] = np.sign(pos.sdzx)*pos.sdzx**2
        lat = float(ref.iloc[0]["Latitude (deg)"]) if ref is not None else (float(pos.iloc[0]["lat"]) if np.isfinite(pos.iloc[0]["lat"]) else 0.0)
        lon = float(ref.iloc[0]["Longitude (deg)"]) if ref is not None else (float(pos.iloc[0]["lon"]) if np.isfinite(pos.iloc[0]["lon"]) else 0.0)
        p_rad, l_rad = np.deg2rad([lat, lon])
        R = np.array([[-np.sin(l_rad), np.cos(l_rad), 0],
                      [-np.sin(p_rad)*np.cos(l_rad), -np.sin(p_rad)*np.sin(l_rad), np.cos(p_rad)],
                      [np.cos(p_rad)*np.cos(l_rad), np.cos(p_rad)*np.sin(l_rad), np.sin(p_rad)]])
        ce = np.einsum("ij,njk,lk->nil", R, cov_ecef, R)
        pos["sigE"], pos["sigN"], pos["sigU"] = np.sqrt(np.maximum(ce[:, 0, 0], 0)), np.sqrt(np.maximum(ce[:, 1, 1], 0)), np.sqrt(np.maximum(ce[:, 2, 2], 0))
        pos["covEN"] = ce[:, 0, 1]
        pos["rhoEN"] = ce[:, 0, 1] / np.maximum(pos.sigE * pos.sigN, 1e-15)
    sat = []
    if stat_path.exists():
        for line in stat_path.open(encoding="utf-8", errors="replace"):
            if not line.startswith("$SAT,"): continue
            f = line.rstrip().split(",")
            try:
                sat.append([int(f[1]), float(f[2]), f[3], int(f[4]), float(f[5]), float(f[6]), float(f[7]), float(f[8]), int(f[9]), float(f[10]), int(f[11]), int(f[12])])
            except (ValueError, IndexError): pass
    sats = pd.DataFrame(sat, columns=["week", "tow", "prn", "freq", "az", "el", "resp", "resc", "vsat", "snr", "fix", "slipc"])
    if not sats.empty:
        sats["time"] = sats.tow - pos.tow.iloc[0]
        sats["constellation"] = sats.prn.str[0].map({"G":"GPS", "R":"GLONASS", "E":"Galileo", "C":"BeiDou", "J":"QZSS", "S":"SBAS"}).fillna("Other")
    return {"pos": pos, "sat": sats, "reference": ref, "has_truth": "de" in pos,
            "pos_path": pos_path, "stat_path": stat_path}


def _save(fig, out, name):
    fig.tight_layout()
    for fmt in EXPORT_FORMATS:
        fig.savefig(out / f"{name}.{fmt}", format=fmt,
                    bbox_inches="tight", dpi=EXPORT_DPI if fmt == "png" else None)
    plt.close(fig)

def _status_shade(ax, p):
    if "q" not in p: return
    t = p.time.to_numpy(); q = p.q.to_numpy()
    for i in range(len(t)-1):
        ax.axvspan(t[i], t[i+1], color={1:"#61a96d",2:"#e9b44c"}.get(q[i],"#dddddd"), alpha=.10, lw=0)

def _title(fig, text): fig.suptitle(text, fontweight="bold", fontsize=15)

def plot_01(d, out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(12,4.2)); _title(fig,"01 · ENU 轨迹与高程")
    ax[0].plot(p.E,p.N,color="#444",lw=1,label="RTK")
    if d["has_truth"]: ax[0].plot(p.gt_E,p.gt_N,"--",color="#cf4b43",lw=1,label="真值")
    ax[0].set(xlabel="东向 E (m)",ylabel="北向 N (m)",title="水平轨迹"); ax[0].axis("equal"); ax[0].legend(); ax[0].grid(alpha=.25)
    _status_shade(ax[1],p); ax[1].plot(p.time,p.U,label="RTK U")
    if d["has_truth"]: ax[1].plot(p.time,p.gt_U,label="真值 U"); ax[1].legend()
    ax[1].set(xlabel="相对时间 (s)",ylabel="ENU 高程 (m)",title="高程与解算状态"); ax[1].grid(alpha=.25)
    _save(fig,out,"plot_01_enu_position")

def plot_02(d,out):
    p=d["pos"]
    if not d["has_truth"]: return _unavailable(out,"plot_02_enu_error","需要真值才能计算 ΔE/ΔN/ΔU")
    fig,axs=plt.subplots(3,1,figsize=(11,7),sharex=True); _title(fig,"02 · ENU 位置误差与 95% 置信区间")
    for ax,c,sig,col in zip(axs,["de","dn","du"],["sigE","sigN","sigU"],[COLORS["E"],COLORS["N"],COLORS["U"]]):
        x=p[c].dropna(); mu=x.mean(); sd=x.std(); ax.plot(p.time,p[c],lw=.65,color=col); ax.axhline(mu,color="black",ls="--",lw=.8,label=f"均值 {mu:.3f} m")
        ax.axhspan(mu-1.96*sd,mu+1.96*sd,color=col,alpha=.12,label=f"经验区间 μ±1.96σ [{mu-1.96*sd:.3f}, {mu+1.96*sd:.3f}] m")
        ax.plot(p.time,1.96*p[sig],color="#777",ls=":",lw=.7); ax.plot(p.time,-1.96*p[sig],color="#777",ls=":",lw=.7,label="滤波 ±1.96σ")
        ax.set_ylabel(c.upper()+" (m)"); ax.grid(alpha=.2); ax.legend(loc="upper right",fontsize=7,ncol=2)
    axs[-1].set_xlabel("相对时间 (s)"); _save(fig,out,"plot_02_enu_error")

def plot_03(d,out):
    p=d["pos"]
    if not d["has_truth"]: return _unavailable(out,"plot_03_horizontal_3d_error","需要真值才能计算水平/三维误差")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"03 · 水平、三维误差与经验 CDF")
    ax[0].plot(p.time,p.eh,label="水平 √(ΔE²+ΔN²)"); ax[0].plot(p.time,p.e3d,label="三维 √(ΔE²+ΔN²+ΔU²)"); ax[0].set(xlabel="时间 (s)",ylabel="误差 (m)"); ax[0].legend(); ax[0].grid(alpha=.2)
    for c,label in [("eh","水平"),("e3d","三维")]:
        x=np.sort(p[c].dropna()); ax[1].plot(x,100*np.arange(1,len(x)+1)/len(x),label=label)
    ax[1].set(xlabel="误差 (m)",ylabel="经验累积概率 (%)",title=f"水平 RMSE={np.sqrt(np.mean(p.eh**2)):.3f} m · 3D RMSE={np.sqrt(np.mean(p.e3d**2)):.3f} m"); ax[1].legend(); ax[1].grid(alpha=.2)
    _save(fig,out,"plot_03_horizontal_3d_error")

def plot_04(d,out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"04 · 解算状态与历元占比")
    ax[0].step(p.time,p.q,where="post",lw=.8); ax[0].set(xlabel="时间 (s)",ylabel="Q 状态",yticks=sorted(p.q.unique()),title="RTKLIB 状态时间序列"); ax[0].grid(alpha=.2)
    vc=p.q.value_counts().sort_index(); names={1:"FIX",2:"FLOAT",3:"SBAS",4:"DGPS",5:"单点",6:"PPP"}; ax[1].bar([names.get(i,str(i)) for i in vc.index],vc.values/len(p)*100,color=["#4b9b64" if i==1 else "#e5ae46" for i in vc.index]); ax[1].set(ylabel="历元比例 (%)",title="状态构成")
    for x,y in enumerate(vc.values/len(p)*100): ax[1].text(x,y,f"{y:.1f}%",ha="center",va="bottom")
    _save(fig,out,"plot_04_solution_status")

def plot_05(d,out):
    p=d["pos"]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"05 · 模糊度固定 Ratio")
    ax[0].plot(p.time,p.ratio,lw=.65); ax[0].axhline(3,color="red",ls="--",label="参考门限 3"); ax[0].set(xlabel="时间 (s)",ylabel="Ratio",ylim=(0,max(5,min(50,p.ratio.quantile(.995)))*1.1)); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].hist(p.ratio.clip(upper=p.ratio.quantile(.99)),bins=40,color="#607da8"); ax[1].axvline(3,color="red",ls="--"); ax[1].set(xlabel="Ratio (截于 99% 分位)",ylabel="历元数",title=f"中位数 {p.ratio.median():.2f} · >3 占比 {(p.ratio>3).mean()*100:.1f}%"); _save(fig,out,"plot_05_ar_ratio")

def _sat_epoch(d):
    s=d["sat"]
    if s.empty: return pd.DataFrame()
    return (s.groupby(["tow","time"])
            .apply(lambda g: pd.Series({"tracked": g.prn.nunique(),
                                        "used": g.loc[g.vsat == 1, "prn"].nunique()}),
                   include_groups=False).reset_index())

def plot_06(d,out):
    s=d["sat"]
    if s.empty: return _unavailable(out,"plot_06_satellite_counts",".stat 中无 $SAT 记录")
    e=_sat_epoch(d); fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True); _title(fig,"06 · 跟踪、使用卫星数与星座组成")
    ax[0].plot(e.time,e.tracked,label="跟踪 PRN 数"); ax[0].plot(e.time,e.used,label="使用频率记录数"); ax[0].set(ylabel="数量"); ax[0].legend(); ax[0].grid(alpha=.2)
    ss=s[(s.freq==1)&(s.vsat==1)].groupby(["time","constellation"]).prn.nunique().unstack(fill_value=0)
    ss.plot.area(ax=ax[1],stacked=True,alpha=.85,color=[COLORS.get(c) for c in ss.columns]); ax[1].set(xlabel="时间 (s)",ylabel="使用卫星数",title="按星座堆叠（L1 频率、vsat=1、按 PRN 去重）"); ax[1].legend(ncol=6,fontsize=7); _save(fig,out,"plot_06_satellite_counts")

def _compute_dop(s):
    rows=[]
    for (tow,t),g in s[(s.freq==1)&(s.vsat==1)].groupby(["tow","time"]):
        g=g.drop_duplicates("prn")
        if len(g)<4: continue
        az,el=np.deg2rad(g.az),np.deg2rad(g.el)
        A=np.c_[np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el),np.ones(len(g))]
        try:
            Q=np.linalg.inv(A.T@A); rows.append([tow,t,*np.sqrt(np.maximum([Q[0,0]+Q[1,1]+Q[2,2],Q[0,0]+Q[1,1],Q[2,2]],0))])
        except np.linalg.LinAlgError: pass
    return pd.DataFrame(rows,columns=["tow","time","PDOP","HDOP","VDOP"])

def plot_07(d,out):
    dop=_compute_dop(d["sat"])
    if dop.empty:return _unavailable(out,"plot_07_dop","有效 $SAT 几何不足以计算 DOP")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"07 · 由卫星视线几何计算的 DOP")
    for c in ["PDOP","HDOP","VDOP"]: ax[0].plot(dop.time,dop[c],label=c)
    ax[0].axhspan(0,2,color="#55a868",alpha=.10); ax[0].axhspan(2,5,color="#ddaa33",alpha=.10); ax[0].axhspan(5,50,color="#c44e52",alpha=.08); ax[0].set(xlabel="时间 (s)",ylabel="DOP",ylim=(0,min(20,dop.PDOP.quantile(.99)*1.2))); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].hist(dop.PDOP,bins=35,color="#6688aa"); ax[1].set(xlabel="PDOP",ylabel="历元数",title=f"median={dop.PDOP.median():.2f} · 95%={dop.PDOP.quantile(.95):.2f}"); _save(fig,out,"plot_07_dop")

def plot_08(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_08_skyplot",".stat 中无卫星方位/高度角")
    fig=plt.figure(figsize=(7,7)); ax=fig.add_subplot(111,projection="polar"); _title(fig,"08 · 卫星天空图（方位角与高度角轨迹）")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(90,0); ax.set_yticks([0,30,60,90]); ax.set_yticklabels(["90°","60°","30°","0°"])
    s=s[s.freq==1]
    for (prn,con),g in s.groupby(["prn","constellation"]):
        g=g.sort_values("time"); th=np.deg2rad(g.az); r=90-g.el; color=COLORS.get(con,"#555")
        ax.plot(th,r,color=color,alpha=.45,lw=.6); ax.scatter(th.iloc[::max(1,len(g)//20)],r.iloc[::max(1,len(g)//20)],s=5,color=color)
        k=int(np.nanargmax(g.el.to_numpy())); ax.text(th.iloc[k],r.iloc[k],prn,fontsize=6,color=color)
    _save(fig,out,"plot_08_skyplot")

def plot_09(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_09_cn0",".stat 中无 C/N0")
    s=s[s.snr>0]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"09 · C/N₀ 信号质量")
    for f,g in s.groupby("freq"): ax[0].scatter(g.time,g.snr,s=2,alpha=.12,label=f"频率 {f}")
    ax[0].set(xlabel="时间 (s)",ylabel="C/N₀ (dB-Hz)"); ax[0].legend(); ax[0].grid(alpha=.2)
    groups=[(c,f,s[(s.constellation==c)&(s.freq==f)].snr.to_numpy()) for c in s.constellation.unique() for f in sorted(s.freq.unique())]; groups=[x for x in groups if len(x[2])]
    ax[1].boxplot([g[2] for g in groups],showfliers=False); ax[1].set_xticks(range(1,len(groups)+1),[f"{c}\nL{f}" for c,f,_ in groups],rotation=25); ax[1].set(ylabel="C/N₀ (dB-Hz)",title="按星座与频率分布"); ax[1].grid(axis="y",alpha=.2); _save(fig,out,"plot_09_cn0")

def plot_10(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_10_elevation",".stat 中无卫星高度角")
    s=s[s.freq==1]; fig,ax=plt.subplots(1,3,figsize=(14,4)); _title(fig,"10 · 卫星高度角与观测质量")
    for prn,g in s.groupby("prn"): ax[0].plot(g.time,g.el,lw=.45,alpha=.35)
    ax[0].set(xlabel="时间 (s)",ylabel="高度角 (°)",title="L1 卫星仰角"); ax[0].grid(alpha=.2)
    ax[1].scatter(s.el,s.snr,s=3,alpha=.18); ax[1].set(xlabel="高度角 (°)",ylabel="C/N₀ (dB-Hz)",title="高度角与信号强度"); ax[1].grid(alpha=.2)
    ax[2].scatter(s.el,s.resp.abs(),s=3,alpha=.18); ax[2].set(xlabel="高度角 (°)",ylabel="|伪距残差| (m)",title="高度角与伪距残差"); ax[2].grid(alpha=.2); _save(fig,out,"plot_10_elevation")

def plot_11(d,out):
    # $SAT residual is a post-fit residual, not a direct multipath observable.
    s=d["sat"]; s=s[(s.freq==1)&(s.resp!=0)]
    if s.empty:return _unavailable(out,"plot_11_multipath","无有效伪距残差；且 .stat 不含独立多路径组合")
    s=s.copy(); s["abs_resid"] = s.resp.abs(); s["elbin"]=pd.cut(s.el,np.arange(0,100,10),include_lowest=True)
    rms=s.groupby("elbin",observed=False).resp.apply(lambda x:np.sqrt(np.mean(x*x)))
    fig,ax=plt.subplots(1,2,figsize=(10,4)); _title(fig,"11 · 伪距残差诊断（多路径代理量，非 MP 组合）")
    ax[0].scatter(s.time,s.resp,s=2,alpha=.12); ax[0].set(xlabel="时间 (s)",ylabel="伪距残差 (m)"); ax[0].grid(alpha=.2)
    valid=[(x.mid,v) for x,v in rms.items() if not pd.isna(x) and np.isfinite(v)]
    if valid: ax[1].plot([x for x,_ in valid],[v for _,v in valid],"o-")
    ax[1].set(xlabel="高度角区间中点 (°)",ylabel="伪距残差 RMS (m)"); ax[1].grid(alpha=.2); _save(fig,out,"plot_11_multipath")

def plot_12(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_12_residuals",".stat 中无观测残差")
    fig,ax=plt.subplots(2,2,figsize=(11,7)); _title(fig,"12 · RTKLIB $SAT 观测残差")
    for a,col,label,scale in [(ax[0,0],"resp","伪距残差 (m)",1),(ax[1,0],"resp","伪距残差 (m)",1),(ax[0,1],"resc","载波残差 (m)",100),(ax[1,1],"resc","载波残差 (cm)",100)]:
        x=s.loc[s[col]!=0,col].to_numpy()*scale
        if a in (ax[0,0],ax[0,1]): a.scatter(s.loc[s[col]!=0,"time"],x,s=2,alpha=.1)
        elif len(x): a.hist(x,bins=60,density=True,alpha=.65); a.axvline(np.mean(x),color="red",ls="--",label=f"均值={np.mean(x):.3g}"); a.legend()
        a.set(xlabel="时间 (s)" if a in (ax[0,0],ax[0,1]) else label,ylabel=label if a in (ax[0,0],ax[0,1]) else "概率密度")
    _save(fig,out,"plot_12_residuals")

def plot_13(d,out):
    s=d["sat"]
    if s.empty:return _unavailable(out,"plot_13_cycle_slip",".stat 中无周跳标记")
    x=s[s.slipc>0]; fig,ax=plt.subplots(1,2,figsize=(11,4)); _title(fig,"13 · RTKLIB 周跳标记")
    if not x.empty: ax[0].scatter(x.time,x.prn,s=8,c=x.slipc,cmap="tab10");
    ax[0].set(xlabel="时间 (s)",ylabel="卫星 PRN",title=f"$SAT slipc>0: {len(x)} 条记录"); ax[0].grid(alpha=.2)
    count=s[s.slipc>0].groupby("time").prn.nunique(); all_t=np.sort(s.time.unique()); y=pd.Series(count,index=count.index).reindex(all_t,fill_value=0).to_numpy(); ax[1].step(all_t,y,where="post",label="每历元涉及卫星数"); ax[1].plot(all_t,np.cumsum(y),label="累计"); ax[1].set(xlabel="时间 (s)",ylabel="数量"); ax[1].legend(); ax[1].grid(alpha=.2); _save(fig,out,"plot_13_cycle_slip")

def plot_14(d,out):
    p=d["pos"]
    fig,ax=plt.subplots(2,2,figsize=(11,7)); _title(fig,"14 · .pos 协方差与误差覆盖")
    for a,c in zip(ax[0],["sigE","sigN","sigU"]): a.plot(p.time,p[c],label=c); a.set_yscale("log"); a.set(xlabel="时间 (s)",ylabel="标准差 (m)"); a.grid(alpha=.2); a.legend()
    ax[1,0].plot(p.time,p.rhoEN); ax[1,0].set(xlabel="时间 (s)",ylabel="相关系数",title="E-N 相关"); ax[1,0].grid(alpha=.2)
    if d["has_truth"]:
        ratio=np.abs(p.de)/np.maximum(1.96*p.sigE,1e-12); ax[1,1].plot(p.time,ratio,lw=.6); ax[1,1].axhline(1,color="red",ls="--"); ax[1,1].set(xlabel="时间 (s)",ylabel="|ΔE|/(1.96σE)",title=f"E 軸誤差包絡覆盖 {(ratio<=1).mean()*100:.1f}%")
    else: ax[1,1].text(.5,.5,"缺真值：无法计算误差覆盖率",ha="center",va="center",transform=ax[1,1].transAxes)
    _save(fig,out,"plot_14_covariance")

def _unavailable(out,name,message):
    fig,ax=plt.subplots(figsize=(8,3)); ax.axis("off"); ax.text(.5,.58,name,ha="center",fontsize=15,weight="bold"); ax.text(.5,.4,message,ha="center",wrap=True); _save(fig,out,name)

def plot_15(d,out):
    # RTKLIB .stat here exposes residuals but not innovations or their covariance.
    _unavailable(out,"plot_15_innovation_nis","输入文件没有滤波创新向量 v 或创新协方差 S；NIS=vᵀS⁻¹v 无法从 .pos/.stat 可靠恢复。")

PLOTTERS=[plot_01,plot_02,plot_03,plot_04,plot_05,plot_06,plot_07,plot_08,plot_09,plot_10,plot_11,plot_12,plot_13,plot_14,plot_15]

def make_dashboard(d,out):
    p=d["pos"]; fig,ax=plt.subplots(2,3,figsize=(15,8)); _title(fig,"RTKLIB 解算质量总览 Dashboard")
    ax[0,0].plot(p.E,p.N,lw=.7); ax[0,0].set(title="ENU 轨迹",xlabel="E (m)",ylabel="N (m)"); ax[0,0].axis("equal")
    if d["has_truth"]:
        ax[0,1].plot(p.time,p.eh,lw=.6,label="水平误差"); ax[0,1].plot(p.time,p.e3d,lw=.6,label="3D误差"); ax[0,1].legend(); ax[0,1].set(title="定位误差（纵轴限至 99.5% 分位）",xlabel="时间 (s)",ylabel="m"); ax[0,1].set_ylim(0,p.e3d.quantile(.995)*1.1)
    else: ax[0,1].text(.5,.5,"无真值",ha="center",transform=ax[0,1].transAxes)
    ax[0,2].step(p.time,p.q,where="post"); ax[0,2].set(title=f"FIX 比例 {(p.q==1).mean()*100:.1f}%",xlabel="时间 (s)",ylabel="Q")
    ax[1,0].plot(p.time,p.ratio,lw=.5); ax[1,0].axhline(3,ls="--",color="red"); ax[1,0].set(title="AR Ratio",xlabel="时间 (s)",ylabel="Ratio")
    if not d["sat"].empty:
        e=_sat_epoch(d); ax[1,1].plot(e.time,e.tracked,label="tracked"); ax[1,1].plot(e.time,e.used,label="used"); ax[1,1].legend(); ax[1,1].set(title="卫星数",xlabel="时间 (s)",ylabel="颗")
        ss=d["sat"].query("freq == 1 and snr > 0"); ax[1,2].scatter(ss.el,ss.snr,s=2,alpha=.12); ax[1,2].set(title="高度角与 C/N₀（L1，有效信号）",xlabel="高度角 (°)",ylabel="dB-Hz")
    for a in ax.flat: a.grid(alpha=.18)
    _save(fig,out,"dashboard")

def main():
    global EXPORT_FORMATS, EXPORT_DPI
    ap=argparse.ArgumentParser(description="RTKLIB 结果文件诊断绘图工具：读取 .pos/.stat/真值并导出分析图")
    ap.add_argument("--pos",type=Path,default=DEFAULT_POS,help="RTKLIB .pos 解算结果")
    ap.add_argument("--reference",type=Path,default=DEFAULT_REF,help="可选真值 CSV；设为 none 可禁用")
    ap.add_argument("--stat",type=Path,default=None,help="可选 .pos.stat 文件，默认自动读取 <pos>.stat")
    ap.add_argument("--output",type=Path,default=DEFAULT_OUT,help="图片输出目录")
    ap.add_argument("--plots",default="all",help="all 或逗号分隔的图号，例如 1,2,3")
    ap.add_argument("--format",default="png",help="导出格式，可多选：png,pdf,svg，例如 png,pdf")
    ap.add_argument("--dpi",type=int,default=180,help="PNG 分辨率")
    args=ap.parse_args()
    EXPORT_FORMATS=tuple(dict.fromkeys(x.strip().lower() for x in args.format.split(",") if x.strip()))
    bad=set(EXPORT_FORMATS)-{"png","pdf","svg"}
    if not EXPORT_FORMATS or bad: ap.error(f"不支持的格式：{', '.join(sorted(bad))}")
    if args.dpi < 72: ap.error("--dpi 至少为 72")
    EXPORT_DPI=args.dpi
    if not args.pos.exists(): ap.error(f"找不到 .pos 文件：{args.pos}")
    ref=None if str(args.reference).lower()=="none" else args.reference
    d=load_data(args.pos,ref,args.stat); args.output.mkdir(parents=True,exist_ok=True)
    chosen=range(1,16) if args.plots.lower()=="all" else sorted(set(int(x) for x in args.plots.split(",")))
    if any(i < 1 or i > 15 for i in chosen): ap.error("图号范围为 1–15")
    for i in chosen:
        PLOTTERS[i-1](d,args.output); print(f"Plot {i:02d}: done")
    make_dashboard(d,args.output); print(f"Dashboard: {args.output/f'dashboard.{EXPORT_FORMATS[0]}'}")
    print(f"Epochs: {len(d['pos'])}; satellite records: {len(d['sat'])}; truth matched: {d['has_truth']}")

if __name__=="__main__": main()
