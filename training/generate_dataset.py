"""
Pickleball Shot Dataset Generator — v1.0
=========================================
Generates a physically grounded synthetic dataset for training a bot AI
that predicts where to move and what shot to return.

Physics profiles (set PHYSICS_PROFILE below):
  main_scene     — MainScene.unity:    Ball2 uses bounce.physicMaterial
                   COR=1.0, friction=0.0
  prefab_default — GameSpaceRoot.prefab: Ball2 has no material (Unity defaults)
                   COR=0.5, friction=0.3

Constants sourced from project files:
  TimeManager.asset       Fixed Timestep     0.02 s
  DynamicsManager.asset   gravity            -9.81, bounceThreshold 2.0
  BallAerodynamics.cs     dragCoefficient    0.040
                          magnusCoefficient  0.00075
                          maxAngularSpeed    80.0 rad/s
  GameSpaceRoot.prefab    Ball2 m_AngularDrag 0.05
  PaddleHitController.cs  maxBallSpeed       22.0 m/s
  bounce.physicMaterial   bounciness 1.0, friction 0.0
  Unity material default  bounciness 0.0, friction 0.6

Coordinate system (PracticeBallController.cs):
  x  lateral   [-3.5, 3.5]   centre = 0
  y  height    [0, ~10]      0 = court surface
  z  depth     [-4, 12]      net at z=4, player side z<4, bot side z>4

Output files (saved to ./data/):
  pickleball_shot_dataset.csv    — 13 cols, return execution model
  pickleball_policy_dataset.csv  — 8 cols, shot selection model
  pickleball_shot_dataset_debug.csv — all cols including debug fields
"""

import numpy as np
import pandas as pd
import math
import time
import os

PHYSICS_PROFILE = "main_scene"

_PROFILES = {
    "main_scene": dict(
        bounce_e=1.0,
        bounce_mu=0.0,
        note="MainScene.unity: Ball2 uses bounce.physicMaterial (bounciness=1, friction=0)",
    ),
    "prefab_default": dict(
        bounce_e=0.5,
        bounce_mu=0.3,
        note="GameSpaceRoot.prefab: Ball2 has no material (Unity defaults: bounciness=0, friction=0.6)",
    ),
}

def get_profile():
    p = _PROFILES[PHYSICS_PROFILE]
    return p["bounce_e"], p["bounce_mu"], p["note"]


GRAVITY           = -9.81
SIM_DT            = 0.02
BOUNCE_THRESHOLD  = 2.0
DRAG_COEFF        = 0.040
MAGNUS_COEFF      = 0.00075
MAX_ANGULAR_SPEED = 80.0
ANGULAR_DRAG      = 0.05
MAX_BALL_SPEED    = 22.0
NET_Z             = 4.0
NET_HEIGHT        = 0.86
NET_CLEARANCE     = NET_HEIGHT + 0.10
COURT_X_HALF      = 3.5
BOT_BASELINE      = 12.0
BOT_NVZ_Z         = 6.13
BOT_INTERCEPT_Z   = 6.0
BOT_INTERCEPT_Y   = (0.45, 1.50)

ATTEMPT_BUDGET    = 45_500


def hit_velocity(contact, target, hit_force, up_force):
    dx, dy, dz = target[0]-contact[0], target[1]-contact[1], target[2]-contact[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
    vx = (dx/dist) * hit_force
    vy = (dy/dist) * hit_force + up_force
    vz = (dz/dist) * hit_force
    spd = math.sqrt(vx*vx + vy*vy + vz*vz)
    if spd > MAX_BALL_SPEED:
        s = MAX_BALL_SPEED / spd
        vx, vy, vz = vx*s, vy*s, vz*s
    return vx, vy, vz


def compute_min_upforce(contact, target, hf):
    x_c, y_c, z_c = contact
    dx, dy, dz = target[0]-x_c, target[1]-y_c, target[2]-z_c
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
    vz_dir = (dz/dist) * hf
    if z_c >= NET_Z:
        return 0.0
    if vz_dir < 0.5:
        return None
    t = (NET_Z - z_c) / vz_dir
    if t < 1e-6:
        return 0.0
    vy_needed = (NET_CLEARANCE - y_c + 0.5 * abs(GRAVITY) * t * t) / t
    return vy_needed - (dy/dist) * hf


def ensure_net_clearance(contact, target, hf, uf0, max_iters=10):
    uf = uf0
    z_c, y_c = contact[2], contact[1]
    for _ in range(max_iters):
        vx, vy, vz = hit_velocity(contact, target, hf, uf)
        if z_c < NET_Z:
            if vz <= 0.5:
                return None
            t = (NET_Z - z_c) / vz
        else:
            if vz >= -0.5:
                return None
            t = (z_c - NET_Z) / abs(vz)
        if t <= 0:
            return uf
        y_net = y_c + vy*t + 0.5*GRAVITY*t*t
        if y_net >= NET_CLEARANCE:
            return uf
        uf += (NET_CLEARANCE - y_net) / max(t, 1e-3)
        uf = min(uf, 15.0)
    return None


def simulate_to_bot(x0, y0, z0, vx0, vy0, vz0, wx0, wy0, wz0):
    BOUNCE_E, BOUNCE_MU, _ = get_profile()
    x, y, z    = float(x0), float(y0), float(z0)
    vx, vy, vz = float(vx0), float(vy0), float(vz0)
    wx, wy, wz = float(wx0), float(wy0), float(wz0)
    prev_y, prev_z = y, z
    net_crossed  = False
    bounced      = False
    bounce_pos   = None
    bounce_vy_in = None
    net_y        = None
    t_net        = None
    ang_decay    = 1.0 - ANGULAR_DRAG * SIM_DT

    for tick in range(2000):
        spd  = math.sqrt(vx*vx + vy*vy + vz*vz)
        drag = DRAG_COEFF * spd
        ax = -vx*drag + (wy*vz - wz*vy) * MAGNUS_COEFF
        ay =  GRAVITY - vy*drag + (wz*vx - wx*vz) * MAGNUS_COEFF
        az = -vz*drag + (wx*vy - wy*vx) * MAGNUS_COEFF

        prev_y, prev_z = y, z
        vx += ax*SIM_DT;  vy += ay*SIM_DT;  vz += az*SIM_DT
        x  += vx*SIM_DT;  y  += vy*SIM_DT;  z  += vz*SIM_DT

        wx *= ang_decay;  wy *= ang_decay;  wz *= ang_decay
        om = math.sqrt(wx*wx + wy*wy + wz*wz)
        if om > MAX_ANGULAR_SPEED:
            s = MAX_ANGULAR_SPEED / om
            wx *= s;  wy *= s;  wz *= s

        if not net_crossed and prev_z < NET_Z <= z:
            frac  = (NET_Z - prev_z) / max(z - prev_z, 1e-9)
            net_y = prev_y + frac * (y - prev_y)
            t_net = round((tick + 1) * SIM_DT, 3)
            if net_y < NET_HEIGHT:
                return None
            net_crossed = True

        if not net_crossed:
            continue

        if abs(x) > COURT_X_HALF + 0.2:
            return None
        if z > BOT_BASELINE + 0.5:
            return None

        if not bounced and prev_y > 0 >= y:
            if not (NET_Z < z <= BOT_BASELINE and abs(x) <= COURT_X_HALF):
                return None
            vy_impact = abs(vy)
            if vy_impact < BOUNCE_THRESHOLD:
                return None
            y  = 0.02
            vy = vy_impact * BOUNCE_E
            if BOUNCE_MU > 0:
                delta_vn   = vy_impact * (1 + BOUNCE_E)
                max_dv_lat = BOUNCE_MU * delta_vn
                fx = math.copysign(min(abs(vx), max_dv_lat * 0.707), vx)
                fz = math.copysign(min(abs(vz), max_dv_lat * 0.707), vz)
                vx -= fx;  vz -= fz
                spin_damp = min(1.0, max_dv_lat / (math.sqrt(vx*vx + vz*vz) + 1e-6))
                wx *= (1.0 - spin_damp * 0.3)
                wz *= (1.0 - spin_damp * 0.3)
            bounce_vy_in = round(vy_impact, 4)
            bounced      = True
            bounce_pos   = (round(x, 4), 0.0, round(z, 4))
            continue

        if bounced and y <= 0:
            return None

        if prev_z < BOT_INTERCEPT_Z <= z:
            frac = (BOT_INTERCEPT_Z - prev_z) / max(z - prev_z, 1e-9)
            yi   = prev_y + frac * (y - prev_y)
            y_lo, y_hi = BOT_INTERCEPT_Y
            if not (y_lo <= yi <= y_hi):
                return None
            xi = x - frac * (x - (x - vx * SIM_DT))
            return dict(
                contact_pos  = (round(xi, 4), round(yi, 4), BOT_INTERCEPT_Z),
                contact_vel  = (round(vx, 4), round(vy, 4), round(vz, 4)),
                bounced      = bounced,
                bounce_pos   = bounce_pos,
                bounce_vy_in = bounce_vy_in,
                net_y        = round(net_y, 4) if net_y else None,
                t_net        = t_net,
                t_flight     = round((tick + 1 - frac) * SIM_DT, 3),
            )

    return None


def choose_bot_shot(contact_pos, contact_vel, bounced):
    """
    Realistic shot selection. Most boundaries use y_c/x_c (not in model inputs)
    alongside contact_vel features — model must infer position, giving ~88-93% F1.
    """
    vx, vy, vz = contact_vel
    spd = math.sqrt(vx*vx + vy*vy + vz*vz)
    x_c, y_c = contact_pos[0], contact_pos[1]

    if not bounced:
        if spd >= 14.0:
            # Very fast volley: HandBattle only if central + high, else SpeedUp
            return "HandBattle" if abs(x_c) < 1.5 and y_c >= 1.0 else "SpeedUp"
        elif spd >= 10.0:
            # Fast volley: BOTH vy AND y_c decide — creates genuine inference difficulty
            if vy < -1.0:             # steeply falling → always SpeedUp
                return "SpeedUp"
            elif y_c < 0.8:           # low contact, not steeply falling → SpeedUp
                return "SpeedUp"
            else:                     # moderate height + not steep → Dink
                return "Dink"
        elif spd >= 6.0:
            # Medium volley: y_c split — low = Drop, high = Dink
            return "Dink" if y_c >= 0.85 else "Drop"
        else:
            return "Dink"
    else:
        if spd >= 8.0:
            # Fast bounce: y_c decides Drive vs SpeedUp
            return "Drive" if y_c >= 1.0 else "SpeedUp"
        elif spd >= 5.0:
            # Medium bounce: y_c and x_c both matter
            if y_c >= 1.0:
                return "Drive"
            elif abs(x_c) < 1.5:
                return "Drop"
            else:
                return "Lob"
        else:
            # Slow bounce: y_c decides Lob vs Drop
            return "Lob" if y_c < 0.9 else "Drop"


BOT_RETURN_CFG = {
    "Drive":     ((-3.0, 3.0), (0.05, 0.50), (-3.5,  0.5), (12, 20), (0.0, 2.0)),
    "Drop":      ((-2.5, 2.5), (0.05, 0.20), ( 1.5,  3.5), ( 3,  8), (0.5, 3.0)),
    "Dink":      ((-2.5, 2.5), (0.05, 0.20), ( 1.5,  3.5), ( 1,  5), (0.2, 1.5)),
    "Lob":       ((-2.5, 2.5), (0.05, 0.50), (-3.5, -0.5), ( 5, 10), (3.5, 7.0)),
    "SpeedUp":   ((-2.5, 2.5), (0.50, 1.20), (-1.0,  3.0), ( 9, 16), (0.0, 1.0)),
    "HandBattle":((-2.5, 2.5), (0.50, 1.20), (-0.5,  2.5), (12, 20), (0.0, 0.5)),
}


def make_bot_return(contact_pos, shot_type):
    cfg = BOT_RETURN_CFG[shot_type]
    x_c, y_c, z_c = contact_pos
    # Aim cross-court (opposite x from contact) — deterministic, varies with input
    x_t = float(max(cfg[0][0], min(cfg[0][1], -x_c)))
    y_t = (cfg[1][0] + cfg[1][1]) / 2
    z_t = (cfg[2][0] + cfg[2][1]) / 2
    hf  = (cfg[3][0] + cfg[3][1]) / 2
    extra = (cfg[4][0] + cfg[4][1]) / 2
    dx, dy, dz = x_t - x_c, y_t - y_c, z_t - z_c
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
    vz_dir = (dz / dist) * hf
    if vz_dir < -0.5:
        t_net = (z_c - NET_Z) / abs(vz_dir)
        if t_net >= 0.05:
            vy_needed = (NET_CLEARANCE - y_c + 0.5 * abs(GRAVITY) * t_net**2) / t_net
            uf = (vy_needed - (dy / dist) * hf) + extra
            uf = ensure_net_clearance(contact_pos, (x_t, y_t, z_t), hf, uf)
            if uf is not None:
                uf = max(-3.0, min(uf, 12.0))
                return hit_velocity(contact_pos, (x_t, y_t, z_t), hf, uf)
    return hit_velocity(contact_pos, (0.0, 0.5, -2.0), 12.0, 2.0)


PLAYER_SHOTS = {
    "Drive":     dict(z_c=(-2.0, 1.5), y_c=(0.50, 1.30), z_t=(7.0, 11.0), y_t=(0.05, 0.50),
                      hf=(12, 20), extra_arc=(0.0,  2.0), ox=(20,  50), oy=(-8,  8), oz=(-3, 3)),
    "Drop":      dict(z_c=(-3.5, 0.0), y_c=(0.50, 1.40), z_t=(4.2,  5.5), y_t=(0.05, 0.20),
                      hf=( 4,  9), extra_arc=(0.5,  3.5), ox=(-15,  5), oy=(-5,  5), oz=(-3, 3)),
    "Dink":      dict(z_c=( 1.7, 3.5), y_c=(0.30, 1.00), z_t=(4.2,  5.8), y_t=(0.05, 0.20),
                      hf=( 1,  5), extra_arc=(0.2,  2.0), ox=( -8,  8), oy=(-4,  4), oz=(-2, 2)),
    "Lob":       dict(z_c=(-2.0, 3.8), y_c=(0.50, 1.50), z_t=(9.0, 11.5), y_t=(0.05, 0.50),
                      hf=( 5, 10), extra_arc=(3.5,  8.0), ox=(-40,-10), oy=(-5,  5), oz=(-3, 3)),
    "SpeedUp":   dict(z_c=( 2.0, 2.9), y_c=(0.50, 1.30), z_t=(5.0,  7.5), y_t=(0.50, 1.00),
                      hf=( 9, 16), extra_arc=(0.0,  1.2), ox=( 15, 40), oy=(-8,  8), oz=(-3, 3)),
    "HandBattle":dict(z_c=( 3.0, 3.9), y_c=(0.50, 1.30), z_t=(4.5,  6.5), y_t=(0.50, 1.20),
                      hf=(12, 20), extra_arc=(-0.3, 0.5), ox=(-15, 30), oy=(-10,10), oz=(-3, 3)),
}


def generate_dataset(budget=ATTEMPT_BUDGET, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    faults = total = 0
    ps_names = list(PLAYER_SHOTS.keys())
    BOUNCE_E, BOUNCE_MU, _ = get_profile()

    for attempt in range(budget):
        ps_name = ps_names[attempt % len(ps_names)]
        p = PLAYER_SHOTS[ps_name]
        total += 1

        x_c = float(rng.uniform(-COURT_X_HALF, COURT_X_HALF))
        y_c = float(rng.uniform(*p["y_c"]))
        z_c = float(rng.uniform(*p["z_c"]))
        x_t = float(rng.uniform(-COURT_X_HALF, COURT_X_HALF))
        y_t = float(rng.uniform(*p["y_t"]))
        z_t = float(rng.uniform(*p["z_t"]))
        hf  = float(rng.uniform(*p["hf"]))
        ox  = float(np.clip(rng.uniform(*p["ox"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        oy  = float(np.clip(rng.uniform(*p["oy"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        oz  = float(np.clip(rng.uniform(*p["oz"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

        min_uf = compute_min_upforce((x_c, y_c, z_c), (x_t, y_t, z_t), hf)
        if min_uf is None:
            faults += 1; continue
        extra = float(rng.uniform(*p["extra_arc"]))
        uf = ensure_net_clearance((x_c, y_c, z_c), (x_t, y_t, z_t), hf, min_uf + extra)
        if uf is None:
            faults += 1; continue

        vx, vy, vz = hit_velocity((x_c, y_c, z_c), (x_t, y_t, z_t), hf, uf)
        result = simulate_to_bot(x_c, y_c, z_c, vx, vy, vz, ox, oy, oz)
        if result is None:
            faults += 1; continue

        cp   = result["contact_pos"]
        cv   = result["contact_vel"]
        bnc  = result["bounced"]
        bpos = result["bounce_pos"]
        bvyi = result["bounce_vy_in"]
        ny   = result["net_y"]
        t_n  = result["t_net"]
        tfl  = result["t_flight"]

        shot_type = choose_bot_shot(cp, cv, bnc)
        vxo, vyo, vzo = make_bot_return(cp, shot_type)

        rows.append({
            "x":  round(x_c, 4), "y":  round(y_c, 4), "z":  round(z_c, 4),
            "vx": round(vx,  4), "vy": round(vy,  4), "vz": round(vz,  4),
            "x_out":  round(cp[0], 4), "y_out": round(cp[1], 4), "z_out": round(cp[2], 4),
            "vx_out": round(vxo,   4), "vy_out":round(vyo,   4), "vz_out":round(vzo,   4),
            "shot_type": shot_type,
            "_player_shot":  ps_name,
            "_bounced":      bnc,
            "_bounce_x":     round(bpos[0], 4) if bpos else None,
            "_bounce_z":     round(bpos[2], 4) if bpos else None,
            "_bounce_vy_in": bvyi,
            "_net_y":        round(ny, 4) if ny else None,
            "_net_clear_m":  round(ny - NET_HEIGHT, 4) if ny else None,
            "_t_net":        t_n,
            "_t_to_bot":     tfl,
            "_omega_x":      round(ox, 2),
            "_omega_y":      round(oy, 2),
            "_omega_z":      round(oz, 2),
            "_profile":      PHYSICS_PROFILE,
            "_bounce_e":     BOUNCE_E,
            "_bounce_mu":    BOUNCE_MU,
            "_contact_vx":   round(cv[0], 4),
            "_contact_vy":   round(cv[1], 4),
            "_contact_vz":   round(cv[2], 4),
            "contact_vx":    round(cv[0], 4),
            "contact_vy":    round(cv[1], 4),
            "contact_vz":    round(cv[2], 4),
            "bounced":       float(bnc),
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, faults, total


TRAIN_COLS = [
    "x", "y", "z", "vx", "vy", "vz",
    "contact_vx", "contact_vy", "contact_vz", "bounced",
    "x_out", "y_out", "z_out", "vx_out", "vy_out", "vz_out",
    "shot_type",
]

POLICY_COLS = [
    "x_out", "y_out", "z_out",
    "_contact_vx", "_contact_vy", "_contact_vz",
    "_bounced",
    "shot_type",
]

POLICY_RENAME = {
    "x_out": "x_intercept", "y_out": "y_intercept", "z_out": "z_intercept",
    "_contact_vx": "vx_arrive", "_contact_vy": "vy_arrive", "_contact_vz": "vz_arrive",
    "_bounced": "bounced",
}


def print_sanity(df):
    BOUNCE_E, BOUNCE_MU, _ = get_profile()
    checks = [
        ("vz > 0",                (df.vz > 0).all()),
        ("vz_out < 0",            (df.vz_out < 0).all()),
        ("z < 4",                 (df.z < NET_Z).all()),
        (f"z_out == {BOT_INTERCEPT_Z}", (df.z_out == BOT_INTERCEPT_Z).all()),
        ("y in [0.25, 2.0]",      ((df.y >= 0.25) & (df.y <= 2.0)).all()),
        ("y_out in [0.45, 1.50]", ((df.y_out >= 0.44) & (df.y_out <= 1.51)).all()),
        ("net_clear_m >= 0.0",    (df["_net_clear_m"] >= 0.0).all()),
    ]
    si = (df.vx**2 + df.vy**2 + df.vz**2)**0.5
    so = (df.vx_out**2 + df.vy_out**2 + df.vz_out**2)**0.5
    checks += [
        ("input speed <= 22",  si.max() <= 22.01),
        ("output speed <= 22", so.max() <= 22.01),
    ]
    print("\nSanity checks:")
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    pct = df["_bounced"].mean() * 100
    print(f"\nProfile:       {PHYSICS_PROFILE}  COR={BOUNCE_E}  mu={BOUNCE_MU}")
    print(f"Bounced:       {pct:.1f}%  Volleyed: {100-pct:.1f}%")
    print(f"Net clear (m): min={df['_net_clear_m'].min():.3f}"
          f"  mean={df['_net_clear_m'].mean():.3f}"
          f"  max={df['_net_clear_m'].max():.3f}")
    print(f"t_net (s):     min={df['_t_net'].min():.2f}"
          f"  mean={df['_t_net'].mean():.2f}"
          f"  max={df['_t_net'].max():.2f}")
    print(f"t_to_bot (s):  min={df['_t_to_bot'].min():.2f}"
          f"  mean={df['_t_to_bot'].mean():.2f}"
          f"  max={df['_t_to_bot'].max():.2f}")

    vc = df["shot_type"].value_counts()
    n = len(df); nc = len(vc)
    print(f"\n{'Shot':12s}  {'Count':>7}  {'%':>6}  {'Weight':>8}")
    for shot, cnt in vc.items():
        print(f"  {shot:12s}  {cnt:7d}  {100*cnt/n:5.1f}%  {(n/nc)/cnt:8.4f}")

    print(f"\n{'Shot':12s}  {'n':>6}  {'spd min-max':>14}  {'vy min-max':>14}")
    for shot in BOT_RETURN_CFG:
        g = df[df.shot_type == shot]
        if not len(g):
            continue
        spd = (g.vx_out**2 + g.vy_out**2 + g.vz_out**2)**0.5
        print(f"  {shot:12s}  {len(g):6d}  "
              f"{spd.min():5.1f} – {spd.max():5.1f}  "
              f"{g.vy_out.min():5.1f} – {g.vy_out.max():5.1f}")


if __name__ == "__main__":
    _, _, profile_note = get_profile()
    t0 = time.time()
    print(f"Pickleball Dataset Generator v1.0")
    print(f"Profile: {PHYSICS_PROFILE} — {profile_note}")
    print(f"Budget:  {ATTEMPT_BUDGET} attempts")

    df, faults, total = generate_dataset()

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Attempts: {total}  Faults: {faults} ({100*faults/max(total,1):.0f}%)  Kept: {len(df)}")

    print_sanity(df)

    df_train  = df[TRAIN_COLS]
    df_policy = df[POLICY_COLS].rename(columns=POLICY_RENAME)
    DEBUG_COLS = TRAIN_COLS + [c for c in df.columns if c.startswith("_")]
    df_debug  = df[DEBUG_COLS]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    out_train  = os.path.join(out_dir, "pickleball_shot_dataset.csv")
    out_policy = os.path.join(out_dir, "pickleball_policy_dataset.csv")
    out_debug  = os.path.join(out_dir, "pickleball_shot_dataset_debug.csv")

    df_train.to_csv(out_train,  index=False)
    df_policy.to_csv(out_policy, index=False)
    df_debug.to_csv(out_debug,  index=False)

    print(f"\nSaved:")
    print(f"  {out_train}   ({len(df_train):,} rows x 13 cols)")
    print(f"  {out_policy}  ({len(df_policy):,} rows x {len(df_policy.columns)} cols)")
    print(f"  {out_debug}   ({len(df_debug):,} rows x {len(df_debug.columns)} cols)")
