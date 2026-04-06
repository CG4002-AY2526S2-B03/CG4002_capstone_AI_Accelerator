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


#Physics constants
GRAVITY           = -9.81    #acts on AI z (height)
SIM_DT            = 0.02
BOUNCE_THRESHOLD  = 2.0      #normal component (AI z) at floor
DRAG_COEFF        = 0.040
MAGNUS_COEFF      = 0.00075
MAX_ANGULAR_SPEED = 80.0
ANGULAR_DRAG      = 0.05
MAX_BALL_SPEED    = 22.0

#Court geometry in AI frame (x=right, y=depth, z=height)
NET_Y             = 5.4      #net plane depth (CourtBoundarySetup: localPosition.z=5.4)
NET_TOP_Z         = 0.9      #net top height (localPosition.y=0.45, size.y=0.9 -> top=0.9)
NET_CLEARANCE_Z   = NET_TOP_Z + 0.10   #sampling safety margin only, not used in sim
FLOOR_Z           = 0.0

COURT_X_MIN       = -3.5
COURT_X_MAX       =  3.5
BOT_Y_MAX         = 12.2     #BotHitController.courtMaxZ
BOT_INTERCEPT_Z   = (0.45, 1.50)   #reachable strike height range (AI z)

#Bounce threshold mode
#"relative": use full ball speed at impact (matches Unity m_BounceThreshold behavior
#            — Unity compares relative speed of the two objects at contact;
#            for a stationary floor, relative speed = ball speed)
#"normal":   use only the normal component abs(vz) — physically purer but not Unity
BOUNCE_THRESHOLD_MODE = "relative"

ATTEMPT_BUDGET    = 65_000


#Hit formula (Player.cs / Bot.cs / BotHitController.cs)
#velocity = dir.normalized * hitForce + Vector3.up * upForce
#Vector3.up = +z in AI frame (height axis)
def hit_velocity(contact, target, hit_force, up_force):
    dx = target[0] - contact[0]
    dy = target[1] - contact[1]
    dz = target[2] - contact[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
    vx = (dx/dist) * hit_force
    vy = (dy/dist) * hit_force
    vz = (dz/dist) * hit_force + up_force   #upForce adds to height (AI z)
    spd = math.sqrt(vx*vx + vy*vy + vz*vz)
    if spd > MAX_BALL_SPEED:
        s = MAX_BALL_SPEED / spd
        vx, vy, vz = vx*s, vy*s, vz*s
    return vx, vy, vz


#Min upForce to clear net (generator only, uses NET_CLEARANCE_Z)
def compute_min_upforce(contact, target, hf):
    x_c, y_c, z_c = contact
    dx = target[0] - x_c
    dy = target[1] - y_c
    dz = target[2] - z_c
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
    vy_dir = (dy/dist) * hf    #depth component of velocity
    if y_c >= NET_Y:
        return 0.0
    if vy_dir < 0.5:
        return None
    t = (NET_Y - y_c) / vy_dir
    if t < 1e-6:
        return 0.0
    #kinematic: z_net = z_c + vz*t + 0.5*GRAVITY*t^2 (GRAVITY is negative)
    #solving for vz: vz_needed = (NET_CLEARANCE_Z - z_c - 0.5*GRAVITY*t^2) / t
    vz_needed = (NET_CLEARANCE_Z - z_c - 0.5 * GRAVITY * t * t) / t
    return vz_needed - (dz/dist) * hf


def ensure_net_clearance(contact, target, hf, uf0, max_iters=10):
    uf = uf0
    y_c, z_c = contact[1], contact[2]
    for _ in range(max_iters):
        vx, vy, vz = hit_velocity(contact, target, hf, uf)
        if y_c < NET_Y:
            if vy <= 0.5:
                return None
            t = (NET_Y - y_c) / vy
        else:
            if vy >= -0.5:
                return None
            t = (y_c - NET_Y) / abs(vy)
        if t <= 0:
            return uf
        z_net = z_c + vz*t + 0.5*GRAVITY*t*t
        if z_net >= NET_CLEARANCE_Z:
            return uf
        uf += (NET_CLEARANCE_Z - z_net) / max(t, 1e-3)
        uf = min(uf, 15.0)
    return None


#Simulator
#AI frame: x=right, y=depth, z=height
#Gravity on z. Net crossing on y. Floor bounce on z=0.
def simulate_to_bot(x0, y0, z0, vx0, vy0, vz0, wx0, wy0, wz0):
    BOUNCE_E, BOUNCE_MU, _ = get_profile()
    x, y, z    = float(x0), float(y0), float(z0)
    vx, vy, vz = float(vx0), float(vy0), float(vz0)
    wx, wy, wz = float(wx0), float(wy0), float(wz0)
    prev_y, prev_z, prev_x = y, z, x
    net_crossed  = False
    bounced      = False
    bounce_pos   = None
    bounce_vz_in = None
    net_z        = None
    t_net        = None
    ang_decay    = 1.0 - ANGULAR_DRAG * SIM_DT

    for tick in range(2000):
        spd  = math.sqrt(vx*vx + vy*vy + vz*vz)
        drag = DRAG_COEFF * spd
        #Drag + Magnus. Gravity on z only.
        ax = -vx*drag + (wy*vz - wz*vy) * MAGNUS_COEFF
        ay = -vy*drag + (wz*vx - wx*vz) * MAGNUS_COEFF
        az =  GRAVITY - vz*drag + (wx*vy - wy*vx) * MAGNUS_COEFF

        #Semi-implicit Euler: velocity first, then position
        prev_y, prev_z, prev_x = y, z, x
        vx += ax*SIM_DT;  vy += ay*SIM_DT;  vz += az*SIM_DT
        x  += vx*SIM_DT;  y  += vy*SIM_DT;  z  += vz*SIM_DT

        #Angular drag + clamp
        wx *= ang_decay;  wy *= ang_decay;  wz *= ang_decay
        om = math.sqrt(wx*wx + wy*wy + wz*wz)
        if om > MAX_ANGULAR_SPEED:
            s = MAX_ANGULAR_SPEED / om
            wx *= s;  wy *= s;  wz *= s

        #Net check: y crosses NET_Y. Ball must clear NET_TOP_Z.
        if not net_crossed and prev_y < NET_Y <= y:
            frac  = (NET_Y - prev_y) / max(y - prev_y, 1e-9)
            net_z = prev_z + frac * (z - prev_z)
            t_net = round(tick * SIM_DT + frac * SIM_DT, 4)   #interpolated
            if net_z < NET_TOP_Z:
                return None
            net_crossed = True

        if not net_crossed:
            continue

        #Bounds
        if x < COURT_X_MIN - 0.2 or x > COURT_X_MAX + 0.2:
            return None
        if y > BOT_Y_MAX + 0.5:
            return None

        #Floor bounce: z crosses 0 on bot side
        if not bounced and prev_z > FLOOR_Z >= z:
            if not (NET_Y < y <= BOT_Y_MAX and COURT_X_MIN <= x <= COURT_X_MAX):
                return None

            #Threshold check (point 3: Unity uses full relative speed)
            impact_speed = math.sqrt(vx*vx + vy*vy + vz*vz)  #relative to stationary floor
            vz_impact    = abs(vz)                             #normal component only
            if BOUNCE_THRESHOLD_MODE == "relative":
                if impact_speed < BOUNCE_THRESHOLD:
                    return None
            else:
                if vz_impact < BOUNCE_THRESHOLD:
                    return None

            z  = 0.02
            vz = vz_impact * BOUNCE_E     #COR applied to normal component only
            if BOUNCE_MU > 0:
                delta_vn   = vz_impact * (1 + BOUNCE_E)
                max_dv_lat = BOUNCE_MU * delta_vn
                vt = math.sqrt(vx*vx + vy*vy)    #pre-scale tangential speed
                if vt > 1e-9:
                    dv = min(vt, max_dv_lat)
                    scale = (vt - dv) / vt
                    vx *= scale;  vy *= scale
                #Spin damp uses pre-scale vt so ratio is consistent
                spin_damp = min(1.0, max_dv_lat / (vt + 1e-6))
                wx *= (1.0 - spin_damp * 0.3)
                wy *= (1.0 - spin_damp * 0.3)
                wz *= (1.0 - spin_damp * 0.3)
            bounce_vz_in = round(vz_impact, 4)
            bounced      = True
            bounce_pos   = (round(x, 4), round(y, 4), FLOOR_Z)
            continue

        if bounced and z <= FLOOR_Z:
            return None

        #Intercept: first tick in bot territory at reachable height
        #vz <= 0.2 excludes strongly-rising shots for more realistic contact timing
        z_lo, z_hi = BOT_INTERCEPT_Z
        if (y > NET_Y + 0.1
                and z_lo <= z <= z_hi
                and COURT_X_MIN <= x <= COURT_X_MAX
                and vz <= 0.2):
            return dict(
                contact_pos  = (round(x, 4), round(y, 4), round(z, 4)),
                contact_vel  = (round(vx, 4), round(vy, 4), round(vz, 4)),
                bounced      = bounced,
                bounce_pos   = bounce_pos,
                bounce_vz_in = bounce_vz_in,
                net_z        = round(net_z, 4) if net_z is not None else None,
                t_net        = t_net,
                t_flight     = round((tick + 1) * SIM_DT, 3),
            )

    return None


#Bot shot policy
def choose_bot_shot(contact_pos, contact_vel, bounced):
    vx, vy, vz = contact_vel
    spd = math.sqrt(vx*vx + vy*vy + vz*vz)
    x_c, y_c, z_c = contact_pos   #y_c=depth, z_c=height

    deep = y_c > 8.0       #ball arrived deep, bot is under pressure
    low  = z_c < 0.75      #ball at low contact height
    high = z_c >= 1.10     #ball at high contact height

    if not bounced:
        if deep:
            return "Drive" if high else "Lob"
        if spd >= 14.0:
            return "HandBattle" if abs(x_c) < 1.5 else "SpeedUp"
        elif spd >= 10.0:
            if high:
                return "Drive"
            elif low:
                return "SpeedUp"
            else:
                return "Dink"
        elif spd >= 6.0:
            if low:
                return "Drop"
            elif high:
                return "Lob"
            else:
                return "Dink"
        else:
            return "Lob" if high else "Drop"
    else:
        #Post-bounce
        if spd >= 8.0:
            return "Drive" if high else "SpeedUp"
        elif spd >= 5.0:
            if high:
                return "Drive"
            elif abs(x_c) < 1.5:
                return "Drop"
            else:
                return "Lob"
        else:
            return "Lob" if low else "Drop"


#Bot return target ranges in AI frame: (x_range, z_range(height), y_target(depth), hf_range, extra_arc)
#Targets are on player side so y_target < NET_Y, vy_out will be negative.
BOT_RETURN_CFG = {
    "Drive":     ((-3.0, 3.0), (0.05, 0.50), (0.5,  4.5), (12, 20), (0.0, 2.0)),
    "Drop":      ((-2.5, 2.5), (0.05, 0.20), (2.0,  4.5), ( 3,  8), (0.5, 3.0)),
    "Dink":      ((-2.5, 2.5), (0.05, 0.20), (2.0,  5.0), ( 1,  5), (0.2, 1.5)),
    "Lob":       ((-2.5, 2.5), (0.05, 0.50), (-2.0, 3.0), ( 5, 10), (6.0, 10.0)),
    "SpeedUp":   ((-2.5, 2.5), (0.50, 1.20), (1.0,  4.0), ( 9, 16), (0.0, 1.0)),
    "HandBattle":((-2.5, 2.5), (0.50, 1.20), (2.0,  5.0), (12, 20), (0.0, 0.5)),
}


def make_bot_return(contact_pos, shot_type, rng):
    cfg = BOT_RETURN_CFG[shot_type]
    x_c, y_c, z_c = contact_pos
    for _ in range(20):
        x_t   = float(rng.uniform(*cfg[0]))
        z_t   = float(rng.uniform(*cfg[1]))
        y_t   = float(rng.uniform(*cfg[2]))
        hf    = float(rng.uniform(*cfg[3]))
        extra = float(rng.uniform(*cfg[4]))
        dx = x_t - x_c;  dy = y_t - y_c;  dz = z_t - z_c
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 1e-6
        vy_dir = (dy/dist) * hf
        if vy_dir > -0.5:
            continue
        t_net = (y_c - NET_Y) / abs(vy_dir)
        if t_net < 0.05:
            continue
        vz_needed = (NET_CLEARANCE_Z - z_c - 0.5 * GRAVITY * t_net**2) / t_net
        uf = (vz_needed - (dz/dist) * hf) + extra
        uf = ensure_net_clearance(contact_pos, (x_t, y_t, z_t), hf, uf)
        if uf is None:
            continue
        uf = max(-3.0, min(uf, 12.0))
        return hit_velocity(contact_pos, (x_t, y_t, z_t), hf, uf)
    return None   #caller must discard this row — no fallback to avoid label mismatch


#Player shot configs in AI frame
#y_c: contact depth (AI y)  y_t: target depth on bot side (AI y > NET_Y)
#z_c: contact height (AI z)  z_t: target height (AI z)
PLAYER_SHOTS = {
    "Drive":     dict(y_c=(-2.0, 1.5), z_c=(0.50, 1.30), y_t=(7.0, 11.0), z_t=(0.05, 0.50),
                      hf=(12, 20), extra_arc=(0.0,  2.0), ox=(20,  50), oy=(-8,  8), oz=(-3, 3)),
    "Drop":      dict(y_c=(-3.5, 0.0), z_c=(0.50, 1.40), y_t=(6.5,  8.5), z_t=(0.05, 0.30),
                      hf=( 4,  9), extra_arc=(0.5,  3.5), ox=(-15,  5), oy=(-5,  5), oz=(-3, 3)),
    "Dink":      dict(y_c=( 1.7, 5.0), z_c=(0.30, 1.00), y_t=(6.0,  7.5), z_t=(0.05, 0.30),
                      hf=( 1,  5), extra_arc=(0.2,  2.0), ox=( -8,  8), oy=(-4,  4), oz=(-2, 2)),
    "Lob":       dict(y_c=(-2.0, 4.5), z_c=(0.50, 1.50), y_t=(9.0, 11.5), z_t=(0.05, 0.50),
                      hf=( 7, 12), extra_arc=(8.0, 12.0), ox=(-40,-10), oy=(-5,  5), oz=(-3, 3)),
    "SpeedUp":   dict(y_c=( 2.0, 4.5), z_c=(0.50, 1.30), y_t=(6.0,  8.5), z_t=(0.50, 1.00),
                      hf=( 9, 16), extra_arc=(0.0,  1.2), ox=( 15, 40), oy=(-8,  8), oz=(-3, 3)),
    "HandBattle":dict(y_c=( 3.5, 5.2), z_c=(0.50, 1.30), y_t=(5.8,  8.0), z_t=(0.50, 1.20),
                      hf=(12, 20), extra_arc=(-0.3, 0.5), ox=(-15, 30), oy=(-10,10), oz=(-3, 3)),
}


#Generator
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

        x_c  = float(rng.uniform(COURT_X_MIN, COURT_X_MAX))
        y_c  = float(rng.uniform(*p["y_c"]))
        z_c  = float(rng.uniform(*p["z_c"]))
        x_t  = float(rng.uniform(COURT_X_MIN, COURT_X_MAX))
        y_t  = float(rng.uniform(*p["y_t"]))
        z_t  = float(rng.uniform(*p["z_t"]))
        hf   = float(rng.uniform(*p["hf"]))
        ox   = float(np.clip(rng.uniform(*p["ox"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        oy   = float(np.clip(rng.uniform(*p["oy"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        oz   = float(np.clip(rng.uniform(*p["oz"]), -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))

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
        bvzi = result["bounce_vz_in"]
        nz   = result["net_z"]
        t_n  = result["t_net"]
        tfl  = result["t_flight"]

        shot_type = choose_bot_shot(cp, cv, bnc)
        ret = make_bot_return(cp, shot_type, rng)
        if ret is None:
            faults += 1; continue
        vxo, vyo, vzo = ret

        rows.append({
            "x":  round(x_c, 4), "y":  round(y_c, 4), "z":  round(z_c, 4),
            "vx": round(vx,  4), "vy": round(vy,  4), "vz": round(vz,  4),
            "contact_vx": round(cv[0], 4),
            "contact_vy": round(cv[1], 4),
            "contact_vz": round(cv[2], 4),
            "bounced":    float(bnc),
            "x_out":  round(cp[0], 4), "y_out": round(cp[1], 4), "z_out": round(cp[2], 4),
            "vx_out": round(vxo,   4), "vy_out":round(vyo,   4), "vz_out":round(vzo,   4),
            "shot_type": shot_type,
            "_player_shot":  ps_name,
            "_bounce_x":     round(bpos[0], 4) if bpos else None,
            "_bounce_y":     round(bpos[1], 4) if bpos else None,
            "_bounce_vz_in": bvzi,
            "_net_z":        round(nz, 4) if nz is not None else None,
            "_net_clear_m":  round(nz - NET_TOP_Z, 4) if nz is not None else None,
            "_t_net":        t_n,
            "_t_to_bot":     tfl,
            "_omega_x":      round(ox, 2),
            "_omega_y":      round(oy, 2),
            "_omega_z":      round(oz, 2),
            "_profile":      PHYSICS_PROFILE,
            "_bounce_e":     BOUNCE_E,
            "_bounce_mu":    BOUNCE_MU,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, faults, total


TRAIN_COLS = [
    #Player hit state at moment of contact (model input for full pipeline)
    "x", "y", "z", "vx", "vy", "vz",
    #Ball state at bot intercept (arrival features, use as inputs for policy-only model; omit player hit state to avoid leakage)
    "contact_vx", "contact_vy", "contact_vz", "bounced",
    #Bot output: intercept position, return velocity, shot choice
    "x_out", "y_out", "z_out", "vx_out", "vy_out", "vz_out",
    "shot_type",
]


def print_sanity(df):
    BOUNCE_E, BOUNCE_MU, _ = get_profile()
    checks = [
        ("vy > 0  (player hits toward bot, +depth)",   (df.vy > 0).all()),
        ("vy_out < 0  (bot returns to player, -depth)", (df.vy_out < 0).all()),
        ("y < NET_Y  (player side)",                    (df.y < NET_Y).all()),
        ("y_out > NET_Y  (bot side)",                   (df.y_out > NET_Y).all()),
        ("z > 0  (contact above floor)",                (df.z > 0).all()),
        ("z_out in BOT_INTERCEPT_Z",                   ((df.z_out >= BOT_INTERCEPT_Z[0]) &
                                                         (df.z_out <= BOT_INTERCEPT_Z[1])).all()),
        ("net_clear_m >= 0.0",                          (df["_net_clear_m"] >= 0.0).all()),
        ("_t_net not null  (net always crossed)",        df["_t_net"].notna().all()),
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

    pct = df["bounced"].mean() * 100
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
              f"{spd.min():5.1f} - {spd.max():5.1f}  "
              f"{g.vy_out.min():5.1f} - {g.vy_out.max():5.1f}")


if __name__ == "__main__":
    _, _, profile_note = get_profile()
    t0 = time.time()
    print(f"Pickleball Dataset Generator v1.1")
    print(f"Coordinate frame: x=right  y=depth  z=height  (matches MqttController.cs)")
    print(f"Net at AI y={NET_Y}  top AI z={NET_TOP_Z}  intercept first-reachable in bot territory")
    print(f"Profile: {PHYSICS_PROFILE} - {profile_note}")
    print(f"Budget:  {ATTEMPT_BUDGET} attempts")

    df, faults, total = generate_dataset()

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Attempts: {total}  Faults: {faults} ({100*faults/max(total,1):.0f}%)  Kept: {len(df)}")

    print_sanity(df)

    df_train = df[TRAIN_COLS]
    DEBUG_COLS = TRAIN_COLS + [c for c in df.columns if c.startswith("_")]
    df_debug  = df[DEBUG_COLS]

    print(f"\nSample training rows:")
    print(df_train.head(6).to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    out_train = os.path.join(out_dir, "pickleball_shot_dataset.csv")
    out_debug = os.path.join(out_dir, "pickleball_shot_dataset_debug.csv")

    df_train.to_csv(out_train, index=False)
    df_debug.to_csv(out_debug, index=False)

    print(f"\nSaved:")
    print(f"  {out_train}   ({len(df_train):,} rows x {len(df_train.columns)} cols)")
    print(f"  {out_debug}   ({len(df_debug):,} rows x {len(df_debug.columns)} cols)")
