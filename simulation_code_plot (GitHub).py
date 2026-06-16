"""
simulation.py
=============
Numerical simulation for "Experiments with a Falling Rod"
Vitor Oliveira, Am. J. Phys. 84, 113-117 (2016)

Reproduces:
  - Fig. 3 (Steel surface)
  - Fig. 4 (Cloth/mouse-pad surface)
  - Fig. 6 (Marble surface)
  - Fig. 7 (Phase diagram)

Method: Finite-difference (Euler integration), dt = 0.001 s
Equations of motion: Eq.(1)-(6) from the original paper.

Author: [Student Name]
Course: BS103A General Physics I, DGIST, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────
g  = 9.81   # gravitational acceleration [m/s^2]
L  = 0.254  # rod length [m]  (25.4 cm, as used in the paper)
dt = 0.001  # time step [s]

# ── Core equations (from the paper) ───────────────────────────────────

def alpha_noslip(theta):
    """
    Angular acceleration under no-slip condition.
    Eq.(1): alpha = (3g / 2L) * cos(theta)
    """
    return (3 * g) / (2 * L) * np.cos(theta)


def f_friction(theta, theta0):
    """
    Frictional force (no-slip).
    Eq.(2): f = (mg/4) * (9*cos(theta)*sin(theta) - 6*sin(theta0)*cos(theta))
    Mass m cancels when computing mu_s = |f|/N, so we set m=1 here.
    """
    return (g / 4) * (9 * np.cos(theta) * np.sin(theta)
                      - 6 * np.sin(theta0) * np.cos(theta))


def N_normal(theta, theta0):
    """
    Normal force (no-slip).
    Eq.(3): N = (mg/4) * (1 + 9*sin^2(theta) - 6*sin(theta0)*sin(theta))
    m=1 convention (same as above).
    """
    return (g / 4) * (1 + 9 * np.sin(theta)**2
                      - 6 * np.sin(theta0) * np.sin(theta))


def alpha_slip(theta, omega, mu_k):
    """
    Angular acceleration under slipping condition.
    Eq.(5): alpha = (2g/L - omega^2 * sin(theta))
                  / (cos(theta) + (3*cos(theta) - 3*mu_k*sin(theta))^-1)
    mu_k > 0 : bottom end moves in -x (backward)
    mu_k < 0 : bottom end moves in +x (forward)
    """
    numerator   = 2 * g / L - omega**2 * np.sin(theta)
    denominator = np.cos(theta) + 1.0 / (3 * np.cos(theta) - 3 * mu_k * np.sin(theta))
    return numerator / denominator


def a_bottom(theta, omega, alpha, mu_k):
    """
    Horizontal acceleration of the bottom end.
    Eq.(6): a_O = (omega^2 * L/2)*cos(theta)
                - (alpha  * L/2)*sin(theta)
                - b * mu_k
    where b = (3*cos(theta) - 3*mu_k*sin(theta))^-1
    """
    b = 1.0 / (3 * np.cos(theta) - 3 * mu_k * np.sin(theta))
    return (omega**2 * L / 2) * np.cos(theta) \
         - (alpha    * L / 2) * np.sin(theta) \
         - b * mu_k


# ── Main simulation function ───────────────────────────────────────────

def simulate_rod(theta0_deg, mu_s, mu_k):
    """
    Simulate the fall of the rod from rest at angle theta0_deg.

    Parameters
    ----------
    theta0_deg : float  Initial release angle [degrees]
    mu_s       : float  Static friction coefficient
    mu_k       : float  Kinetic friction coefficient (magnitude)

    Returns
    -------
    theta_arr : np.ndarray  Angle at each time step [degrees]
    xO_arr    : np.ndarray  Horizontal position of bottom end [mm]
    """
    theta0   = np.radians(theta0_deg)
    theta    = theta0
    omega    = 0.0      # angular velocity [rad/s]
    x_O      = 0.0      # bottom-end position [m]
    v_O      = 0.0      # bottom-end velocity [m/s]
    slipping = False
    mu_k_eff = mu_k     # effective kinetic friction (sign encodes direction)

    theta_arr = [theta]
    xO_arr    = [0.0]

    for _ in range(300_000):
        if theta <= 0.001:
            break

        if not slipping:
            # ── No-slip phase ──────────────────────────────────────────
            f = f_friction(theta, theta0)
            N = N_normal(theta, theta0)

            if N > 1e-9 and abs(f) > mu_s * N:
                # Slip begins: direction determined by sign of f
                slipping = True
                mu_k_eff = mu_k if f > 0 else -mu_k
            else:
                alp    = alpha_noslip(theta)
                omega += alp * dt
                theta -= omega * dt
                theta_arr.append(theta)
                xO_arr.append(0.0)
                continue

        # ── Slipping phase ─────────────────────────────────────────────
        # Update mu_k sign if velocity direction reverses
        if   v_O < -1e-6:
            mu_k_eff =  mu_k   # moving backward → positive mu_k
        elif v_O >  1e-6:
            mu_k_eff = -mu_k   # moving forward  → negative mu_k

        alp    = alpha_slip(theta, omega, mu_k_eff)
        a_O    = a_bottom(theta, omega, alp, mu_k_eff)

        omega += alp * dt
        theta -= omega * dt
        v_O   += a_O  * dt
        x_O   += v_O  * dt

        theta_arr.append(theta)
        xO_arr.append(x_O * 1000)   # convert m → mm

    return np.degrees(np.array(theta_arr)), np.array(xO_arr)


# ── Plot helper ────────────────────────────────────────────────────────

def plot_surface(ax, configs, title, ylim=None):
    """
    Plot x_O vs theta for a given surface (list of configs).
    configs: list of (theta0_deg, mu_s, mu_k, marker, label)
    """
    markers = ['o', 'o', '.']
    for i, (th0, ms, mk, label) in enumerate(configs):
        ths, xs = simulate_rod(th0, ms, mk)
        ax.plot(ths, xs, markers[i], markersize=3, label=label)
    ax.set_xlim(80, 0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel('θ (deg)', fontsize=11)
    ax.set_ylabel('x (mm)', fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')


# ── Figure 1: Displacement plots (Figs. 3, 4, 6 of paper) ─────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(
    'Simulated Horizontal Displacement of Bottom End vs. Falling Angle\n'
    '(Replicating Figs. 3, 4, 6 — Oliveira 2016)',
    fontsize=12, fontweight='bold'
)

plot_surface(axes[0],
    [(75.8, 0.26, 0.185, '75.8°'),
     (26.0, 0.26, 0.185, '26.0°'),
     (5.6,  0.26, 0.185, '5.6°')],
    'Steel Surface  (μs = 0.26, μk = 0.185)',
    ylim=(-26, 2)
)
plot_surface(axes[1],
    [(74.2, 1.9, 0.8, '74.2°'),
     (26.6, 1.9, 0.8, '26.6°'),
     (9.8,  1.9, 0.8, '9.8°')],
    'Cloth Surface / Mouse Pad  (μs = 1.9, μk = 0.8)',
    ylim=(-1, 13)
)
plot_surface(axes[2],
    [(77.0, 0.9, 0.33, '77.0°'),
     (27.1, 0.9, 0.33, '27.1°'),
     (6.4,  0.9, 0.33, '6.4°')],
    'Marble Stone Surface  (μs = 0.9, μk = 0.33)'
)

plt.tight_layout()
plt.savefig('simulation_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: simulation_plots.png")


# ── Figure 2: Phase diagram (Fig. 7 of paper) ─────────────────────────

fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))

theta0_range = np.linspace(1, 89, 600)

# Critical-angle boundary (solid): theta_s = theta_c → f(theta_c)=0
# mu_s = |f(theta_c)| / N(theta_c)  from Eq.(4)
mu_critical = []
for th0_deg in theta0_range:
    th0 = np.radians(th0_deg)
    thc = np.arcsin(min((2/3) * np.sin(th0), 1.0))
    num = abs(9*np.cos(thc)*np.sin(thc) - 6*np.sin(th0)*np.cos(thc))
    den =     1 + 9*np.sin(thc)**2      - 6*np.sin(th0)*np.sin(thc)
    mu_critical.append(num / den if den > 1e-9 else np.nan)

# No-slip boundary (dashed): slip never begins → mu_s evaluated at theta=theta0
mu_noslip = []
for th0_deg in theta0_range:
    th0 = np.radians(th0_deg)
    num = abs(9*np.cos(th0)*np.sin(th0) - 6*np.sin(th0)*np.cos(th0))
    den =     1 + 9*np.sin(th0)**2      - 6*np.sin(th0)*np.sin(th0)
    mu_noslip.append(num / den if den > 1e-9 else np.nan)

ax2.plot(theta0_range, mu_critical, 'k-',  lw=1.8, label='Backward/Forward boundary')
ax2.plot(theta0_range, mu_noslip,   'k--', lw=1.8, label='No-slip boundary')

# Experimental data points
exp = [
    # (theta0, mu_s, marker, label)
    (74.2, 1.9,  'o', 'Cloth (μs=1.9)'),
    (26.6, 1.9,  'o', None),
    (9.8,  1.9,  'o', None),
    (77.0, 0.9,  's', 'Marble (μs=0.9)'),
    (27.1, 0.9,  's', None),
    (6.4,  0.9,  's', None),
    (75.8, 0.26, '^', 'Steel (μs=0.26)'),
    (26.0, 0.26, '^', None),
    (5.6,  0.26, '^', None),
]
for th0, mu, mk, lbl in exp:
    ax2.plot(th0, mu, mk, color='k', ms=8,
             label=lbl if lbl else '_nolegend_')

# Region labels
ax2.text(45, 2.6,  'No slip',               fontsize=10, ha='center', style='italic')
ax2.text(45, 1.4,  'Forward slip only',      fontsize=10, ha='center', style='italic')
ax2.text(45, 0.25, 'Backward → Forward slip',fontsize=10, ha='center', style='italic')

ax2.set_xlim(0, 90)
ax2.set_ylim(0, 3.2)
ax2.set_xlabel('Initial release angle  θ₀ (deg)', fontsize=12)
ax2.set_ylabel('Static friction coefficient  μs',  fontsize=12)
ax2.set_title('Phase Diagram: Initial Slipping Direction\n'
              '(Replicating Fig. 7 — Oliveira 2016)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: phase_diagram.png")
print("\nAll figures generated successfully.")
