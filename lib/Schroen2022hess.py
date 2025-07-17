import numpy as np
import pandas
import matplotlib.pyplot as plt

# from .Schroen2017hess import get_footprint, Wr, Wr_approx
from neptoon.corrections.theory.calibration_functions import (
    Schroen2017,
)
from neptoon.corrections.theory.neutrons_to_soil_moisture import (
    gravimetric_soil_moisture_to_neutrons_koehli_etal_2021,
    neutrons_to_total_grav_soil_moisture_koehli_etal_2021,
)

# from .Koehli2021fiw import sm2N_Koehli  # , N2SM_Schmidt_single

N0_scaling = 0.77  # scaling factor to compare Desilets with Koehli/Schmidt function


def Field_at_Distance(
    R=0,
    theta1=0.05,
    theta2=0.10,
    hum=5,
    bd=1.43,
    N0=1000,
    air_pressure=1013.25,
    max_radius=500,
    resolution=0.01,
    verbose=True,
):
    """
    Function to calculate N_eff as a whole
    """
    N1 = gravimetric_soil_moisture_to_neutrons_koehli_etal_2021(
        theta1,
        hum,
        N0 / N0_scaling,
        koehli_method_form="Mar21_uranos_drf",
    )
    N2 = gravimetric_soil_moisture_to_neutrons_koehli_etal_2021(
        theta2,
        hum,
        N0 / N0_scaling,
        koehli_method_form="Mar21_uranos_drf",
    )
    # N1 = (
    #     sm2N_Koehli(
    #         theta1,
    #         h=hum,
    #         off=0.0,
    #         bd=bd,
    #         func="vers2",
    #         method="Mar21_uranos_drf",
    #         bio=0,
    #     )
    #     * N0
    #     / N0_scaling
    # )
    # N2 = (
    #     sm2N_Koehli(
    #         theta2,
    #         h=hum,
    #         off=0.0,
    #         bd=bd,
    #         func="vers2",
    #         method="Mar21_uranos_drf",
    #         bio=0,
    #     )
    #     * N0
    #     / N0_scaling
    # )

    r = pandas.Series(np.arange(resolution, max_radius, resolution), name="r")
    W = pandas.DataFrame(index=r)
    W["r_rescaled"] = Schroen2017.rescale_distance(
        distance_from_sensor=W.index,
        atmospheric_pressure=air_pressure,
    )
    R_rescaled = Schroen2017.rescale_distance(
        distance_from_sensor=R, atmospheric_pressure=air_pressure
    )
    # W["w"] = Wr(W.index, sm=theta1, hum=hum)
    W["w"] = Schroen2017.horizontal_weighting(W.index, theta1, hum)

    W["wn"] = W.w / W.w.sum()
    W["a"] = np.nan
    W.loc[R:, "a"] = 1 / np.pi * np.arccos(R / W.loc[R:].index)
    # W.loc[R_rescaled:, "a"] = (
    #     1 / np.pi * np.arccos(R_rescaled / W.loc[R_rescaled:].r_rescaled)
    # )
    W["wa"] = W.wn * W.a
    w = W.wa.sum()
    N_eff = (1 - w) * N1 + w * N2
    # sm_eff = N2SM_Schmidt_single(N_eff / N0 * N0_scaling, hum=hum, bd=bd)
    # print("neptoon", N_eff, hum, bd, lw, owe, method)
    sm_eff = (
        neutrons_to_total_grav_soil_moisture_koehli_etal_2021(
            N_eff, N0 / N0_scaling, hum
        )
        # * bd
    )
    # print(N_eff, N0, N0_scaling, sm_eff, sm_eff2)

    if verbose:
        print(
            "R={0:.1f}, theta1={1:.1%}, theta2={2:.1%}, theta_eff={3:.1%}, N1={4:.0f}, N2={5:.0f}, N_eff={6:.0f}, Influence={7:.1%}, Contribution={8:.1%}".format(
                R,
                theta1,
                theta2,
                sm_eff,
                N1,
                N2,
                N_eff,
                w,
                w * N2 / ((1 - w) * N1 + w * N2),
            )
        )
    return (
        N1,
        N2,
        N_eff,
        sm_eff,
        (1 - w) * N1 / ((1 - w) * N1 + w * N2),
        w * N2 / ((1 - w) * N1 + w * N2),
    )


def Plot_Field(
    ax=None,
    Var=None,
    R=0,
    footprint_radius=200,
    annotate=None,
    overlay=None,
    fontsize=10,
    title="Signal contribution",
    extent=500,
    cmap="Greys",
    vmax_factor=4,
    x_marker=None,
):
    """
    Plot of a matrix field
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title, loc="left")

    R2 = int(R / 2)
    extent2 = int(extent / 2)
    extent4 = int(extent / 4)

    ax.imshow(
        Var,
        interpolation="none",
        cmap=cmap,
        vmax=np.max(Var) * vmax_factor,
        origin="lower",
        extent=(-extent2, extent2, -extent2, extent2),
    )

    if annotate:
        ax.annotate(
            "{0:.1%}".format(annotate[0]),
            (-R2 - extent4, 0),
            fontsize=fontsize,
            ha="center",
            va="center",
        )
        ax.annotate(
            "{0:.1%}".format(annotate[1]),
            (R2 + extent4, 0),
            fontsize=fontsize,
            ha="center",
            va="center",
        )

    grid100 = np.arange(-extent / 2 * 0.8, extent / 2 * 0.8 + 1, extent * 0.2)
    xticks = np.arange(0, extent + 1, extent * 0.2)
    ax.set_xticks(grid100, ["%.0f" % x for x in grid100])
    ax.set_yticks(grid100, ["%.0f" % x for x in grid100[::-1]])
    # Zoom to extend
    ax.set_xlim(-extent2, extent2)
    ax.set_ylim(-extent2, extent2)
    # Tick markers inside
    ax.tick_params(axis="y", direction="in", length=4)
    ax.tick_params(axis="x", direction="in", length=4)
    # Ticks an all sides
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    # Central lines
    ax.plot(0, 0, color="black", marker="+", ms=10, mew=1)

    # Distance marker
    if x_marker:
        ax.annotate(
            "$R=%.0f\,$m" % R,
            xy=(R, extent2),
            xytext=(R, extent2 * 1.14),
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "wedge"},
        )
        # for arrows, see https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancyarrow_demo.html

    # Footprint circle
    fpc = plt.Circle(
        (0, 0), footprint_radius, fc="none", ec="black", ls="--", alpha=0.3
    )
    ax.add_patch(fpc)

    return ax


def Generate_Splitfields(size, R):
    size = size - 1
    size2 = int(size / 2)
    split_field = np.array([[0] * (size2 + R) + [1] * (size2 - R)], np.int8).repeat(
        size, 0
    )
    return split_field


def Plot_Ntheta(
    ax, R, theta1, theta2, thetaeff, N1, N2, Neff, hum=5, off=0, bd=1.43, N0=1500
):

    th = np.arange(0.01, 0.9, 0.01)
    N = [
        gravimetric_soil_moisture_to_neutrons_koehli_etal_2021(
            x,
            hum,
            N0 / N0_scaling,
            koehli_method_form="Mar21_uranos_drf",
        )
        for x in th
    ]
    # sm2N_Koehli(
    #     x, h=hum, off=off, bd=bd, func="vers2", method="Mar21_uranos_drf", bio=0
    # )
    # * N0
    # / N0_scaling

    ax.plot(th, N, color="silver", lw=3, zorder=0)
    ax.scatter(
        theta1,
        N1,
        color="C1",
        marker=">",
        s=100,
        lw=2,
        fc="none",
        label=r"$\hat\theta_1$ = {0:4.1f}%,  $N_1$ = {1:4.0f} cph".format(
            theta1 * 100, N1
        ),
    )
    ax.scatter(
        theta2,
        N2,
        color="C0",
        marker="<",
        s=100,
        lw=2,
        fc="none",
        label=r"$\hat\theta_2$ = {0:4.1f}%,  $N_2$ = {1:4.0f} cph".format(
            theta2 * 100, N2
        ),
    )
    ax.scatter(
        thetaeff,
        Neff,
        color="k",
        marker="o",
        s=100,
        lw=2,
        fc="none",
        label=r"$\hat\theta$   = {0:4.1f}%,  $\hat N$  = {1:4.0f} cph".format(
            thetaeff * 100, Neff
        ),
    )

    ax.plot([0, theta1, theta1], [N1, N1, 0], color="C1", lw=1)
    ax.plot([0, theta2, theta2], [N2, N2, 0], color="C0", lw=1)
    ax.plot([0, thetaeff, thetaeff], [Neff, Neff, 0], color="k", lw=1)
    ax.legend(
        ncol=1,
        fancybox=False,
        framealpha=1,
        edgecolor="none",
        handlelength=1,
        borderaxespad=1,
        loc="upper right",
    )
    xticks = np.arange(0, 0.9, 0.1)
    ax.set_xticks(xticks, ["{0:.0%}".format(x) for x in xticks])
    ax.set_xlim(0, 0.80)
    ax.set_ylim(np.min(N), np.max(N))
    ax.set_xlabel("Water content (%)")
    ax.set_ylabel("Neutron counts (cph)")
    ax.set_title("Effective signal", loc="left")


def Plot_Sensibility(ax, N1, Neff):
    hourly = 1 / np.sqrt(N1)
    daily = hourly / np.sqrt(24)
    Nrel = 1 - N1 / Neff
    if np.abs(Nrel) > hourly:
        bc = "C2"
    elif np.abs(Nrel) > daily:
        bc = "C0"
    else:
        bc = "C3"
    bc = "silver"

    ax.bar(0, np.abs(Nrel), color=bc, log=True)  # 'silver'
    ax.bar(0, hourly, width=1, fc="C1", ec="none", alpha=0.1)  # , hatch='//'
    ax.bar(0, daily, width=1, fc="C1", ec="none", alpha=0.1)  # , hatch='\\\\'
    ax.axhline(hourly, ls="--", lw=1, color="k")
    ax.axhline(daily, ls="--", lw=1, color="k")
    # ax.axhline(hourly/np.sqrt(12), ls='--', lw=1, color='k')

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylim(0.001, 1)
    ax.set_yticks([0.001, 0.01, 0.10], ["0.1%", "1%", "10%"])
    ax.set_title("Sensibility")
    ax.set_ylabel("Change\n$1-N_1\,/\hat N$", ha="right")
    ax.yaxis.set_label_coords(-0.03, 1)
    ax.grid(which="both", color="k", alpha=0.1)
    ax.annotate("hourly", (0.45, hourly * 1.1), ha="right")
    ax.annotate("daily", (0.45, daily * 1.1), ha="right")
    ax.annotate(
        "${0:+.2}$%".format(100 * Nrel),
        (0, np.abs(Nrel) * 1.2),
        ha="center",
        fontsize=12,
    )

    ax2 = ax.twinx()
    ax2.set_ylim(0.001, 1)
    ax2.set_yscale("log")
    ax2.set_yticks([hourly, daily], ["{0:.1%}".format(hourly), "{0:.1%}".format(daily)])
    ax2.set_ylabel("Precision\nlimit, $\sigma_N$", ha="right")
    ax2.yaxis.set_label_coords(1.08, 1)


def optimize_R(
    R,
    theta1,
    theta2,
    air_pressure=1013.25,
    dN=0.01,
    N0=1500,
    bd=1.43,
    verbose=False,
    max_radius=500,
):

    N1, N2, Neff, thetaeff, c1, c2 = Field_at_Distance(
        R,
        theta1=theta1,
        theta2=theta2,
        air_pressure=air_pressure,
        N0=N0,
        bd=bd,
        verbose=verbose,
        max_radius=max_radius,
    )
    return 1 - Neff / N1 - dN
