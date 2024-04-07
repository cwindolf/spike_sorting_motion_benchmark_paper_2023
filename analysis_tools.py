import numpy as np
import h5py
import pickle
from dredge import motion_util as mu


def dispmap_from_me(motion_est, geom=None, ys=None, times=None):
    if times is None:
        times = motion_est.time_bin_centers_s
    if ys is None:
        ys = np.unique(geom[:, -1])
    displacement = motion_est.disp_at_s(times, ys, grid=True)
    return mu.get_motion_estimate(
        time_bin_centers_s=times,
        spatial_bin_centers_um=ys,
        displacement=displacement,
    )


def spikes_from_mearec(measim_h5, gtme):
    spike_trains = []
    amplitudes = []
    depths = []

    with h5py.File(measim_h5) as h5:
        template_locations = h5["template_locations"][:]
        cell_y = template_locations[:, template_locations.shape[1] // 2, -1]
        cell_a = h5["voltage_peaks"][:].max(1)

        for j in range(len(cell_y)):
            st = h5[f"spiketrains/{j}/times"][:]
            spike_trains.append(st)
            amplitudes.append(np.full(st.shape, cell_a[j]))
            depths.append(cell_y[j] + gtme.disp_at_s(st, cell_y[j], grid=True).squeeze())

    a = np.concatenate(amplitudes)
    y = np.concatenate(depths)
    t = np.concatenate(spike_trains)
    return a, y, t


def spikes_from_ks(rez2_mat, fs=32000):
    with h5py.File(rez2_mat) as h5:
        st0 = h5["rez/st0"][:]
    t = st0[0] / fs
    y = st0[1]
    a = st0[2]
    return a, y, t


def spikes_from_dredge(subtraction_h5):
    with h5py.File(subtraction_h5) as h5:
        zlim = h5["geom"][:].min(), h5["geom"][:].max()
        y = h5["point_source_localizations"][:, 2]
        valid = np.clip(y, *zlim) == y
        y = y[valid]
        if "denoised_ptp_amplitudes" in h5:
            a = h5["denoised_ptp_amplitudes"][:][valid]
        else:
            a = h5["ptp_amplitudes"][:][valid]
        t = h5["times_seconds"][:][valid]
    return a, y, t


def dispmap_from_mearec(measim_h5):
    """Have to reconstruct this from cell motions..."""
    with h5py.File(measim_h5) as h5:
        template_locations = h5["template_locations"][:]
        geom = h5["channel_positions"][:]
        timestamps = h5["timestamps"][:]

        drifts = 0
        for dli in h5["drift_list"]:
            drift_fs = h5[f"drift_list/{dli}/drift_fs"][()]
            drift_times = np.arange(timestamps.min(), timestamps.max(), 1 / drift_fs)
            drift_centers = 0.5 / drift_fs + drift_times
            drift_factors = h5[f"drift_list/{dli}/drift_factors"][:]
            drift_vector = h5[f"drift_list/{dli}/drift_vector_um"][:]
            drifts = drifts + drift_factors[:, None] * drift_vector[None, :]

    cell_y = template_locations[:, template_locations.shape[1] // 2, -1]

    cell_order = np.argsort(cell_y)
    cell_y = cell_y[cell_order]
    drifts = drifts[cell_order]

    motion_est = mu.get_motion_estimate(
        time_bin_centers_s=drift_centers,
        spatial_bin_centers_um=cell_y,
        displacement=drifts,
    )

    return motion_est, dispmap_from_me(motion_est, geom)


def motion_est_from_rez(rez2_mat, fs=32000):
    with h5py.File(rez2_mat) as h5:
        dshift = h5["rez/dshift"][:]
        NT = h5["rez/ops/NT"][()][0]
        ycoords = h5["rez/ycoords"][:]
    delta_bin = NT / fs
    time_bin_starts_s = np.arange(dshift.shape[1]) * delta_bin
    time_bin_centers_s = time_bin_starts_s + delta_bin / 2
    nblocks = (dshift.shape[0] + 1) // 2
    yl = np.floor(ycoords.ptp() / nblocks).astype("int") - 1
    mins = np.linspace(ycoords.min(), ycoords.max() - yl - 1, 2 * nblocks - 1)
    maxs = mins + yl
    centers = (mins + maxs) / 2
    return mu.get_motion_estimate(
        -dshift, time_bin_centers_s=time_bin_centers_s, spatial_bin_centers_um=centers
    )


def dispmap_from_ks(gt_motion, rez2_mat, fs=32000):
    ksme = motion_est_from_rez(rez2_mat, fs=fs)
    return ksme, dispmap_from_me(ksme, ys=gt_motion.spatial_bin_centers_um, times=gt_motion.time_bin_centers_s)


def dispmap_from_dredge(gt_motion, motion_est_pkl):
    with open(motion_est_pkl, "rb") as jar:
        dredgeme = pickle.load(jar)
    return dredgeme, dispmap_from_me(dredgeme, ys=gt_motion.spatial_bin_centers_um, times=gt_motion.time_bin_centers_s)


def squares(ax, x, y, s, **kwargs):
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    patches = [Rectangle((x_, y_), s_, s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection