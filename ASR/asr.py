import numpy as np
import logging

def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1):
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power
    artifacts. Specifically, only windows are retained which have less than a
    certain fraction of "bad" channels, where a channel is bad in a window if
    its power is above or below a given upper/lower threshold (in standard
    deviations from a robust estimate of the EEG power distribution in the
    channel).

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        Continuous data set, assumed to be appropriately high-passed (e.g. >
        1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    zthresholds : 2-tuple
        The minimum and maximum standard deviations within which the power of
        a channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).

    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.

    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but not shorter than half a cycle of the high-pass filter that was
        used. Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).

    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).

    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.int32(np.round(np.arange(0, ns - N, (N * (1 - win_overlap)))))
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):

        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove
    if np.max(zthresholds) > 0:
        mask1 = swz[-(np.int32(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + np.int32(max_bad_chans - 1), :] < np.min(zthresholds))

    # combine the two thresholds
    remove_mask = np.logical_or.reduce((mask1, mask2))
    removed_wins = np.where(remove_mask)

    # reconstruct the samples to remove
    sample_maskidx = []
    for i in range(len(removed_wins[0])):
        if i == 0:
            sample_maskidx = np.arange(
                offsets[removed_wins[0][i]], offsets[removed_wins[0][i]] + N)
        else:
            sample_maskidx = np.vstack((
                sample_maskidx,
                np.arange(offsets[removed_wins[0][i]],
                          offsets[removed_wins[0][i]] + N)
            ))

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, 1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
        return clean, sample_mask
    else:
        sample_mask = np.ones((1, ns), dtype=bool)
        return X, sample_mask
