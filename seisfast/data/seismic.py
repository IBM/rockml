
"""
Module: RockML
author: daniela.szw@ibm.com, rosife@br.ibm.com, sallesd@br.ibm.com
copyright: IBM Confidential
copyright: OCO Source Materials
copyright: © IBM Corp. All Rights Reserved
date: 2020
IBM Certificate of Originality
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.stats import norm

import seisfast.io


def _average_trace(t):
    avgt = np.mean(t, axis=1)
    return avgt


def _trace_energy(t):
    et = np.square(t).sum()
    return et


def _total_energy(t):
    total_energy = 0

    for k in range(t.shape[1]):
        e = _trace_energy(t[:, k])
        total_energy += e

    return total_energy


def _trace_semblance(t):
    m = t.shape[1]

    ta = _average_trace(t)
    ea = _trace_energy(ta)
    et = _total_energy(t)

    result = (m * ea) / et

    return result


def _nmo(trace, x, velocities, initial_time=0.0, sample_rate=0.002):
    """
    Applies normal move out correction.

    Parameters
    ----------

    trace: array
        seismic trace
    x: float
        offset in meters
    velocities: array
        velocities in meters/second
    initial_time: float
        initial time
    sample_rate: float
        sample rate in seconds
    """

    output = np.zeros(trace.shape)

    times = np.arange(initial_time, (len(trace) - 1) * sample_rate + initial_time + 1,
                      sample_rate)

    for k in range(len(trace)):

        # times[k] is t0, the place where the value should be
        # t is the time the value is now
        # we want t -> t0

        t = int(np.sqrt(times[k] ** 2 + (x ** 2) / (velocities[k] ** 2)) / sample_rate)

        if (t < 0) or (t > (len(trace) - 1)):
            continue

        output[k] = trace[t]

    return output


def _clip_image(image, clip_perc=2):
    p10, p90 = np.percentile(image, (clip_perc, 100 - clip_perc))

    image[image < p10] = p10
    image[image > p90] = p90
    image -= p10
    image /= (p90 - p10)

    image = image * 255

    return image


def _check_equalization_parameters(samples, mean, stddev):
    tc = norm.cdf(samples.ravel(), mean, stddev)

    tp = norm.ppf(tc, mean, stddev)

    if np.isnan(tp).any():
        return False

    if not np.isfinite(tp).all():
        return False

    return True


def _get_increment(value):
    if value >= 1000:
        return 100
    elif value >= 100:
        return 10
    elif value >= 1:
        return 0.1
    elif value >= 0.1:
        return 0.01


def _write_segy_slice(input_segy, output_segy, sample_interval, slice_type, label_image, index,
                      data_type, c,
                      fil="74 3d poststack inline number",
                      fxl="75 3d poststack crossline number",
                      fx="72 x coordinate of ensemble position (cdp)",
                      fy="73 y coordinate of ensemble position (cdp)"):
    if slice_type == "inline":

        xl_res = input_segy.get_crossline_resolution()
        range_xlines = input_segy.get_range_crosslines()
        range_time_depth = input_segy.get_range_time_depth()

        trace_map = input_segy.get_trace_map()
        traces = trace_map[int(
            (index - input_segy.get_range_inlines()[0]) / input_segy.get_inline_resolution()),
                 :].tolist()

        for t in traces:

            if t < 0:
                continue

            trace = input_segy.trace_header(t)

            if trace is None:
                c += 1
                continue

            idx = trace[fxl]
            x = trace[fx]
            y = trace[fy]

            output_segy.write_trace_header(None, x, y, sample_interval, index, idx,
                                           int((idx - int(range_xlines[0])) / max(xl_res,
                                                                                  1)) + 1,
                                           c + 1, range_time_depth[0])

            if data_type == 5:
                output_segy.write_trace_data(
                    list(label_image[:, int((idx - int(range_xlines[0])) / max(xl_res, 1))]))
            elif data_type == 3:
                output_segy.write_trace_data([int(i) for i in list(
                    label_image[:, int((idx - int(range_xlines[0])) / max(xl_res, 1))])])

            c += 1

    else:

        il_res = input_segy.get_inline_resolution()
        range_ilines = input_segy.get_range_inlines()
        range_time_depth = input_segy.get_range_time_depth()

        trace_map = input_segy.get_trace_map()
        traces = trace_map[:, int((index - input_segy.get_range_crosslines()[
            0]) / input_segy.get_crossline_resolution())].tolist()

        for t in traces:

            if t < 0:
                continue

            trace = input_segy.trace_header(t)

            if trace is None:
                c += 1
                continue

            idx = trace[fil]
            x = trace[fx]
            y = trace[fy]

            output_segy.write_trace_header(None, x, y, sample_interval, idx, index,
                                           int((idx - int(range_ilines[0])) / max(il_res,
                                                                                  1)) + 1,
                                           c + 1, range_time_depth[0])

            if data_type == 5:
                output_segy.write_trace_data(
                    list(label_image[:, int((idx - int(range_ilines[0])) / max(il_res, 1))]))
            elif data_type == 3:
                output_segy.write_trace_data([int(i) for i in list(
                    label_image[:, int((idx - int(range_ilines[0])) / max(il_res, 1))])])

            c += 1

    return c


def _get_index_from_name(name):
    terms = name.split("_")
    for t in terms:
        aux = t.replace(".png", "")
        if aux.isdigit():
            return int(aux)

    assert False, "could not parse index number"


# post stack

def export(segy, out_path="", format='npy', scope_ilines=[], scope_xlines=[]):
    """
    Exports seismic slices as files.

    Parameters
    ----------

    segy: obj
        segy object
    out_path: str
        path where the output will be written
    format: str
        npy, png or tiff
    scope_ilines: list
        range of inlines to export
    scope_xlines: list
        range of xlines to export

    """

    if len(scope_ilines) == 0:
        scope_ilines = segy.get_range_inlines()

    if len(scope_xlines) == 0:
        scope_xlines = segy.get_range_crosslines()

    rangeil = segy.get_range_inlines()
    rangexl = segy.get_range_crosslines()

    resil = segy.get_inline_resolution()
    resxl = segy.get_crossline_resolution()

    print("\nexporting inlines...")

    for il in range(scope_ilines[0], scope_ilines[1] + 1, resil):
        image = segy.get_inline(int((il - rangeil[0]) / resil))

        if format == "npy":
            np.save(os.path.join(out_path,
                                 os.path.basename(segy.get_filename()) + ".inline_" + str(il)),
                    image)
        elif format == "tiff":
            # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".inline_" + str(il) + ".tiff"), image)
            Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                segy.get_filename()) + ".inline_" + str(il) + ".tiff"))
        elif format == "png":
            image = _clip_image(image)
            image = image.astype(np.uint8)
            # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".inline_" + str(il) + ".png"), image)
            Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                segy.get_filename()) + ".inline_" + str(il) + ".png"))

    print("\nexporting crosslines...")

    for xl in range(scope_xlines[0], scope_xlines[1] + 1, resxl):
        image = segy.get_crossline(int((xl - rangexl[0]) / resxl))

        if format == "npy":
            np.save(os.path.join(out_path,
                                 os.path.basename(segy.get_filename()) + ".crossline_" + str(
                                     xl)), image)
        elif format == "tiff":
            # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".crossline_" + str(xl) + ".tiff"), image)
            Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                segy.get_filename()) + ".crossline_" + str(xl) + ".tiff"))
        elif format == "png":
            image = _clip_image(image)
            image = image.astype(np.uint8)
            # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".crossline_" + str(xl) + ".png"), image)
            Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                segy.get_filename()) + ".crossline_" + str(xl) + ".png"))


def export_horizons(segy, horizons, out_path="", format='npy', scope_ilines=[], scope_xlines=[],
                    type='label'):
    """
    Exports seismic slices with horizons.

    Parameters
    ----------

    segy: obj
        segy object
    out_path: str
        path where the output will be written
    format: str
        npy or png
    scope_ilines: list
        range of inlines to export
    scope_xlines: list
        range of xlines to export
    type: str
        line or label

    """

    # TODO: use a color map
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
              [255, 255, 0], [255, 0, 255], [0, 255, 255],
              [255, 128, 0], [128, 255, 0],
              [255, 0, 128], [128, 0, 255],
              [0, 255, 128], [0, 128, 255]]

    if type == "line" and format != "png":
        print("\nline type should be used with png format")
        return

    if len(scope_ilines) == 0:
        scope_ilines = segy.get_range_inlines()

    if len(scope_xlines) == 0:
        scope_xlines = segy.get_range_crosslines()

    rangeil = segy.get_range_inlines()
    rangexl = segy.get_range_crosslines()

    resil = segy.get_inline_resolution()
    resxl = segy.get_crossline_resolution()

    print("\nexporting inlines...")

    for il in range(scope_ilines[0], scope_ilines[1] + 1, resil):

        if type == "line":
            image = segy.get_inline(int((il - rangeil[0]) / resil))
            image = _clip_image(image).astype(np.uint8)
            image = np.dstack((image, image, image))
        else:
            mask = np.zeros((segy.get_num_samples(), segy.get_num_crosslines()), dtype=np.uint8)

        hor_label = 0

        for horizon in horizons:

            hor_label += 1

            point_map = horizon.get_point_map(save=True)

            il_hor_idx = int(il - horizon.get_range_inlines()[0])

            if (il_hor_idx < 0) or (il_hor_idx > (point_map.shape[0] - 1)):
                continue

            line = point_map[il_hor_idx, :]

            for l in range(line.shape[0]):

                if line[l] < 0:
                    continue

                # translates from horizon coordinates to seismic coordinates
                # this is necessary since the horizon may not defined for the whole seismic
                xl_hor_idx = int(l + horizon.get_range_crosslines()[0])
                xl_seis_idx = int((xl_hor_idx - segy.get_range_crosslines()[0]) / segy.get_crossline_resolution())

                h = int((line[l] - segy.get_range_time_depth()[
                    0]) / segy.get_time_depth_resolution())

                if type == "label":
                    mask[h:, xl_seis_idx] = hor_label
                else:
                    image[h - 1:h + 2, xl_seis_idx, 0] = colors[hor_label - 1][0]
                    image[h - 1:h + 2, xl_seis_idx, 1] = colors[hor_label - 1][1]
                    image[h - 1:h + 2, xl_seis_idx, 2] = colors[hor_label - 1][2]

        if format == "png":
            if type == "label":
                aux = mask / len(horizons) * 255
                aux = aux.astype(np.uint8)
                # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".inline_label_" + str(il) + ".png"), aux.astype(np.uint8))
                Image.fromarray(aux).save(os.path.join(out_path, os.path.basename(
                    segy.get_filename()) + ".inline_label_" + str(il) + ".png"))
            else:
                # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".inline_line_" + str(il) + ".png"), image)
                Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                    segy.get_filename()) + ".inline_line_" + str(il) + ".png"))
        elif format == "npy":
            np.save(os.path.join(out_path,
                                 os.path.basename(segy.get_filename()) + ".inline_label_" + str(
                                     il)), mask)

    print("\nexporting crosslines...")

    for xl in range(scope_xlines[0], scope_xlines[1] + 1, resxl):

        if type == "line":
            image = segy.get_crossline(int((xl - rangexl[0]) / resxl))
            image = _clip_image(image).astype(np.uint8)
            image = np.dstack((image, image, image))
        else:
            mask = np.zeros((segy.get_num_samples(), segy.get_num_inlines()), dtype=np.uint8)

        hor_label = 0

        for horizon in horizons:

            hor_label += 1

            point_map = horizon.get_point_map(save=True)

            xl_hor_idx = int(xl - horizon.get_range_crosslines()[0])

            if (xl_hor_idx < 0) or (xl_hor_idx > (point_map.shape[1] - 1)):
                continue

            line = point_map[:, xl_hor_idx]

            for l in range(line.shape[0]):

                if line[l] < 0:
                    continue

                # translates from horizon coordinates to seismic coordinates
                # this is necessary since the horizon may not defined for the whole seismic
                il_hor_idx = int(l + horizon.get_range_inlines()[0])
                il_seis_idx = int((il_hor_idx - segy.get_range_inlines()[0]) / segy.get_inline_resolution())

                h = int((line[l] - segy.get_range_time_depth()[
                    0]) / segy.get_time_depth_resolution())

                if type == "label":
                    mask[h:, il_seis_idx] = hor_label
                else:
                    image[h - 1:h + 2, il_seis_idx, 0] = colors[hor_label - 1][0]
                    image[h - 1:h + 2, il_seis_idx, 1] = colors[hor_label - 1][1]
                    image[h - 1:h + 2, il_seis_idx, 2] = colors[hor_label - 1][2]

        if format == "png":
            if type == "label":
                aux = mask / len(horizons) * 255
                aux = aux.astype(np.uint8)
                # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".crossline_label_" + str(xl) + ".png"), aux.astype(np.uint8))
                Image.fromarray(aux).save(os.path.join(out_path, os.path.basename(
                    segy.get_filename()) + ".crossline_label_" + str(xl) + ".png"))
            else:
                # skimage.io.imsave(os.path.join(out_path, os.path.basename(segy.get_filename()) + ".crossline_line_" + str(xl) + ".png"), image)
                Image.fromarray(image).save(os.path.join(out_path, os.path.basename(
                    segy.get_filename()) + ".crossline_line_" + str(xl) + ".png"))
        elif format == "npy":
            np.save(os.path.join(out_path, os.path.basename(
                segy.get_filename()) + ".crossline_label_" + str(xl)), mask)


def get_writable_post_stack_segy(input_segy, output_path, data_type=3, num_traces=None):
    """
    Returns a writable PostStackSEGY object which is a shallow copy of input segy.

    Parameters
    ----------

    input_segy: obj
        input segy object
    output_path: str
        output segy file path
    data_type: int
        seisfast type
    num_traces: int
        num traces
    """

    assert not os.path.isfile(output_path), "output file already exists"

    if num_traces is None:
        traces = input_segy.get_num_traces()
    else:
        traces = num_traces

    samples = input_segy.get_num_samples()

    output_segy = seisfast.io.PostStackSEGY(output_path, traces, samples, data_type)

    output_segy.write_textual_header(input_segy.textual_header())

    bh = input_segy.binary_header()
    bh["10 seisfast sample format code"] = data_type
    output_segy.write_binary_header(bh)

    return output_segy


def export_labels_as_segy(input_segy, input_path, slice_type, output_segy,
                          srange=None, step=1,
                          fil="74 3d poststack inline number",
                          fxl="75 3d poststack crossline number",
                          fx="72 x coordinate of ensemble position (cdp)",
                          fy="73 y coordinate of ensemble position (cdp)"):
    """
    Exports label images as segy.

    Parameters
    ----------

    input_segy: obj
        segy object
    input_path: str
        path with label images
    slice_type: str
        inline or crossline
    output_segy: obj
        output segy object
    srange: array
        slice range
    step: int
        slice step
    fil: str
        inline field name
    fxl: str
        crossline field name
    fx: str
        geo x field name
    fy: str
        geo y field name
    """

    print("\nexporting labels...")

    if srange is None:
        if slice_type == "inline":
            srange = input_segy.get_range_inlines()
        else:
            srange = input_segy.get_range_crosslines()

    if slice_type == "inline":
        width = input_segy.get_num_crosslines()
    else:
        width = input_segy.get_num_inlines()

    height = input_segy.get_num_samples()

    trace_map = input_segy.get_trace_map()

    assert srange is not None, "srange should be passed"

    count = 0

    for name in sorted(os.listdir(input_path)):

        file = os.path.join(input_path, name)

        if ".png" in file:
            label_image = np.array(Image.open(file), dtype=np.int32)

            assert (label_image.shape[0] == height) and (
                    label_image.shape[1] == width), "image size is not consistent"

            idx = _get_index_from_name(file)

            if slice_type == "inline":
                line_idx = int((idx - input_segy.get_range_inlines()[
                    0]) / input_segy.get_inline_resolution())
            else:
                line_idx = int((idx - input_segy.get_range_crosslines()[
                    0]) / input_segy.get_crossline_resolution())

            if srange[0] <= idx <= srange[1]:
                if (count % step == 0):
                    print(idx)

                    for t in range(label_image.shape[1]):

                        if slice_type == "inline":
                            th = input_segy.trace_header(trace_map[line_idx, t])
                        else:
                            th = input_segy.trace_header(trace_map[t, line_idx])

                        output_segy.write_trace_header(th)

                        output_segy.write_trace_data(label_image[:, t].tolist())

            count += 1

    # if srange is None:
    #     if slice_type == "inline":
    #         srange = input_segy.get_range_inlines()
    #     else:
    #         srange = input_segy.get_range_crosslines()
    #
    # num_slices = len([name for name in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, name))])
    #
    # data_type = 3
    #
    # if slice_type == "inline":
    #     traces = input_segy.get_num_crosslines()
    # else:
    #     traces = input_segy.get_num_inlines()
    #
    # samples = input_segy.get_num_samples()
    # sample_interval = input_segy.get_time_depth_resolution() * 1000
    #
    # output_segy = seisfast.io.SEGY(output_path, int(traces * num_slices), int(samples), data_type)
    #
    # # segy.start_write_mode()
    #
    # output_segy.write_textual_header(None)
    # output_segy.write_binary_header(None, sample_interval)
    #
    # c = 0
    # count = 0
    #
    # for name in os.listdir(input_path):
    #
    #         file = os.path.join(input_path, name)
    #
    #         if ".png" in file:
    #             label_image = np.array(Image.open(file))
    #             idx = _get_index_from_name(file)
    #
    #             if srange[0] <= idx <= srange[1]:
    #                 if (count % step == 0):
    #                     print(idx)
    #                     c = _write_segy_slice(input_segy, output_segy, sample_interval, slice_type, label_image, idx, data_type, c,
    #                                           fil, fxl, fx, fy)
    #
    #             count += 1

    return


# pre stack


def get_cdp_sources(segy, qil=None, qxl=None):
    """
    Returns source/shot id's related to a cdp position.

    If qil or qxl is missing, the whole section is considered.

    Parameters
    ----------

    segy: obj
        segy object

    qil: int
        Inline index (optional).

    qxl: int
        Crossline index (optional).

    Returns
    -------

    active_sources: list

    """

    cdps = segy.get_cdps()

    active_sources = []

    cdp_traces = []

    for s in cdps:

        il, xl = segy.get_il_xl_from_cdp_id(s)

        if qil and qxl:
            if (il == qil) and (xl == qxl):
                cdp_traces += segy.get_cdp_gather(s)
        elif qil:
            if il == qil:
                cdp_traces += segy.get_cdp_gather(s)
        elif qxl:
            if xl == qxl:
                cdp_traces += segy.get_cdp_gather(s)

    sources = segy.get_sources()

    for s in sources:

        source_traces = segy.get_source_gather(s)

        t = set(cdp_traces).intersection(set(source_traces))

        if len(t) > 0:
            active_sources.append(s)

    return active_sources


def get_fold_map(segy):
    """
    Returns a fold map.

    Parameters
    ----------

    segy: obj
        segy object

    Returns
    -------

    fold: 2d array

    """

    rangeil = segy.get_range_inlines()
    rangexl = segy.get_range_crosslines()

    resil = segy.get_inline_resolution()
    resxl = segy.get_crossline_resolution()

    fold = np.zeros((int((rangeil[1] - rangeil[0]) / resil + 1),
                     int((rangexl[1] - rangexl[0]) / resxl + 1)))

    for il in range(rangeil[0], rangeil[1] + 1, resil):
        for xl in range(rangexl[0], rangexl[1] + 1, resxl):
            fold[int((il - rangeil[0]) / resil), int((xl - rangexl[0]) / resxl)] = len(
                segy.get_cdp_gather(il, xl))

    return fold


def get_cdp_gather(segy, il, xl=None, clip=False, clip_perc=2, offset_order=False,
                   return_offsets=False, return_positions=False):
    """
    Returns cdp gather.

    Parameters
    ----------

    segy: obj
        segy object
    il: int
        inline number
    xl: int
        crossline number
    clip: bool
        whether histogram clipping should be performed
    clip_perc: float
        clipping percentage
    offset_order: bool
        whether traces should be sorted by offset
    return_offsets: bool
        whether ordered offsets should also be returned
    return_positions: bool
        whether ordered positions should also be returned
    Returns
    -------

    image: 2d array

    """

    gather = segy.get_cdp_gather(il, xl)

    image = np.zeros((segy.trace_samples(), len(gather)))

    distances = []

    if offset_order:

        for k in range(len(gather)):
            th = segy.trace_header(gather[k])
            sx = th["22 source coordinate x"]
            sy = th["23 source coordinate y"]
            gx = th["24 group coordinate x"]
            gy = th["25 group coordinate y"]
            # distances.append(np.sqrt((sx-gx)**2 + (sy-gy)**2))
            # distances.append(th["02 trace sequence number within segy file"])
            distances.append(th["12 distance from center of source point to the center of the receiver group"])

        idx = np.argsort(np.array(distances), kind='mergesort')
        gather = np.array(gather)[idx].tolist()

        distances = np.array(distances)[idx]

    for k in range(len(gather)):
        image[:, k] = segy.trace_data(gather[k])

    if clip:
        image = _clip_image(image, clip_perc)

    if return_offsets:
        assert offset_order, "offset_order must be true to return offsets"

    if return_positions:
        assert offset_order, "offset_order must be true to return positions"

    result = [image]

    if return_offsets:
        result.append(distances)

    if return_positions:
        result.append(gather)

    return tuple(result) if len(result) > 1 else result[0]


def get_source_gather(segy, source_id, clip=False, clip_perc=2, offset_order=False,
                      return_offsets=False, return_positions=False):
    """
    Returns source gather.

    Parameters
    ----------

    segy: obj
        segy object
    source_id: int
        source id
    clip: bool
        whether histogram clipping should be performed
    clip_perc: float
        clipping percentage
    offset_order: bool
        whether traces should be sorted by offset
    return_offsets: bool
        whether ordered offsets should also be returned
    return_positions: bool
        whether ordered positions should also be returned
    Returns
    -------

    image: 2d array

    """

    gather = segy.get_source_gather(source_id)

    image = np.zeros((segy.trace_samples(), len(gather)))

    distances = []

    if offset_order:

        for k in range(len(gather)):
            th = segy.trace_header(gather[k])
            sx = th["22 source coordinate x"]
            sy = th["23 source coordinate y"]
            gx = th["24 group coordinate x"]
            gy = th["25 group coordinate y"]
            #distances.append(np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2))
            distances.append(th["12 distance from center of source point to the center of the receiver group"])

        idx = np.argsort(np.array(distances), kind='mergesort')
        gather = np.array(gather)[idx].tolist()

    for k in range(len(gather)):
        image[:, k] = segy.trace_data(gather[k])

    if clip:
        image = _clip_image(image, clip_perc)

    if return_offsets:
        assert offset_order, "offset_order must be true to return offsets"

    if return_positions:
        assert offset_order, "offset_order must be true to return positions"

    result = [image]

    if return_offsets:
        result.append(distances)

    if return_positions:
        result.append(gather)

    return tuple(result) if len(result) > 1 else result[0]


def get_receiver_gather(segy, receiver_id, clip=False, clip_perc=2, offset_order=False,
                        return_offsets=False, return_positions=False):
    """
    Returns receiver gather.

    Parameters
    ----------

    segy: obj
        segy object
    receiver_id : str
        receiver_id
    clip: bool
        whether histogram clipping should be performed
    clip_perc: float
        clipping percentage
    offset_order: bool
        whether traces should be sorted by offset
    return_offsets: bool
        whether ordered offsets should also be returned
    return_positions: bool
        whether ordered positions should also be returned
    Returns
    -------

    image: 2d array

    """

    gather = segy.get_receiver_gather(receiver_id)

    image = np.zeros((segy.trace_samples(), len(gather)))

    distances = []

    if offset_order:

        for k in range(len(gather)):
            th = segy.trace_header(gather[k])
            sx = th["22 source coordinate x"]
            sy = th["23 source coordinate y"]
            gx = th["24 group coordinate x"]
            gy = th["25 group coordinate y"]
            #distances.append(np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2))
            distances.append(th["12 distance from center of source point to the center of the receiver group"])

        idx = np.argsort(np.array(distances), kind='mergesort')
        gather = np.array(gather)[idx].tolist()

    for k in range(len(gather)):
        image[:, k] = segy.trace_data(gather[k])

    if clip:
        image = _clip_image(image, clip_perc)

    if return_offsets:
        assert offset_order, "offset_order must be true to return offsets"

    if return_positions:
        assert offset_order, "offset_order must be true to return positions"

    result = [image]

    if return_offsets:
        result.append(distances)

    if return_positions:
        result.append(gather)

    return tuple(result) if len(result) > 1 else result[0]


def get_source_positions(segy):
    """
    Returns source positions.

    Parameters
    ----------

    segy: obj
        segy object

    Returns
    -------

    sx: array
        geo x coordinates
    sy: array
        geo y coordinates

    """

    sources = segy.get_sources()

    sx = []
    sy = []

    for s in sources:

        gather = segy.get_source_gather(s)

        for k in range(1):
            th = segy.trace_header(gather[k])

            sx.append(th["22 source coordinate x"])
            sy.append(th["23 source coordinate y"])

    return sx, sy


def get_receiver_positions(segy):
    """
    Returns receiver positions.

    Parameters
    ----------

    segy: obj
        segy object

    Returns
    -------

    gx: array
        geo x coordinates
    gy: array
        geo y coordinates

    """

    receivers = segy.get_receivers()

    gx = []
    gy = []

    for s in receivers:

        gather = segy.get_receiver_gather(s)

        for k in range(1):
            th = segy.trace_header(gather[k])

            gx.append(th["24 group coordinate x"])
            gy.append(th["25 group coordinate y"])

    return gx, gy


def get_cdp_positions(segy):
    """
    Returns cdp positions.

    Parameters
    ----------

    segy: obj
        segy object

    Returns
    -------

    cdpx: array
        geo x coordinates
    cdpy: array
        geo y coordinates

    """

    cdps = segy.get_cdps()

    cdpx = []
    cdpy = []

    for s in cdps:

        gather = segy.get_cdp_gather(s)

        for k in range(1):
            th = segy.trace_header(gather[k])

            cdpx.append(th["72 x coordinate of ensemble position (cdp)"])
            cdpy.append(th["73 y coordinate of ensemble position (cdp)"])

    return cdpx, cdpy


def get_source_gather_positions(segy, source_id):
    """
    Returns source gather positions.

    Parameters
    ----------

    segy: obj
        segy object
    source_id : str
        source id

    Returns
    -------

    sx: float
        source geo x coordinate
    sy: float
        source geo y coordinate
    cdpx: array
        cdp geo x coordinates
    cdpy: array
        cdp geo y coordinates
    gx: array
        receiver geo x coordinates
    gy: array
        receiver geo y coordinates
    """

    gather = segy.get_source_gather(source_id)

    sx = None
    sy = None
    cdpx = []
    cdpy = []
    gx = []
    gy = []

    for k in range(len(gather)):
        th = segy.trace_header(gather[k])

        if sx is None:
            sx = th["22 source coordinate x"]
            sy = th["23 source coordinate y"]

        cdpx.append(th["72 x coordinate of ensemble position (cdp)"])
        cdpy.append(th["73 y coordinate of ensemble position (cdp)"])

        gx.append(th["24 group coordinate x"])
        gy.append(th["25 group coordinate y"])

    return sx, sy, cdpx, cdpy, gx, gy


def get_cdp_gather_positions(segy, il, xl=None):
    """
    Returns cdp gather positions.

    Parameters
    ----------

    segy: obj
        segy object
    il : int
        inline
    xl : int
        crossline (optional). If not given, il is considered as a cdp id

    Returns
    -------

    sx: array
        source geo x coordinates
    sy: array
        source geo y coordinates
    cdpx: float
        cdp geo x coordinate
    cdpy: float
        cdp geo y coordinate
    gx: array
        receiver geo x coordinates
    gy: array
        receiver geo y coordinates
    """

    gather = segy.get_cdp_gather(il, xl)

    sx = []
    sy = []
    cdpx = None
    cdpy = None
    gx = []
    gy = []

    for k in range(len(gather)):
        th = segy.trace_header(gather[k])

        if cdpx is None:
            cdpx = th["72 x coordinate of ensemble position (cdp)"]
            cdpy = th["73 y coordinate of ensemble position (cdp)"]

        sx.append(th["22 source coordinate x"])
        sy.append(th["23 source coordinate y"])

        gx.append(th["24 group coordinate x"])
        gy.append(th["25 group coordinate y"])

    return sx, sy, cdpx, cdpy, gx, gy


def get_receiver_gather_positions(segy, receiver_id):
    """
    Returns receiver gather positions.

    Parameters
    ----------

    segy: obj
        segy object
    receiver_id : str
        receiver id

    Returns
    -------

    sx: array
        source geo x coordinates
    sy: array
        source geo y coordinates
    cdpx: array
        cdp geo x coordinates
    cdpy: array
        cdp geo y coordinates
    gx: float
        receiver geo x coordinate
    gy: float
        receiver geo y coordinate
    """

    gather = segy.get_receiver_gather(receiver_id)

    sx = []
    sy = []
    cdpx = []
    cdpy = []
    gx = None
    gy = None

    for k in range(len(gather)):
        th = segy.trace_header(gather[k])

        if gx is None:
            gx = th["24 group coordinate x"]
            gy = th["25 group coordinate y"]

        cdpx.append(th["72 x coordinate of ensemble position (cdp)"])
        cdpy.append(th["73 y coordinate of ensemble position (cdp)"])

        sx.append(th["22 source coordinate x"])
        sy.append(th["23 source coordinate y"])

    return sx, sy, cdpx, cdpy, gx, gy


def nmo_corrected_cdp_gather(segy, il, xl, velocities, clip=False, clip_perc=2, initial_time=0,
                             sample_rate=0.002):
    """
    Returns a NMO-corrected cdp gather.

    Parameters
    ----------

    segy: obj
        segy object
    il: int
        inline number
    xl: int
        crossline number
    velocities: array
        velocity values for each time.
    clip: bool
        whether histogram clipping should be performed
    clip_perc: float
        clipping percentage
    initial_time: float
        initial time
    sample_rate: float
        sample rate

    Returns
    -------

    output: 2d array

    """

    gather, offsets = get_cdp_gather(segy, il, xl, offset_order=True, return_offsets=True)

    output = np.zeros(gather.shape)

    assert len(velocities) == gather.shape[
        0], "velocity array length should be equal to gather height"

    for trace in range(gather.shape[1]):
        output[:, trace] = _nmo(gather[:, trace], offsets[trace], velocities, initial_time,
                                sample_rate)

    if clip:
        output = _clip_image(output, clip_perc)

    return output


def get_semblance_image(segy, il, xl, velocity_range=(), velocity_step=100, window_size=1,
                        initial_time=0, sample_rate=0.002):
    """
    Returns a cdp semblance image.

    Parameters
    ----------

    segy: obj
        segy object
    il: int
        inline number
    xl: int
        crossline number
    velocity_range: tuple
        velocity value range
    velocity_step: number
        velocity increment
    window_size: number
        semblance window size
    initial_time: float
        initial time
    sample_rate: float
        sample rate

    Returns
    -------

    semblance: 2d array

    """

    gather, offsets = get_cdp_gather(segy, il, xl, offset_order=True, return_offsets=True)

    semblance = np.zeros(
        (gather.shape[0], int(((velocity_range[1] - velocity_range[0]) / velocity_step) + 1)))

    velocities = np.arange(velocity_range[0], velocity_range[1] + 1, velocity_step)

    for v in range(len(velocities)):

        new_gather = np.zeros(gather.shape)

        print(velocities[v])

        for trace in range(gather.shape[1]):
            new_gather[:, trace] = _nmo(gather[:, trace], offsets[trace],
                                        [velocities[v]] * gather.shape[0], initial_time,
                                        sample_rate)

        for t in range(window_size, gather.shape[0] - window_size):
            s = _trace_semblance(new_gather[t - window_size:t + window_size + 1, :])
            semblance[t, v] = s

    semblance[np.isnan(semblance)] = 0

    return semblance


# general

def get_amplitude_stats(segy):
    """
    Returns amplitude statistics.

    Parameters
    ----------

    segy: obj
        segy object

    Returns
    -------

    min_amp: float

    max_amp: float

    """

    min_amp = sys.float_info.max
    max_amp = 0

    num_traces = segy.get_num_traces()

    print("\nanalyzing traces...")

    for t in range(num_traces):

        trace = np.array(segy.trace_data(t))

        if trace.min() < min_amp:
            min_amp = trace.min()

        if trace.max() > max_amp:
            max_amp = trace.max()

    return min_amp, max_amp


def get_equalization_parameters(segy, type="inline", step=None, increment=None):
    """
    Computes equalization parameters.

    Parameters
    ----------

    segy: obj
        segy object
    type: str
        "source", "receiver", "cdp", "inline", "crossline"
    step: int
        step for images/gathers to consider
    increment: float
        increment for standard deviation

    Returns
    -------

    mean: float

    std_dev: float

    """

    mean = 0
    std_dev = None

    if type == "source":

        print("\nanalyzing sources...")

        sources = segy.get_sources()

        total = len(np.arange(0, len(sources), step))
        count = 1
        curr_s = None

        while count <= total:

            print("\nnew trial: " + str(std_dev))

            if curr_s is None:
                curr_s = 0

            for s in range(curr_s, len(sources), step):

                print(str(count) + " of " + str(total))

                samples = get_source_gather(segy, sources[s])

                if std_dev is None:
                    if not np.isnan(samples.ravel().std()):
                        std_dev = samples.ravel().std()
                        print(std_dev)

                if std_dev is None:
                    continue

                if increment is None:
                    increment = _get_increment(std_dev)

                curr_s = s

                if not _check_equalization_parameters(samples, mean, std_dev):
                    std_dev += increment
                    break

                count += 1

    elif type == "receiver":

        print("\nanalyzing receivers...")

        receivers = segy.get_receivers()

        total = len(np.arange(0, len(receivers), step))
        count = 1
        curr_s = None

        while count <= total:

            print("\nnew trial: " + str(std_dev))

            if curr_s is None:
                curr_s = 0

            for s in range(curr_s, len(receivers), step):

                print(str(count) + " of " + str(total))

                samples = get_receiver_gather(segy, receivers[s])

                if std_dev is None:
                    if not np.isnan(samples.ravel().std()):
                        std_dev = samples.ravel().std()
                        print(std_dev)

                if std_dev is None:
                    continue

                if increment is None:
                    increment = _get_increment(std_dev)

                curr_s = s

                if not _check_equalization_parameters(samples, mean, std_dev):
                    std_dev += increment
                    break

                count += 1

    elif type == "cdp":

        print("\nanalyzing cdp's...")

        range_inlines = segy.get_range_inlines()
        inline_resolution = segy.get_inline_resolution()
        num_xlines = segy.get_num_crosslines()

        range_crosslines = segy.get_range_crosslines()
        crossline_resolution = segy.get_crossline_resolution()

        inlines = np.arange(range_inlines[0], range_inlines[1] + 1, inline_resolution).tolist()
        crosslines = np.arange(range_crosslines[0], range_crosslines[1] + 1,
                               crossline_resolution).tolist()

        cdps = []

        for il in inlines:
            for xl in crosslines:
                cdp_id = int((il - range_inlines[0]) / inline_resolution) * num_xlines + int(
                    (xl - range_crosslines[0]) / crossline_resolution)
                cdps.append(cdp_id)

        total = len(np.arange(0, len(cdps), step))
        count = 1
        curr_s = None

        while count <= total:

            print("\nnew trial: " + str(std_dev))

            if curr_s is None:
                curr_s = 0

            for s in range(curr_s, len(cdps), step):

                print(str(count) + " of " + str(total))

                samples = get_cdp_gather(segy, cdps[s])

                if std_dev is None:
                    if not np.isnan(samples.ravel().std()):
                        std_dev = samples.ravel().std()
                        print(std_dev)

                if std_dev is None:
                    continue

                if increment is None:
                    increment = _get_increment(std_dev)

                curr_s = s

                if not _check_equalization_parameters(samples, mean, std_dev):
                    std_dev += increment
                    break

                count += 1

    elif type == "inline":

        range_inlines = segy.get_range_inlines()

        if step is None:
            inline_resolution = segy.get_inline_resolution()
        else:
            inline_resolution = step

        print("\nanalyzing inlines...")

        total = len(np.arange(range_inlines[0], range_inlines[1] + 1, inline_resolution))
        count = 1
        curr_il = None

        while count <= total:

            print("\nnew trial: " + str(std_dev))

            if curr_il is None:
                curr_il = range_inlines[0]

            for il in range(curr_il, range_inlines[1] + 1, inline_resolution):

                print(str(count) + " of " + str(total))

                samples = segy.get_inline(int((il - range_inlines[0]) / inline_resolution))

                if std_dev is None:
                    if not np.isnan(samples.ravel().std()):
                        std_dev = samples.ravel().std()
                        print(std_dev)

                if std_dev is None:
                    continue

                if increment is None:
                    increment = _get_increment(std_dev)

                curr_il = il

                if not _check_equalization_parameters(samples, mean, std_dev):
                    std_dev += increment
                    break

                count += 1

    elif type == "crossline":

        range_crosslines = segy.get_range_crosslines()

        if step is None:
            crossline_resolution = segy.get_crossline_resolution()
        else:
            crossline_resolution = step

        print("\nanalyzing crosslines...")

        total = len(
            np.arange(range_crosslines[0], range_crosslines[1] + 1, crossline_resolution))
        count = 1
        curr_xl = None

        while count <= total:

            print("\nnew trial: " + str(std_dev))

            if curr_xl is None:
                curr_xl = range_crosslines[0]

            for xl in range(curr_xl, range_crosslines[1] + 1, crossline_resolution):

                print(str(count) + " of " + str(total))

                samples = segy.get_crossline(
                    int((xl - range_crosslines[0]) / crossline_resolution))

                if std_dev is None:
                    if not np.isnan(samples.ravel().std()):
                        std_dev = samples.ravel().std()
                        print(std_dev)

                if std_dev is None:
                    continue

                if increment is None:
                    increment = _get_increment(std_dev)

                curr_xl = xl

                if not _check_equalization_parameters(samples, mean, std_dev):
                    std_dev += increment
                    break

                count += 1

    return mean, std_dev


def get_writable_pre_stack_segy(input_segy, output_path):
    """
    Returns a writable PreStackSEGY object which is a shallow copy of input segy.

    Parameters
    ----------

    input_segy: obj
        input segy object
    output_path: str
        output segy file path
    """

    assert not os.path.isfile(output_path), "output file already exists"

    data_type = 5
    traces = input_segy.get_num_traces()
    samples = input_segy.get_num_samples()

    output_segy = seisfast.io.PreStackSEGY(output_path, traces, samples, data_type)

    output_segy.write_textual_header(input_segy.textual_header())
    output_segy.write_binary_header(input_segy.binary_header())

    return output_segy


def write_gather_2_segy(input_segy, data, gather_type, gather_id, output_segy,
                        offset_order=False):
    """
    Exports gather image to segy.

    Parameters
    ----------

    input_segy: obj
        input segy object
    data: 2D numpy array
        seisfast to be exported
    gather_type: str
        "source", "receiver" or "cdp"
    gather_id: mixed
        gather id
    output_segy: obj
        output segy object
    offset_order: bool
        whether seisfast is ordered by offset
    """

    gather = []

    if gather_type == "cdp":
        if offset_order:
            _, gather = get_cdp_gather(input_segy, gather_id, offset_order=True,
                                       return_positions=True)
        else:
            gather = input_segy.get_cdp_gather(gather_id)
    elif gather_type == "source":
        if offset_order:
            _, gather = get_source_gather(input_segy, gather_id, offset_order=True,
                                          return_positions=True)
        else:
            gather = input_segy.get_source_gather(gather_id)
    elif gather_type == "receiver":
        if offset_order:
            _, gather = get_receiver_gather(input_segy, gather_id, offset_order=True,
                                            return_positions=True)
        else:
            gather = input_segy.get_receiver_gather(gather_id)

    assert len(gather) == data.shape[1], "seisfast and gather should have the same size"

    for t in range(data.shape[1]):
        th = input_segy.trace_header(gather[t])
        output_segy.write_trace_header(th)

        output_segy.write_trace_data(data[:, t].tolist())

    return
