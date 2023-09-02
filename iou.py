"""
Code graciously donated by Morgan Schwartz
"""
import warnings

import numpy as np
# from deepcell.applications import CellTracking
from skimage.segmentation import relabel_sequential


def find_boundary_box(object_mask):
    y_coordinates = (np.sum(object_mask, axis=0) > 0).nonzero()[0]
    x_coordinates = (np.sum(object_mask, axis=1) > 0).nonzero()[0]
    x_min, y_min, x_max, y_max = (
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    )
    return [x_min, y_min, x_max, y_max]


def iou_vec(A, B):
    low = np.s_[..., :2]
    high = np.s_[..., 2:]
    A, B = A.copy(), B.copy()
    A[high] += 1
    B[high] += 1
    intrs = (
        np.maximum(0, np.minimum(A[high], B[high]) - np.maximum(A[low], B[low]))
    ).prod(-1)
    return intrs / ((A[high] - A[low]).prod(-1) + (B[high] - B[low]).prod(-1) - intrs)


def match_cell(
    iou_matrix,
    iou_orig,
    cells,
    next_cells,
    relabeled_next,
    threshold,
    next_img,
):
    # Keep track of which (next_img) cells don't have matches
    unmatched_cells = set()
    # Don't reuse labels (if multiple cells in next_img match)
    used_cells_src = set()
    # Identify index of cell in first frame that matches to each cell in
    # second frame
    max_indices = np.argmax(iou_matrix, axis=0)

    for next_cell, matched_cell in enumerate(max_indices):
        if next_cell == 0:
            continue
        # If the matched_cells has been used or is background,
        # record next_cell as unmatched
        # Also puts skipped (nonexistent) labels into unmatched cells,
        # neccesitates relabeling
        if matched_cell in used_cells_src or matched_cell == 0:
            unmatched_cells.add(next_cell)
            continue
        match_fill = cells[matched_cell - 1]
        next_fill = next_cells[next_cell - 1]
        # Check if matched_cell has more than one match
        count_matches = np.count_nonzero(max_indices == matched_cell)

        if count_matches == 1:
            # Add matched cell to relabeled image if it meets iou threshold
            if iou_matrix[matched_cell][next_cell] > threshold:
                relabeled_next = np.where(
                    next_img == next_fill, match_fill, relabeled_next
                )

            else:
                unmatched_cells.add(next_cell)
        # If more than one match, look for best match
        else:
            # Find next_cell with highest iou
            best_matched_next = np.argmax(iou_matrix, axis=1)[matched_cell]
            # If best_matched_next is not next_cell, then record unmatched
            if best_matched_next != next_cell:
                unmatched_cells.add(next_cell)
                continue

            # Check if the match iou meets threshold before saving as match
            if iou_matrix[matched_cell][best_matched_next] > threshold:
                relabeled_next = np.where(
                    next_img == next_cells[best_matched_next - 1],
                    match_fill,
                    relabeled_next,
                )
            else:
                unmatched_cells.add(next_cell)
        # Record matched_cell as used because all possible matches
        # have been checked
        # and either recorded or rejected based on threshold
        used_cells_src.add(matched_cell)
    iou_updated = iou_matrix
    if len(used_cells_src) > 0:
        iou_updated = np.delete(iou_matrix, np.array(list(used_cells_src)), axis=0)
    if len(unmatched_cells) > 0:
        iou_updated = iou_updated[:, np.array([0, *list(unmatched_cells)])]

    if len(used_cells_src) > 0:
        cells = np.delete(cells, np.array(list(used_cells_src)) - 1, axis=0)

    if len(unmatched_cells) > 0:
        next_cells = next_cells[np.array(list(unmatched_cells)) - 1]
    else:
        next_cells = []
    # slice the iou matrix so that it does not have mapped old/next
    # identities
    return (
        iou_updated,
        cells,
        next_cells,
        relabeled_next,
        unmatched_cells,
        used_cells_src,
    )


def create_new_lineage(y):
    """Create a blank lineage dict for ids that have already been
    linked via IOU. Link only based on overlap,
    so there are no divisions/daughters/parents/deaths

    Args:
        y: (np.array) label image stack.

    Returns:
        dict: a nested dict (lineage for .trk)
    """
    new_lineage = {}
    for i, frame in enumerate(y):
        # Add to frames field if ID exists
        cells_in_frame = np.unique(frame)
        cells_in_frame = np.delete(cells_in_frame, np.where(cells_in_frame == 0))
        cells_in_frame = list(cells_in_frame)

        for cell in cells_in_frame:
            cell = int(cell)
            if cell in new_lineage:
                new_lineage[cell]["frames"].append(i)

            # Or create a new dict because its a new cell
            else:
                new_lineage[cell] = {
                    "label": cell,
                    "frames": [i],
                    "daughters": [],
                    "capped": False,
                    "frame_div": None,
                    "parent": None,
                }

    return new_lineage


def link_ids_via_iou(img, next_img, threshold=0.1):
    """
    Link labels for next_img based on intersection over union (IoU)
    with current img. Threshold is cutoff for matching. Cells without
    a match in current img (new cells) get a new label.
    Args:
        img: (np.array) first frame with labels to be linked
        next_img: (np.array) next frame with labels to be replaced
        threshold: (optional, double) cuttoff value for IoU
    Returns:
        np.array: next_img with labels matched to img
    """

    # relabel to remove skipped values and make things easy to keep track of
    next_img = relabel_sequential(next_img)[0]
    # Identify unique, nonzero cell labels in each frame
    cells = np.unique(img[np.nonzero(img)])
    orig_cells = cells
    next_cells = np.unique(next_img[np.nonzero(next_img)])

    # No cells to work with
    if len(cells) == 0:
        warnings.warn("First frame is empty")
        return next_img

    # No values to reassign
    if len(next_cells) == 0:
        warnings.warn("Next frame is empty")
        return next_img

    # cell_ind = 0 --> background
    cell_boundaries = np.array([find_boundary_box(img == cell) for cell in cells])
    next_cell_boundaries = np.array(
        [find_boundary_box(next_img == next_cell) for next_cell in next_cells]
    )
    iou_bb = iou_vec(cell_boundaries[:, None], next_cell_boundaries[None])
    iou = np.pad(iou_bb, ((1, 0), (1, 0)), "constant")
    relabeled_next = np.zeros(next_img.shape, dtype=next_img.dtype)

    (
        iou_updated,
        cells,
        next_cells,
        relabeled_next,
        unmatched_cells,
        used_cells_src,
    ) = match_cell(iou, iou, cells, next_cells, relabeled_next, threshold, next_img)

    while (
        ((iou_updated > threshold).sum().sum() > 0)
        and (len(cells) > 0)
        and (len(next_cells) > 0)
    ):
        (
            iou_updated,
            cells,
            next_cells,
            relabeled_next,
            unmatched_cells,
            used_cells_src,
        ) = match_cell(
            iou_updated,
            iou,
            cells,
            next_cells,
            relabeled_next,
            threshold,
            next_img,
        )

    # Finish adding unmatched next_cells to relabeled_img with new label value
    # To account for any new cells that appear, create labels by adding to the
    # max number of cells

    unmatched_cells = set(next_cells)
    if len(unmatched_cells) > 0:
        # Find max label value between img and relabeled_next and add one to
        # create new label
        new_label = max(np.max(orig_cells), np.max(relabeled_next)) + 1

        # Add each unmatched cell to relabeled_next with a new unique label
        # value
        for unmatched_cell in unmatched_cells:
            relabeled_next = np.where(
                next_img == unmatched_cell, new_label, relabeled_next
            )
            new_label += 1

    # Check that we are returning same number of cells as we started with
    assert len(np.unique(relabeled_next)) == len(np.unique(next_img))

    return relabeled_next


# class IOUTracking(CellTracking):

def predict(y_old, **kwargs):
    """
    Link labels across labeled movie based on intersection over union (iou)

    Args:
        y: (np array) label image stack.

    Returns:
        dict: a nested dict (lineage for .trk)
    """
    y_tracked = np.zeros_like(y_old)
    # link up ids based on first frame
    num_frames = y_old.shape[0]
    y_tracked[0] = y_old[0]
    for frame in range(num_frames - 1):
        y_tracked[frame + 1] = link_ids_via_iou(y_tracked[frame], y_old[frame + 1])

    # create a new lineage based on the linked up ids
    # new_lineage_cyto = create_new_lineage(y_old)
    return y_tracked
