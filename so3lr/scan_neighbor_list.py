"""Neighbor list function that uses `lax.scan` to compute the distance between particles

This implementation was inspired by and copies parts of the original code from
a similar function provided in the JAX-SPH project:
https://github.com/tumaer/jax-sph

published under the following license:
=============================================================================
MIT License

Copyright (c) 2024 Chair of Aerodynamics and Fluid Mechanics @ TUM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=============================================================================
"""

from functools import partial
from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit

from jax_md import space
from jax_md.partition import (
    MaskFn,
    NeighborFn,
    NeighborList,
    NeighborListFns,
    NeighborListFormat,
    PartitionError,
    PartitionErrorCode,
    _displacement_or_metric_to_metric_sq,
    _fractional_cell_size,
    _neighboring_cells,
    cell_list,
    is_format_valid,
    is_box_valid,
    is_sparse,
    shift_array,
)
from jax_md.partition import neighbor_list as vmap_neighbor_list

PEC = PartitionErrorCode


def get_particle_cells(idx, cl_capacity, N):
    """
    Given a cell list idx of shape (nx, ny, nz, cell_capacity), we first
    enumerate each cell and then return a list of shape (N,) containing the
    number of the cell each particle belongs to.
    """
    # containes particle indices in each cell (num_cells, cell_capacity)
    idx = idx.reshape(-1, cl_capacity)

    # (num_cells, cell_capacity) of
    # [[0,0,...0],[1,1,...1],...,[num_cells-1,num_cells-1,...num_cells-1]
    list_cells = jnp.broadcast_to(jnp.arange(idx.shape[0])[:, None], idx.shape)

    idx = jnp.reshape(idx, (-1,))  # flatten
    list_cells = jnp.reshape(list_cells, (-1,))  # flatten

    ordering = jnp.argsort(idx)  # each particle is only once in the cell list
    particle_cells = list_cells[ordering][:N]
    return particle_cells


def scan_neighbor_list(
    displacement_or_metric: space.DisplacementOrMetricFn,
    box: space.Box,
    r_cutoff: float,
    dr_threshold: float = 0.0,
    capacity_multiplier: float = 1.25,
    buffer_size_multiplier: float = 1.25,
    minimum_cell_size_multiplier: float = 1.0,
    disable_cell_list: bool = False,
    mask_self: bool = True,
    fractional_coordinates: bool = False,
    custom_mask_function: Optional[MaskFn] = None,
    format: NeighborListFormat = NeighborListFormat.Sparse,
    num_partitions: int = 8,
    **static_kwargs,
) -> NeighborFn:
    """Modified JAX-MD neighbor list function that uses `lax.scan` to compute the
    distance between particles to save memory.

    Original: https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py

    Returns a function that builds a list neighbors for collections of points.

    Neighbor lists must balance the need to be jit compatible with the fact that
    under a jit the maximum number of neighbors cannot change (owing to static
    shape requirements). To deal with this, our `neighbor_list` returns a
    `NeighborListFns` object that contains two functions: 1)
    `neighbor_fn.allocate` create a new neighbor list and 2) `neighbor_fn.update`
    updates an existing neighbor list. Neighbor lists themselves additionally
    have a convenience `update` member function.

    Note that allocation of a new neighbor list cannot be jit compiled since it
    uses the positions to infer the maximum number of neighbors (along with
    additional space specified by the `capacity_multiplier`). Updating the
    neighbor list can be jit compiled; if the neighbor list capacity is not
    sufficient to store all the neighbors, the `did_buffer_overflow` bit
    will be set to `True` and a new neighbor list will need to be reallocated.

    Here is a typical example of a simulation loop with neighbor lists:

    .. code-block:: python

        init_fn, apply_fn = simulate.nve(energy_fn, shift, 1e-3)
        exact_init_fn, exact_apply_fn = simulate.nve(exact_energy_fn, shift, 1e-3)

        nbrs = neighbor_fn.allocate(R)
        state = init_fn(random.PRNGKey(0), R, neighbor_idx=nbrs.idx)

        def body_fn(i, state):
        state, nbrs = state
        nbrs = nbrs.update(state.position)
        state = apply_fn(state, neighbor_idx=nbrs.idx)
        return state, nbrs

        step = 0
        for _ in range(20):
        new_state, nbrs = lax.fori_loop(0, 100, body_fn, (state, nbrs))
        if nbrs.did_buffer_overflow:
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += 1

    Args:
        displacement: A function `d(R_a, R_b)` that computes the displacement
        between pairs of points.
        box: Either a float specifying the size of the box or an array of
        shape `[spatial_dim]` specifying the box size in each spatial dimension.
        r_cutoff: A scalar specifying the neighborhood radius.
        dr_threshold: A scalar specifying the maximum distance particles can move
        before rebuilding the neighbor list.
        capacity_multiplier: A floating point scalar specifying the fractional
        increase in maximum neighborhood occupancy we allocate compared with the
        maximum in the example positions.
        disable_cell_list: An optional boolean. If set to `True` then the neighbor
        list is constructed using only distances. This can be useful for
        debugging but should generally be left as `False`.
        mask_self: An optional boolean. Determines whether points can consider
        themselves to be their own neighbors.
        custom_mask_function: An optional function. Takes the neighbor array
        and masks selected elements. Note: The input array to the function is
        `(n_particles, m)` where the index of particle 1 is in index in the first
        dimension of the array, the index of particle 2 is given by the value in
        the array
        fractional_coordinates: An optional boolean. Specifies whether positions
        will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
        If this is set to True then the `box_size` will be set to `1.0` and the
        cell size used in the cell list will be set to `cutoff / box_size`.
        format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
        for details about the different choices for formats. Defaults to `Dense`.
        **static_kwargs: kwargs that get threaded through the calculation of
        example positions.
    Returns:
        A NeighborListFns object that contains a method to allocate a new neighbor
        list and a method to update an existing neighbor list.
    """
    # assert disable_cell_list is False, "Works only with a cell list"
    # assert not fractional_coordinates, "Works only with real coordinates"
    # assert format == NeighborListFormat.Sparse, "Works only with sparse neighbor list"
    assert custom_mask_function is None, "Custom masking not implemented"

    is_format_valid(format)
    box = lax.stop_gradient(box)
    r_cutoff = lax.stop_gradient(r_cutoff)
    dr_threshold = lax.stop_gradient(dr_threshold)

    box = jnp.float32(box)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff**2
    threshold_sq = (dr_threshold / jnp.float32(2)) ** 2
    metric_sq = _displacement_or_metric_to_metric_sq(displacement_or_metric)

    # cell_size = cutoff
    # assert jnp.all(cell_size < box / 3.0), "Don't use scan with very few cells"

    def neighbor_list_fn(
        position: jnp.ndarray,
        neighbors: Optional[NeighborList] = None,
        extra_capacity: int = 0,
        **kwargs,
    ) -> NeighborList:
        def neighbor_fn(position_and_error, max_occupancy=None):
            position, err = position_and_error
            N, dim = position.shape
            cl_fn = None
            cl = None
            cell_size = None

            if not disable_cell_list:
                if neighbors is None:
                    _box = kwargs.get('box', box)
                    cell_size = cutoff * minimum_cell_size_multiplier
                    if fractional_coordinates:
                        err = err.update(PEC.MALFORMED_BOX, is_box_valid(_box))
                        cell_size = _fractional_cell_size(
                            _box, cutoff) * minimum_cell_size_multiplier
                        _box = 1.0
                    if jnp.all(cell_size < _box / 3.):
                        cl_fn = cell_list(
                            _box, cell_size, buffer_size_multiplier=buffer_size_multiplier)
                        cl = cl_fn.allocate(
                            position, extra_capacity=extra_capacity)
                else:
                    cell_size = neighbors.cell_size
                    cl_fn = neighbors.cell_list_fn
                    if cl_fn is not None:
                        cl = cl_fn.update(
                            position, neighbors.cell_list_capacity)

                # if neighbors is None:  # cl.shape = (nx, ny, nz, cell_capacity, dim)
                #     cell_size = cutoff
                #     cl_fn = cell_list(box, cell_size, capacity_multiplier)
                #     cl = cl_fn.allocate(position, extra_capacity=extra_capacity)
                # else:
                #     cell_size = neighbors.cell_size
                #     cl_fn = neighbors.cell_list_fn
                #     if cl_fn is not None:
                #         cl = cl_fn.update(position, neighbors.cell_list_capacity)

            if cl is None:
                # cl_capacity = None
                # idx = candidate_fn(position.shape)

                # raise NotImplementedError(
                #    "Cell list is None, not yet implemented for this neighbor list backend."
                # )

                cl_capacity = N

                # idx = jnp.zeros((1, 1, 1, cl_capacity), dtype=jnp.int32)
                # cell_idx = [idx]  # shape: (1, 1, 1, cl_capacity, 1)
                # cell_idx = jnp.concatenate(cell_idx, axis=-2)
                # cell_idx = jnp.reshape(cell_idx, (-1, cell_idx.shape[-2]))

                cell_idx = (jnp.arange(N, dtype=jnp.int32)[:, None]).T

                considered_neighbors = N
                num_cells = 1

                particle_cells = jnp.zeros((N,), dtype=jnp.int32)

                #volumetric_factor = 1.0

            else:
                # err = err.update(PEC.CELL_LIST_OVERFLOW, cl.did_buffer_overflow)
                # idx = cell_list_candidate_fn(cl.id_buffer, position.shape)
                # print(f"Neighbor list: n_atoms={position.shape[0]}, buffer_size={cl.id_buffer.size}")
                # cl_capacity = cl.cell_capacity

                err = err.update(PEC.CELL_LIST_OVERFLOW,
                                 cl.did_buffer_overflow)
                cl_capacity = cl.cell_capacity

                idx = cl.id_buffer

                cell_idx = [idx]  # shape: (nx, ny, nz, cell_capacity, 1)

                for dindex in _neighboring_cells(dim):
                    if np.all(dindex == 0):
                        continue
                    cell_idx += [shift_array(idx, dindex)]

                cell_idx = jnp.concatenate(cell_idx, axis=-2)
                cell_idx = jnp.reshape(cell_idx, (-1, cell_idx.shape[-2]))
                num_cells, considered_neighbors = cell_idx.shape

                particle_cells = get_particle_cells(idx, cl_capacity, N)

                # if dim == 2:
                #     # the area of a circle with r=1/3 is 0.34907
                #     volumetric_factor = 0.34907
                # elif dim == 3:
                #     # the volume of a sphere with r=1/3 is 0.15514
                #     volumetric_factor = 0.15514
            volumetric_factor = 0.5235987755982988
            if format is NeighborListFormat.OrderedSparse:
                volumetric_factor /= 2

            d = partial(metric_sq, **kwargs)
            d = space.map_bond(d)

            # number of particles per partition N_sub
            # np.ceil used to pad last partition with < num_partitions entries
            N_sub = int(np.ceil(N / num_partitions))
            num_pad = N_sub * num_partitions - N
            particle_cells = jnp.pad(
                particle_cells,
                (
                    0,
                    num_pad,
                ),
                constant_values=-1,
            )

            num_edges_sub = int(
                N_sub * considered_neighbors * volumetric_factor * capacity_multiplier
            )

            def scan_body(carry, input):
                """Compute neighbors over a subset of particles

                The largest object here is of size (N_sub*considered_neighbors), where
                considered_neighbors in 3D is 27 * cell_capacity.
                """

                occupancy = carry
                slice_from = input

                _entries = lax.dynamic_slice(
                    particle_cells, (slice_from,), (N_sub,))
                _idx = cell_idx[_entries]

                if mask_self:
                    particle_idx = slice_from + jnp.arange(N_sub)
                    _idx = jnp.where(_idx == particle_idx[:, None], N, _idx)

                if num_pad > 0:
                    _idx = jnp.where(_entries[:, None] != -1, _idx, N)

                sender_idx = (
                    jnp.broadcast_to(
                        jnp.arange(N_sub, dtype="int32")[:, None], _idx.shape
                    )
                    + slice_from
                )
                if num_pad > 0:
                    sender_idx = jnp.clip(sender_idx, a_max=N)

                sender_idx = jnp.reshape(sender_idx, (-1,))
                receiver_idx = jnp.reshape(_idx, (-1,))
                dR = d(position[sender_idx], position[receiver_idx])

                mask = (dR < cutoff_sq) & (receiver_idx < N)
                out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)
                if format is NeighborListFormat.OrderedSparse:
                    mask = mask & (receiver_idx < sender_idx)

                cumsum = jnp.cumsum(mask)
                index = jnp.where(mask, cumsum - 1,
                                  considered_neighbors * N - 1)
                receiver_idx = out_idx.at[index].set(receiver_idx)
                sender_idx = out_idx.at[index].set(sender_idx)
                occupancy += cumsum[-1]

                carry = occupancy
                y = jnp.stack(
                    (receiver_idx[:num_edges_sub], sender_idx[:num_edges_sub])
                )
                overflow = cumsum[-1] > num_edges_sub
                return carry, (y, overflow)

            carry = jnp.array(0)
            xs = jnp.array([i * N_sub for i in range(num_partitions)])
            occupancy, (idx, overflows) = lax.scan(
                scan_body, carry, xs, length=num_partitions
            )
            err = err.update(PEC.CELL_LIST_OVERFLOW, overflows.sum())
            idx = idx.transpose(1, 2, 0).reshape(2, -1)

            def prune(idx):
                receiver_idx, sender_idx = idx
                mask = (receiver_idx < sender_idx)

                out_idx = N * jnp.ones(receiver_idx.shape, jnp.int32)

                cumsum = jnp.cumsum(mask)
                index = jnp.where(mask, cumsum - 1,
                                  considered_neighbors * N - 1)
                receiver_idx = out_idx.at[index].set(receiver_idx)
                sender_idx = out_idx.at[index].set(sender_idx)
                occupancy = cumsum[-1]
                #occupancy = occupancy // 2
                return jnp.stack((receiver_idx, sender_idx)), occupancy

            # sort to enable pruning later
            ordering = jnp.argsort(idx[1])
            idx = idx[:, ordering]
            #if format is NeighborListFormat.OrderedSparse:
            #    idx, occupancy = prune(idx)

            if max_occupancy is None:
                _extra_capacity = N * extra_capacity
                max_occupancy = int(
                    occupancy * capacity_multiplier + _extra_capacity)
                if max_occupancy > idx.shape[-1]:
                    max_occupancy = idx.shape[-1]
                if not is_sparse(format):
                    capacity_limit = N - 1 if mask_self else N
                elif format is NeighborListFormat.Sparse:
                    capacity_limit = N * (N - 1) if mask_self else N**2
                else:
                    capacity_limit = N * (N - 1) // 2
                if max_occupancy > capacity_limit:
                    max_occupancy = capacity_limit
            idx = idx[:, :max_occupancy]
            update_fn = neighbor_list_fn if neighbors is None else neighbors.update_fn
            return NeighborList(
                idx,
                position,
                err.update(PEC.NEIGHBOR_LIST_OVERFLOW,
                           occupancy > max_occupancy),
                cl_capacity,
                max_occupancy,
                format,
                cell_size,
                cl_fn,
                update_fn,
            )  # pytype: disable=wrong-arg-count

        nbrs = neighbors
        if nbrs is None:
            return neighbor_fn((position, PartitionError(jnp.zeros((), jnp.uint8))))

        neighbor_fn = partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        d = partial(metric_sq, **kwargs)
        d = jax.vmap(d)

        return lax.cond(
            jnp.any(d(position, nbrs.reference_position) > threshold_sq),
            (position, nbrs.error),
            neighbor_fn,
            nbrs,
            lambda x: x,
        )

    def allocate_fn(
        position: jnp.ndarray, extra_capacity: int = 0, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

    @jit
    def update_fn(
        position: jnp.ndarray, neighbors: NeighborList, **kwargs
    ) -> NeighborList:
        return neighbor_list_fn(position, neighbors, **kwargs)

    # pytype: disable=wrong-arg-count
    return NeighborListFns(allocate_fn, update_fn)
