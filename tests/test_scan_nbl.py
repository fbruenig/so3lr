import jax
import jax.numpy as jnp
import jax_md

import pathlib

from ase.io import read
import pytest

from jax_md import partition
from jax_md.space import DisplacementOrMetricFn, Box, raw_transform

from so3lr.scan_neighbor_list import scan_neighbor_list


print(f'Jax backend: {jax.default_backend()}')
# If you want to perform simulations in float64 you have to call this before any JAX compuation
jax.config.update('jax_enable_x64', False)

package_dir = pathlib.Path(__file__).parent.parent.resolve()
atoms = read(package_dir / 'tests/test_data/water_64.xyz')
atoms.wrap()

# Repeat in one direction.
atoms = atoms * [1, 1, 2]

@pytest.mark.parametrize(
    "cutoff",
    [3.0, 6.0],
)
@pytest.mark.parametrize(
    "capacity_multiplier",
    [1.2],
)
@pytest.mark.parametrize(
    "buffer_size_multiplier",
    [1.2],
)
@pytest.mark.parametrize(
    "minimum_cell_size_multiplier",
    [1.0],
)
@pytest.mark.parametrize(
    "disable_cell_list",
    [True, False],
)
@pytest.mark.parametrize(
    "format",
    [partition.NeighborListFormat(1), partition.NeighborListFormat(2)],
)
@pytest.mark.parametrize(
    "fractional_coordinates",
    [True, False],
)
@pytest.mark.parametrize(
    "num_partitions",
    [8, 16],
)
def test_scan_neighbor_list(cutoff,
                           capacity_multiplier,
                           buffer_size_multiplier,
                           minimum_cell_size_multiplier,
                           disable_cell_list,
                           format,
                           fractional_coordinates,
                           num_partitions):
    """
    Test the scan neighbor list function.
    """

    box = jnp.diag(jnp.array(atoms.get_cell()))
    # Create displacement and shift functions for periodic boundary conditions
    displacement, shift = jax_md.space.periodic_general(
            box=box,
            fractional_coordinates=fractional_coordinates
    )
    positions = jnp.array(atoms.get_positions())
    # Convert positions to fractional coordinates
    if fractional_coordinates:
        inv_box = 1/box
        positions = raw_transform(inv_box, positions)

    def test_nbl(func, **kwargs):
        """
        Run a test on the neighbor list function.
        """
        # Create the neighbor list
        neighbor_fn = func(
            displacement,
            box,
            r_cutoff=cutoff,
            dr_threshold=0.0,
            capacity_multiplier=capacity_multiplier,
            buffer_size_multiplier=buffer_size_multiplier,
            minimum_cell_size_multiplier=minimum_cell_size_multiplier,
            disable_cell_list=disable_cell_list,
            format=format,
            **kwargs
        )

        # Allocate the neighbor list
        nbrs = neighbor_fn.allocate(positions)

        # Update the neighbor list with the initial positions
        nbrs = nbrs.update(positions)

        return nbrs


    nbrs_jax_md = test_nbl(partition.neighbor_list)

    nbrs_scan = test_nbl(scan_neighbor_list, num_partitions = num_partitions)

    assert not nbrs_jax_md.idx.shape == (0, 0)
    assert not nbrs_scan.idx.shape == (0, 0)
    assert not jnp.all(nbrs_jax_md == 0)
    assert jnp.all(nbrs_jax_md.idx.shape == nbrs_scan.idx.shape)
    assert jnp.all(nbrs_jax_md.idx == nbrs_scan.idx)
    assert jnp.all(nbrs_jax_md.reference_position == nbrs_scan.reference_position)

