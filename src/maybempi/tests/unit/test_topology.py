"""Groups, Cartesian topologies and derived datatypes of the serial stand-in."""

import numpy as np
import pytest

from maybempi import get_mpi
from maybempi.serial import SerialCartcomm, SerialGroup

MPI = get_mpi(False)
comm = MPI.COMM_WORLD


def test_groups():
    group = comm.Get_group()
    assert isinstance(group, MPI.Group) and comm.group.size == 1
    assert group.Get_size() == 1 and group.Get_rank() == 0 and group.rank == 0
    assert MPI.Group.Translate_ranks(group, [0], comm.Get_group()) == [0]
    assert group.Translate_ranks() == [0]
    assert group.Translate_ranks([0, MPI.PROC_NULL], MPI.GROUP_EMPTY) == [
        MPI.UNDEFINED,
        MPI.PROC_NULL,
    ]
    with pytest.raises(ValueError, match="not in a group"):
        group.Translate_ranks([1])
    empty = group.Excl([0])
    assert empty.Get_size() == 0 and empty.Get_rank() == MPI.UNDEFINED
    assert group.Incl([0]).Compare(group) == MPI.IDENT
    assert group.Incl([]).Compare(group) == MPI.UNEQUAL
    assert group.Excl([]).Get_size() == 1
    assert MPI.Group.Union(group, empty).Get_size() == 1
    assert MPI.Group.Intersection(group, empty).Get_size() == 0
    assert MPI.Group.Difference(group, empty).Get_size() == 1
    assert MPI.Group.Difference(group, group).Get_size() == 0
    assert group.Dup().Get_size() == 1 and group.Free() is None
    assert repr(empty) == "SerialGroup(size=0)" and not MPI.GROUP_NULL
    assert comm.Create(group).Get_size() == 1
    assert comm.Create_group(empty) is MPI.COMM_NULL


def test_cartesian_communicators():
    cart = comm.Create_cart([1, 1], periods=[True, False], reorder=True)
    assert isinstance(cart, MPI.Cartcomm) and isinstance(cart, MPI.Intracomm)
    assert cart.Get_topology() == MPI.CART and comm.Get_topology() == MPI.UNDEFINED
    assert comm.topology == MPI.UNDEFINED and cart.topology == MPI.CART
    assert cart.Get_topo() == ([1, 1], [1, 0], [0, 0]) == cart.topo
    assert cart.dims == [1, 1] and cart.periods == [1, 0] and cart.coords == [0, 0]
    assert cart.ndim == cart.Get_dim() == 2
    assert cart.Shift(0, 1) == (0, 0)  # periodic: its own neighbour
    assert cart.Shift(1, 1) == (MPI.PROC_NULL, MPI.PROC_NULL)
    assert cart.Shift(1, 0) == (0, 0)
    assert cart.Get_coords(0) == [0, 0]
    assert cart.Get_cart_rank([5, 0]) == 0  # wraps along the periodic axis
    with pytest.raises(ValueError, match="outside the grid"):
        cart.Get_cart_rank([0, 1])
    with pytest.raises(ValueError, match="2 dimensions"):
        cart.Get_cart_rank([0])
    with pytest.raises(ValueError, match="only rank 0"):
        cart.Get_coords(1)
    with pytest.raises(ValueError, match="direction 2"):
        cart.Shift(2, 1)
    sub = cart.Sub([False, True])
    assert sub.Get_topo() == ([1], [0], [0])
    with pytest.raises(ValueError, match="1 flags"):
        cart.Sub([True])
    assert isinstance(cart.Dup(), SerialCartcomm) and cart.Clone().dims == [1, 1]
    assert "dims=[1, 1]" in repr(cart)
    assert comm.Create_cart([1]).periods == [0]
    with pytest.raises(ValueError, match="more than 1 process"):
        comm.Create_cart([2, 1])
    with pytest.raises(ValueError, match="1 periods for 2"):
        comm.Create_cart([1, 1], [True])
    # messages work on the Cartesian communicator too
    cart.send(3, dest=cart.Shift(0, 1)[1])
    assert cart.recv(source=cart.Shift(0, -1)[0]) == 3


def test_split_type_and_compute_dims():
    assert comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() == 1
    assert comm.Split_type(MPI.UNDEFINED) is MPI.COMM_NULL
    assert MPI.Compute_dims(1, 3) == [1, 1, 1]
    assert MPI.Compute_dims(1, [0, 1]) == [1, 1]
    with pytest.raises(ValueError, match="1 process"):
        MPI.Compute_dims(4, 2)
    with pytest.raises(ValueError, match="do not fit"):
        MPI.Compute_dims(1, [2])
    assert MPI.Get_version() == (4, 0)
    assert MPI.THREAD_MULTIPLE == 3 and MPI.INFO_NULL is not None


def test_named_datatypes():
    assert MPI.DOUBLE.Get_size() == 8 and MPI.DOUBLE.size == 8
    assert MPI.INT32_T.size == 4 and MPI.BYTE.size == 1
    assert MPI.C_DOUBLE_COMPLEX.size == 16
    assert MPI.DOUBLE.Get_extent() == (0, 8) and MPI.DOUBLE.lb == 0
    assert MPI.DOUBLE.ub == MPI.DOUBLE.extent == 8
    assert MPI.DOUBLE.Get_name() == "DOUBLE"
    assert MPI.DOUBLE.Commit() is MPI.DOUBLE and MPI.DOUBLE.Free() is None
    copy = MPI.DOUBLE.Dup()
    assert copy is not MPI.DOUBLE and copy.size == 8
    for char, datatype in MPI._typedict.items():
        assert datatype.size == np.dtype(char).itemsize, char


def test_derived_datatypes():
    contiguous = MPI.DOUBLE.Create_contiguous(3)
    assert contiguous.size == 24 and contiguous.extent == 24
    vector = MPI.DOUBLE.Create_vector(3, 2, 4)
    assert vector.size == 48 and vector.Get_extent() == (0, 80)
    assert MPI.DOUBLE.Create_vector(0, 2, 4).extent == 0
    sub = MPI.DOUBLE.Create_subarray([4, 5], [2, 3], [1, 1], order=MPI.ORDER_C)
    assert sub.size == 48 and sub.Get_extent() == (0, 160)
    fortran = MPI.DOUBLE.Create_subarray([4, 5], [2, 3], [1, 1], order=MPI.ORDER_F)
    assert fortran.size == 48 and fortran.extent == 160
    nested = vector.Create_contiguous(2)
    assert nested.size == 96 and nested.extent == 160
    with pytest.raises(ValueError, match="non-negative"):
        MPI.DOUBLE.Create_contiguous(-1)
    with pytest.raises(ValueError, match="negative strides"):
        MPI.DOUBLE.Create_vector(2, 1, -1)
    with pytest.raises(ValueError, match="does not fit"):
        MPI.DOUBLE.Create_subarray([4], [3], [2])
    with pytest.raises(ValueError, match="same, nonzero length"):
        MPI.DOUBLE.Create_subarray([4, 4], [1], [0])
    with pytest.raises(ValueError, match="ORDER_C or ORDER_F"):
        MPI.DOUBLE.Create_subarray([4], [1], [0], order=7)


@pytest.mark.parametrize("order", ["C", "F"])
def test_derived_datatypes_select_the_elements_numpy_would(order):
    comm_ = MPI.COMM_SELF.Dup()
    grid = np.arange(60.0).reshape(3, 4, 5)
    if order == "F":
        grid = np.asfortranarray(grid)
    mpi_order = MPI.ORDER_C if order == "C" else MPI.ORDER_F
    block = MPI.DOUBLE.Create_subarray(
        [3, 4, 5], [2, 2, 3], [1, 1, 2], order=mpi_order
    ).Commit()
    flat = grid.reshape(-1, order="A")  # the memory order
    received = np.zeros(12)
    comm_.Send([flat, 1, block], dest=0)
    comm_.Recv(received)
    expected = grid[1:3, 1:3, 2:5].reshape(-1, order=order)
    np.testing.assert_array_equal(received, expected)
    vector = MPI.DOUBLE.Create_vector(3, 2, 4).Commit()
    comm_.Send([np.arange(12.0), 1, vector], dest=0)
    part = np.zeros(6)
    comm_.Recv(part)
    np.testing.assert_array_equal(part, [0, 1, 4, 5, 8, 9])
    comm_.Send([np.arange(20.0), 2, vector], dest=0)  # two copies, an extent apart
    two = np.zeros(12)
    comm_.Recv(two)
    np.testing.assert_array_equal(two, [0, 1, 4, 5, 8, 9, 10, 11, 14, 15, 18, 19])
    comm_.Send([np.arange(20.0), vector], dest=0)  # as many copies as fit
    comm_.Recv(two)
    np.testing.assert_array_equal(two[:6], [0, 1, 4, 5, 8, 9])


def test_group_and_empty_group_helpers():
    assert SerialGroup(0).Translate_ranks() == []
