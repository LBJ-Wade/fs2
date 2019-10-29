import fs._fs as c

def this_node():
    """
    Returns:
        this_node (int): MPI rank of this node
    """

    return c.comm_this_node()


def n_nodes():
    """
    Returns:
        n_nodes (int): number of MPI nodes.
    """

    return c.comm_n_nodes()
