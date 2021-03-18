import numpy as np


def transform(new_frame: np.array):
    """
    Compute the homogenious transformation matrix from the current coordinate
    system into a new coordinate system.

    Given the pose of the new reference frame ``new_frame`` in the current
    reference frame, compute the homogenious transformation matrix from the
    current reference frame into the new reference frame.
    ``transform(new_frame)`` can, for example, be used to get the transformation
    from the corrdinate frame of the current link in a kinematic chain to the
    next link in a kinematic chain. Here, the next link's origin (``new_frame``)
    is specified relative to the current link's origin, i.e., in the current
    link's coordinate system.


    Parameters
    ----------
    new_frame: np.array
        The pose of the new coordinate system's origin. This is a 6-dimensional
        vector consisting of the origin's position and the frame's orientation
        (xyz Euler Angles): [x, y, z, alpha, beta, gamma].

    Returns
    -------
    transformation_matrix : np.ndarray
        A 4x4 matrix representing the homogenious transformation.


    Notes
    -----
    For performance reasons, it is better to sequentially apply multiple
    transformations to a vector (or set of vectors) than to first multiply a
    sequence of transformations and then apply them to a vector afterwards.

    """

    # Note: this is a naive implementation, but avoids a scipy dependency
    # if scipy does get introduced later, it can be substituted by
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html

    new_frame = np.asarray(new_frame)
    alpha, beta, gamma = - new_frame[3:]

    rot_x = np.array([
        [1, 0,              0],
        [0, np.cos(alpha),  np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])
    rot_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    rot_z = np.array([
        [np.cos(gamma),  np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    transform = np.eye(4)
    # Note: apply inverse rotation
    transform[:3, :3] = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    transform[:3, 3] = - new_frame[:3]

    return transform


def inverseTransform(old_frame):
    """
    Compute the homogenious transformation matrix from the current coordinate
    system into the old coordinate system.

    Given the pose of the current reference frame in the old reference frame
    ``old_frame``, compute the homogenious transformation matrix from the new
    reference frame into the old reference frame. For example,
    ``inverseTransform(camera_frame)`` can, be used to compute the
    transformation from a camera's coordinate frame to the world's coordinate
    frame assuming the camera frame's pose is given in the world's coordinate
    system.

    Parameters
    ----------
    old_frame: {np.array, None}
        The pose of the old coordinate system's origin. This is a 6-dimensional
        vector consisting of the origin's position and the frame's orientation
        (xyz Euler Angles): [x, y, z, alpha, beta, gamma].

    Returns
    -------
    transformation_matrix : np.ndarray
        A 4x4 matrix representing the homogenious transformation.


    Notes
    -----
    For performance reasons, it is better to sequentially apply multiple
    transformations to a vector (or set of vectors) than to first multiply a
    sequence of transformations and then apply them to a vector afterwards.

    """

    # Note: this is a naive implementation, but avoids a scipy dependency
    # if scipy does get introduced later, it can be substituted by
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html

    old_frame = np.asarray(old_frame)
    alpha, beta, gamma = old_frame[3:]

    rot_x = np.array([
        [1, 0,              0],
        [0, np.cos(alpha),  np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])
    rot_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    rot_z = np.array([
        [np.cos(gamma),  np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    transform = np.eye(4)
    transform[:3, :3] = np.matmul(rot_x, np.matmul(rot_y, rot_z))
    transform[:3, 3] = old_frame[:3]

    return transform


def transformBetween(old_frame: np.array, new_frame: np.array):
    """
    Compute the homogenious transformation matrix between two frames.

    ``transformBetween(old_frame, new_frame)`` computes the
    transformation from the corrdinate system with pose ``old_frame`` to
    the corrdinate system with pose ``new_frame`` where both origins are
    expressed in the same reference frame, e.g., the world's coordinate frame.
    For example, ``transformBetween(camera_frame, tool_frame)`` computes
    the transformation from a camera's coordinate system to the tool's
    coordinate system assuming the pose of both corrdinate frames is given in
    a shared world frame (or any other __shared__ frame of reference).

    Parameters
    ----------
    old_frame: np.array
        The pose of the old coordinate system's origin. This is a 6-dimensional
        vector consisting of the origin's position and the frame's orientation
        (xyz Euler Angles): [x, y, z, alpha, beta, gamma].
    new_frame: np.array
        The pose of the new coordinate system's origin. This is a 6-dimensional
        vector consisting of the origin's position and the frame's orientation
        (xyz Euler Angles): [x, y, z, alpha, beta, gamma].

    Returns
    -------
    transformation_matrix : np.ndarray
        A 4x4 matrix representing the homogenious transformation.

    Notes
    -----
    If the reference frame and ``old_frame`` are identical, use ``transform``
    instead.

    If the reference frame and ``new_frame`` are identical, use
    ``transformInverse`` instead.

    For performance reasons, it is better to sequentially apply multiple
    transformations to a vector than to first multiply a sequence of
    transformations and then apply them to a vector afterwards.

    """

    return np.matmul(transform(new_frame), inverseTransform(old_frame))


def homogenize(cartesian_vector: np.array):
    """
    Convert a vector from cartesian coordinates into homogenious coordinates.

    Parameters
    ----------
    cartesian_vector: np.array
        The vector to be converted.

    Returns
    -------
    homogenious_vector: np.array
        The vector in homogenious coordinates.

    """

    shape = cartesian_vector.shape
    homogenious_vector = np.ones((*shape[:-1], shape[-1] + 1))
    homogenious_vector[..., :-1] = cartesian_vector
    return homogenious_vector


def cartesianize(homogenious_vector: np.array):
    """
    Convert a vector from homogenious coordinates to cartesian coordinates.

    Parameters
    ----------
    homogenious_vector: np.array
        The vector in homogenious coordinates.

    Returns
    -------
    cartesian_vector: np.array
        The vector to be converted.

    """

    return homogenious_vector[..., :-1] / homogenious_vector[..., -1]
