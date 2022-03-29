import numpy as np
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float64MultiArray


def homographyToMsg(G: np.array) -> Float64MultiArray:
    if (G.shape != (3, 3)):
        raise ValueError("Expected (3, 3) Grix, got {}".forG(G.shape))
    msg = Float64MultiArray(
        layout=MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="rows", size=G.shape[0]),
                MultiArrayDimension(label="cols", size=G.shape[1])
            ],
            data_offset=0
        ),
        data=G.flatten().tolist()
    )

    return msg

if __name__ == "__main__":
    G = np.eye(3, 3)
    msg = homographyToMsg(G)
    print(msg)
