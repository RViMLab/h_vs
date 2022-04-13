import numpy as np
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float64MultiArray


def mat3DToMsg(mat: np.array) -> Float64MultiArray:
    if (mat.shape != (3, 3)):
        raise ValueError(f"Expected (3, 3) matrix, got {mat.shape}")
    msg = Float64MultiArray(
        layout=MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="rows", size=mat.shape[0]),
                MultiArrayDimension(label="cols", size=mat.shape[1])
            ],
            data_offset=0
        ),
        data=mat.flatten().tolist()
    )

    return msg

def updateCroppedPrincipalPoint(top_left: np.ndarray, K: np.ndarray) -> np.ndarray:
    r"""Updates the camera's principal point under image cropping.
    Args:
        top_left (np.ndarray): Top left corner of cropped image
        K (np.ndarray): Intrinsic camera parameters
    Return:
        K_prime (np.ndarray): Updated intrinsic camera parameters (OpenCV convention)
    """
    K_prime = K
    K_prime[0,2] -= top_left[1]  # cx 
    K_prime[1,2] -= top_left[0]  # cy
    return K_prime


def updateScaledPrincipalPoint(src_shape: tuple, dst_shape: tuple, K: np.ndarray) -> np.ndarray:
    r"""Updates the camera's principal point under image scaling.
    Args:
        src_shape (tuple): Source shape HxWxC/HxW
        dst_shape (tuple): Destination shape HxWxC/HxW
        K (np.ndarray): Intrinsic camera parameters
    Return:
        K_prime (np.ndarray): Updated intrinsic camera parameters (OpenCV convention)
    """
    K_prime = K

    scale = np.divide(dst_shape[:2], src_shape[:2])
    K_prime[0,2] *= scale[1]  # cx
    K_prime[1,2] *= scale[0]  # cy
    return K_prime

if __name__ == "__main__":
    mat = np.eye(3, 3)
    msg = mat3DToMsg(mat)
    print(msg)
