import cv2
from cv2 import error as cv2_error
from wasabi import msg
def find_and_capture(max_index=5):
    """
    Find accessible camera indices up to max_index, capture an image from each available camera, and save the images.

    Parameters:
    max_index (int): The highest camera index to check. Default is 5.
    """
    for idx in range(max_index + 1):  # includes max_index
        cap = cv2.VideoCapture(idx)  # CAP_DSHOW helps on Windows
        try:
            if cap.isOpened():
                msg.text(f"✅ Camera available at index {idx}", color ="green")
                ret, frame = cap.read()
                if ret:
                    filename = f"camera_{idx}_capture.jpg"
                    cv2.imwrite(filename, frame)
                    msg.text(f"Image captured and saved as '{filename}'", color="green")
                else:
                    msg.text(f"Failed to capture image from index {idx}", color="red")
            else:
                msg.text(f"No camera at index {idx}",color="yellow")
        except cv2_error as e:
            msg.text(f"Encountered CV2 error: {str(e)} at index {idx}", color="red")
        except Exception as e:
            msg.text(f"Unexpected error: {str(e)} at index {idx}", color="red")
        finally:
            cap.release()


if __name__ == "__main__":
    find_and_capture()
