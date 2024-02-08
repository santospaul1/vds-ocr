import cv2
from PIL import Image
import pytesseract

# Path to the Tesseract executable (update this according to your system)
#pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'

# Load the pre-trained Haar Cascade classifier for vehicle detection
# Specify the full path to the haarcascade_car.xml file
cascade_path = 'haarcascade_car.xml'
vehicle_cascade = cv2.CascadeClassifier(cascade_path)


# Function to perform OCR on an image
def perform_ocr(image):
    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    print("Extracted Text:")
    print(text)

    return text  # Return the extracted text


# Open the video file (replace 'video.mp4' with your video file)
cap = cv2.VideoCapture('x.mp4')

# Open the file to write extracted text
output_file = open('one.txt', 'w')

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform vehicle detection
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected vehicles and extract number plates
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the region of interest (number plate area)
        roi = frame[y:y + h, x:x + w]

        # Convert the cropped region to PIL Image format
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Perform OCR on the cropped region and get the extracted text
        text = perform_ocr(pil_image)

        # Write the extracted text to the file
        output_file.write("Number Plate: {}\n".format(text))

        # Display the extracted number plate text on the frame
        cv2.putText(frame, "Number Plate: {}".format(text), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Vehicle Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Close the output file
output_file.close()
