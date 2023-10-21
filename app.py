from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import numpy as np

app = Flask(__name__)

def process_image(image_path):
    # Your provided code here...
    # ...
    img0 = cv2.imread(image_path)
    resized_image1 = cv2.resize(img0, (800, 600))
    resized_image = img0.copy()

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(gray_image, (7, 7), sigmaX=0, sigmaY=0)

    lower = 10  # Example lower intensity threshold
    upper = 51  # Example upper intensity threshold

    # Apply a binary threshold to get a mask of pixels within the specified range
    mask = cv2.inRange(img_blur, lower, upper)

    # Convert the mask to a binary image using cv2.threshold
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    #ret, thresh = cv2.threshold(img_blur, 73, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    # x1, y1, w1, h1 = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    rect_points = cv2.boxPoints(rect)
    rect_points = np.int0(rect_points)

    # Draw the rotated rectangle on the original image
    img_with_rect = resized_image.copy()  # Make a copy of the original image
    cv2.polylines(img_with_rect, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)

    pt_A = rect_points[0]
    pt_B = rect_points[1]
    pt_C = rect_points[2]
    pt_D = rect_points[3]

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Define input and output points
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply perspective transformation
    out = cv2.warpPerspective(resized_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # out1 = cv2.flip(out, 1)
    # # Convert the result to RGB for display
    # #out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotated_image1 = cv2.rotate(rotated_image , cv2.ROTATE_90_COUNTERCLOCKWISE)

    if(rect[2]>=49 and rect[2]<=90):
        out1 = cv2.flip(out, 1)
        rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #rotated_image = cv2.flip(rotated_image, 1)
        #print("case1")
    else:
        out1 = cv2.flip(out, 1)
        rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #print("case2")

    source = cv2.resize(rotated_image, (1267,1428))

    current_directory = os.getcwd()
    model_filename = "best.pt"
    model_filepath = os.path.join(current_directory, model_filename)
    model = YOLO(model_filepath)  # pretrained YOLOv8n model 'C:\\Windows\\System32\\runs\\detect\\train5\\weights\\best.pt'
    #source = cv2.imread(image_path)
    results = model(source) 

    for r in results:
        #  print(r.boxes.xyxy)
        coord_list = r.boxes.xyxy.tolist()
        #print(r.boxes.conf)
        #  print(r.boxes)
        conf_list=r.boxes.conf.tolist()
        #  print(r.boxes)
# print(type(coord_list[0][0]))
# print(len(coord_list))

    image = np.copy(source)

    rice_type=[]
    chalkiness=[]
    pos=[]
    ar=[]
    length=[]
    breadth=[]

    filtered_coords = []

    # Iterate over the coordinates and corresponding confidence scores
    for coords, confidence in zip(coord_list, conf_list):
        if confidence > 0.5:
            filtered_coords.append(coords)

    mult = 0.0425
    # Loop through the bounding boxes
    for i, (startX, startY, endX, endY) in enumerate(filtered_coords):
    # Extract the region of interest (ROI)

        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        roi = image[startY:endY, startX:endX]
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), sigmaX=0, sigmaY=0)
        ret, thresh = cv2.threshold(img_blur, 73, 255, cv2.THRESH_BINARY)

        # Calculate proportion of white pixels
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        white_proportion = white_pixels / total_pixels

        if white_proportion >= 0.17:  # Adjust this threshold as needed
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the index of the largest contour
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]

                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                l1 = w1*mult
                b1 = h1*mult
                length.append(l1)
                breadth.append(b1)
                # Draw bounding box on original image
                cv2.rectangle(image, (startX + x1, startY + y1), (startX + x1 + w1, startY + y1 + h1), (0, 255, 0), 2)
                #cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                cv2.putText(image, str(i), (startX+x1, startY+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                aspect_ratio = float(w1) / h1
                ar.append(aspect_ratio)
                
                ret1, thresh1 = cv2.threshold(img_blur, 142, 255, cv2.THRESH_BINARY) #150
                
                white1 = cv2.countNonZero(thresh1)
                total1 = thresh1.size
                white_proportion1 = white1 / total1
                
                if aspect_ratio >= 3.5:
                    rice_type.append(f"Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pos.append(filtered_coords[i])
                    # cv2.drawContours(tempimg, [cnt], -1, (255, 0, 0), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                    # cv2.putText(tempimg, "chalky", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 2)  
                    #cv2.putText(tempimg, "Basmati", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        chalkiness.append(f"non-chalky_{i}") 
                else:
                    rice_type.append(f"Non-Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    pos.append(filtered_coords[i])
                    # cv2.drawContours(tempimg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                        # cv2.putText(tempimg, "chalky", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 2)
                    else:
                        chalkiness.append(f"non-chalky_{i}") 
                    

            else:
                #print("Not enough white pixels at position")
                pass
    total = sum(length)
    average = total / len(length)    
    return rice_type, chalkiness, pos, ar, image, resized_image1, average

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded file
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Save the uploaded file
        uploaded_file.save('static//uploads//uploaded_image.jpg')

        # Process the uploaded image
        rice_type, chalkiness, pos, ar, source, resized_image1, average = process_image('static//uploads//uploaded_image.jpg')
        
        # Save the processed image
        cv2.imwrite('static//resized_image1.jpg', resized_image1)
        cv2.imwrite('static//processed_image.jpg', source)  # Assuming 'source' is the processed image

        # Pass the lists to the result template
        return render_template('result.html', 
                               rice_type=rice_type, 
                               chalkiness=chalkiness, 
                               pos=pos, 
                               ar=ar,
                               average=average, 
                               resized_image1='static//resized_image1.jpg', 
                               processed_image='static//processed_image.jpg')
    else:
        return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)