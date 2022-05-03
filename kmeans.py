
grayImage = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

img_grey = cv2.imread('./data/2+_1_0_200_200_400.png', cv2.IMREAD_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(img_grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

thresh = 127
im_bw = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

print("im_bw",im_bw.shape)

plt.figure()
plt.imshow(im_bw, cmap="gray")
plt.axis("off") # Oculta los ejes
plt.show() # Muestra la imagen

#img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

#print(img_binary.shape)

#opencv_rgb_img = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)

cv2.imwrite('./data/bkac1.png',opencv_rgb_img)


image = imread(os.path.join("images","../data/bkac1.png"))
image.shape
print("Image",image.shape)
print("Image",image)

x1 = image.reshape(-1, 3)

#print("x1", x1)

kmeans = KMeans(n_clusters=2, random_state=32).fit(x1)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]

plt.figure()
plt.imshow(segmented_img)
plt.axis("off") # Oculta los ejes
plt.show() # Muestra la imagen