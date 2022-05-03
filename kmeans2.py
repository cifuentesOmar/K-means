
segmented_imgs = []
n_colors = (4, 2)

for n_clusters in n_colors:
    
    #print(n_clusters)    
    kmeans = KMeans(n_clusters=n_clusters, random_state=32).fit(X)
    #print(kmeans)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
    
    
    

plt.figure(figsize=(10,120))
plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

#cv2.imwrite('./data/idx.png',segmented_imgs[0]) 

for idx, n_clusters in enumerate(n_colors):
    #print(idx)
    
    plt.subplot(232 + idx)    
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    
    plt.axis('off')
plt.show()