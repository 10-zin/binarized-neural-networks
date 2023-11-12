import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Sample random images for each label
num_samples_per_class = 10
samples = {label: [] for label in range(10)}

for image, label in mnist_trainset:
    if len(samples[label]) < num_samples_per_class:
        samples[label].append(image)
    if all(len(samples[label]) == num_samples_per_class for label in range(10)):
        break


sample_images = torch.cat([torch.stack(samples[label]) for label in range(10)])
sample_labels = torch.tensor([label for label in range(10) for _ in range(num_samples_per_class)])

# Flatten the images for clustering
sample_images_flat = sample_images.view(sample_images.shape[0], -1)

# Visualize the sampled MNIST images

fig, axes = plt.subplots(10, num_samples_per_class, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i in range(10):
    for j in range(num_samples_per_class):
        ax = axes[i, j]
        ax.imshow(samples[i][j].squeeze(), cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.set_title(f"Label: {i}")
plt.savefig("mnist.png")
# plt.show()

# WORKING 
# --------------------------------------------------------------------------------
# Reduce dimensionality to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
sample_images_3d = tsne.fit_transform(sample_images_flat)

# Labels for each group of 10 contiguous elements (0-9)
labels = np.repeat(np.arange(10), 10)

# Create a figure and a 3D subplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, )

# Extract the 3D coordinates
x = sample_images_3d[:, 0]
y = sample_images_3d[:, 1]
# z = sample_images_3d[:, 2]

# Plot each point and annotate it with its corresponding label
for i in range(len(x)):
    print(plt.cm.Set2(labels[i] % 10.))
    ax.scatter(x[i], y[i], marker='o', color=plt.cm.Set1((labels[i]) / 10.))
    ax.text(x[i], y[i], '%s' % (str(labels[i])), size=9, zorder=1, color='k')

ax.set_xlabel('TSNE Component 1')
ax.set_ylabel('TSNE Component 2')

plt.title('2D TSNE Visualization of Sample Images')
plt.savefig("2dtsne-mnist.png")
plt.show()

print(sample_images_3d.shape)
exit()
# --------------------------------------------------------------------------------

# # Perform k-means clustering
# kmeans = KMeans(n_clusters=9, random_state=42, n_init=30, init='k-means++', max_iter=500)
# cluster_labels = kmeans.fit_predict(sample_images_flat)

# """
# kmeans_cluster2_mnist_label = {}
# for i in range(0, 100, 10):
#     for j in range(i, i+10)
#         kmeans_label = get_max_freq_label(j) O(n)
#     kmeans_cluster2_mnist_label[kmeans_label]=i%10
# """



# ref = [(l.item(), p) for l, p in zip(sample_labels, cluster_labels)]
# print(ref)
# acc = sum(1 for l, p in zip(sample_labels, cluster_labels) if l.item()==p)/100
# print(acc)

# from collections import Counter

# cluster_to_digit = {}
# def convert_cluster_to_actual(labels_with_clusters):
#     # Step 1: Create a dictionary to hold lists of cluster labels for each actual digit
#     digit_to_clusters = {i: [] for i in range(10)}
    
#     for actual_digit, cluster_label in labels_with_clusters:
#         digit_to_clusters[actual_digit].append(cluster_label)
    
#     # Step 2: Find the most common cluster label for each actual digit
    
#     for digit, clusters in digit_to_clusters.items():
#         most_common_cluster = Counter(clusters).most_common(1)[0][0]
#         cluster_to_digit[most_common_cluster] = digit

#     # Step 3: Convert cluster labels to actual digit labels
#     corrected_labels = [(actual_digit, cluster_to_digit.get(cluster_label, 'Unknown')) 
#                         for actual_digit, cluster_label in labels_with_clusters]

#     return corrected_labels

# # Example usage:
# # labels_with_clusters = [(0, 4), (0, 4), ...]  # Add your full list here
# corrected_labels = convert_cluster_to_actual(ref)
# print(cluster_to_digit)
# print(corrected_labels)

# acc = sum(1 for l, p in corrected_labels if l==p)/100
# print(acc)

# annotations = [label[1] for label in corrected_labels]

# # Create the scatter plot
# plt.figure(figsize=(10, 8))
# for i in range(len(sample_images_3d)):
#     # print(sample_images_3d, sample_images_3d.shape)
#     # break
#     # print(sample_images_3d[i][0], sample_images_3d[i][1])
#     # print(i, annotations[i])
#     plt.scatter(sample_images_3d[i][0], sample_images_3d[i][1], color=plt.cm.Set1(annotations[i] / 10.))
#     plt.text(sample_images_3d[i][0], sample_images_3d[i][1], f'L:{corrected_labels[i][0]}, C:{corrected_labels[i][1]}', fontsize=9)

# # Title and labels
# plt.title('2D Graph of MNIST Digits with Corrected Labels')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# # Show the plot with legends
# plt.legend()
# plt.show()

# exit()


# # from sklearn.cluster import DBSCAN
# # clustering = DBSCAN(eps=0.5, min_samples=2).fit(sample_images_2d)
# # print(clustering.labels_)
# # exit()

# # Create a mapping from cluster label to most frequent true label in the cluster
# cluster_to_label_mapping = {}
# for cluster_idx in range(10):
#     # Find the most common label in each cluster
#     labels_in_cluster = sample_labels[cluster_labels == cluster_idx]
#     most_common_label = np.bincount(labels_in_cluster).argmax()
#     cluster_to_label_mapping[cluster_idx] = most_common_label

# # Plot the clusters with the correct visualization
# plt.figure(figsize=(12, 8))

# # Plot each cluster
# for cluster_idx in range(10):
#     # Find indices of points in this cluster
#     indices_in_cluster = cluster_labels == cluster_idx
#     correct_label = cluster_to_label_mapping[cluster_idx]
#     correct_indices = (sample_labels == correct_label) & indices_in_cluster
#     incorrect_indices = (sample_labels != correct_label) & indices_in_cluster

#     # Plot correct classifications
#     plt.scatter(sample_images_2d[correct_indices, 0], sample_images_2d[correct_indices, 1],
#                 color='green', label=f"Cluster {cluster_idx} (Label {correct_label})", alpha=0.6)

#     # Plot incorrect classifications
#     incorrect_points = plt.scatter(sample_images_2d[incorrect_indices, 0], sample_images_2d[incorrect_indices, 1],
#                                    color='red', alpha=0.6)

#     # Annotate the incorrectly classified points with their actual label
#     for index in np.where(incorrect_indices)[0]:
#         plt.annotate(sample_labels[index].item(),
#                      (sample_images_2d[index, 0], sample_images_2d[index, 1]),
#                      textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# # Title and labels
# plt.title('t-SNE visualization of MNIST with K-means clustering')
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')

# # Create legend for correct and incorrect classifications only
# handles, labels = plt.gca().get_legend_handles_labels()
# unique_labels = []
# unique_handles = []
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_labels.append(label)
#         unique_handles.append(handle)
# plt.legend(unique_handles, unique_labels, loc="best")

# plt.show()