import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def feature_selection(input_csv, flag):
    df = pd.read_csv(input_csv)
    feature_pixels = df['feature'].apply(lambda x: list(map(np.float32, x.split()))).to_list()
    label = df['label'].values
    file = df['file'].values
    feature_pixels = np.array(feature_pixels)
    print('feature_pixel:', feature_pixels.shape)
    # feature_PCA = PCA_selection(feature_pixels, components=200)
    # PCA_str = [' '.join(map(str, row)) for row in feature_PCA]
    # feature_LDA = LDA_selection(feature_pixels, label, flag)
    # LDA_str = [' '.join(map(str, row)) for row in feature_LDA]
    feature_Isomap = Isomap_selection(feature_pixels, n_components=100)
    Isomap_str = [' '.join(map(str, row)) for row in feature_Isomap]
    
    filename = os.path.splitext(os.path.basename(input_csv))[0]
    
    # df_pca = pd.DataFrame({'file':file, 'selected': PCA_str, 'label': label})
    # df_pca.to_csv(filename + '_pca.csv', index=False)
    
    # df_lda = pd.DataFrame({'file':file, 'selected': LDA_str, 'label': label})
    # df_lda.to_csv(filename + '_lda.csv', index=False)
    
    df_isomap = pd.DataFrame({'file':file, 'selected': Isomap_str, 'label': label})
    df_isomap.to_csv(filename + '_isomap.csv', index=False)
    # tsne_visualize(feature_PCA, feature_LDA, feature_Isomap, label, 0.15, flag)
    
def PCA_selection(feature_pixels, components):
    pca = PCA(n_components=components)
    feature_PCA = pca.fit_transform(feature_pixels)
    print('pca:', feature_PCA.shape)
    return feature_PCA
    
def LDA_selection(feature_pixels, label, flag):
    if (flag == 'fer2013' or flag == 'raf-db'):
        components = 6
    elif(flag == 'fer-plus'):
        components = 7
    lda = LDA(n_components=components)
    feature_LDA = lda.fit_transform(feature_pixels, label)
    print('lda', feature_LDA.shape)
    return feature_LDA

def Isomap_selection(feature_pixels, n_components=100):
    isomap = Isomap(n_neighbors=25, n_components=n_components)
    feature_isomap = isomap.fit_transform(feature_pixels)
    return feature_isomap

def tsne_visualize(feature_PCA, feature_LDA, feature_Isomap, label, sample_rate, flag):
    # encode the label to number so that to fit the scattering
    label = np.array(label)
    le = LabelEncoder()
    num_label = le.fit_transform(label)
    label_names = le.classes_
    
    # sample from feature_PCA, feature_LDA (in the same data point)
    df = pd.DataFrame({'label':num_label})
    sampled_index = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=sample_rate, random_state=42),
    ).index
    sampled_pca = feature_PCA[sampled_index]
    sampled_lda = feature_LDA[sampled_index]
    sampled_isomap = feature_Isomap[sampled_index]
    sampled_label = num_label[sampled_index]
    
    tsne1 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne3 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    pca_tsne = tsne1.fit_transform(sampled_pca)
    lda_tsne = tsne2.fit_transform(sampled_lda)
    isomap_tsne = tsne3.fit_transform(sampled_isomap)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    scatter1 = axs[0].scatter(pca_tsne[:, 0], pca_tsne[:, 1], c=sampled_label, cmap='tab10', s=10, alpha=0.7)
    axs[0].set_title('PCA visualization')
    axs[0].set_xlabel('PCA t-SNE component 1')
    axs[0].set_ylabel('PCA t-SNE component 2')
    
    scatter2 = axs[1].scatter(lda_tsne[:, 0], lda_tsne[:, 1], c=sampled_label, cmap='tab10', s=10, alpha=0.7)
    axs[1].set_title('LDA visualization')
    axs[1].set_xlabel('LDA t-SNE component 1')
    axs[1].set_ylabel('LDA t-SNE component 2')
    
    scatter3 = axs[2].scatter(isomap_tsne[:, 0], isomap_tsne[:, 1], c=sampled_label, cmap='tab10', s=10, alpha=0.7)
    axs[1].set_title('Isomap visualization')
    axs[1].set_xlabel('Isomap t-SNE component 1')
    axs[1].set_ylabel('Isomap t-SNE component 2')
    
    unique_labels = np.unique(num_label)
    colors = scatter1.cmap(scatter1.norm(unique_labels))
    handles = [mpatches.Patch(color=colors[i], label=label_names[i]) for i in range(len(unique_labels))]
    axs[1].legend(handles=handles, title='Classes', loc='upper left', bbox_to_anchor=(1.05, 1))
    if (flag == 'fer-plus'):
        title = 'FER-PLUS'
    elif (flag == 'raf-db'):
        title = 'RAF-DB'
    plt.suptitle(f'Dataset: {title}')
    plt.tight_layout()
    plt.show()
    
feature_selection('feature_csv\\fer_train_hog.csv', 'fer-plus')
# feature_selection('feature_csv\\fer_test_hog.csv', 'fer-plus')
# feature_selection('feature_csv\\rafdb_train_hog.csv', 'raf-db')
# feature_selection('feature_csv\\rafdb_test_hog.csv', 'raf-db')